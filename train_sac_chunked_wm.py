import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false') # Don't let JAX hog the GPU before torch inits
os.environ.setdefault('MUJOCO_GL', 'egl') # Headless for video rendering

import pathlib
import time
import elements
import jax
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import ogbench

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from sac_chunked.sac_chunk_agent import ChunkAgent
from helpers.sac_wm_utils import set_seed_everywhere
from sac_chunked.chunk_utils import real_chunk_transitions
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from sac_chunked.wm_diagnostics import wm_report, print_wm_report
from wm.dyna import success_seed_indices, imagine_transitions
from helpers.interop import numeric_metrics, unwrap
from helpers.ogbench_methods import OGBenchMethods
from helpers.online_replay import OnlineReplay

OBS_KEY = 'state'
ACTION_KEY = 'action'
ENV_ACTION_LOW = -1.0
ENV_ACTION_HIGH = 1.0

def load_config(folder, argv=None):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    return config

def build_agent_config(config, batch_size, seq_len, logdir):
    return elements.Config(
        **config.agent,
        logdir=str(logdir),
        seed=config.seed,
        jax=config.jax,
        batch_size=batch_size,
        batch_length=seq_len,
        replay_context=0,
        report_length=seq_len,
        replica=0,
        replicas=1,
    )

def build_real_env(env_name, load_offline_dataset):
    if load_offline_dataset:
        return OGBenchMethods.load_ogbench(env_name)
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    return env, None, None

def _param_norm(params):
    """ Norm over FLOATING-POINT leaves only. tree_leaves(params) also returns
        integer counters, and squaring those made the reported norm move for
        reasons unrelated to the weights. """
    leaves = [x for x in jax.tree_util.tree_leaves(params)
              if hasattr(x, 'dtype')
              and jax.numpy.issubdtype(x.dtype, jax.numpy.floating)]
    if not leaves:
        return float('nan')
    squares = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves]
    total = jax.numpy.sum(jax.numpy.stack(squares))
    return float(jax.device_get(total)) ** 0.5

def _prefixed(d, default_prefix):
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

def _wm_update(wm_agent, replay, batch_size, seq_len, rng, global_step):
    """ DreamerV3 world-model update on real replay sequences. The ONLY
        consumers of this model are the Dyna transition generator (extra
        critic training data) and the wm_report diagnostics -- the actor and
        the acting path never touch it. Co-trained so the model keeps
        tracking the state distribution the policy is currently visiting. """
    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
    batch = OGBenchMethods.to_jax(batch_np)
    batch.pop('discount', None)
    batch['seed'] = wm_agent._seeds(global_step, wm_agent.train_mirrored)
    wm_carry = wm_agent.init_train(batch_size)
    wm_carry, outs, wm_mets = wm_agent.train(wm_carry, batch)
    return wm_mets

def _agent_update(bridge, policy, replay, wm_batch, seq_len, chunk_config,
                  chunk_len, device, rng, gamma, gamma_h, metrics_on=True):
    """ QC-FQL with Dyna critic-batch augmentation.

        The ACTOR update is plain QC-FQL on real chunks -- untouched, real
        rows only. The world model changes exactly one thing: the CRITIC also
        trains on dyna_batch imagined chunk transitions per update,
        down-weighted by dyna_weight. Each imagined transition is

            (real obs s, chunk a ~ pi(s), model reward, model cont,
             decoded next obs)

        with target  r_hat + gamma^h * cont_hat * Q(s_hat_next).  The seed
        observation is REAL, so the critic's input distribution is anchored;
        only the chunk, the reward and the next state are counterfactual --
        which is the point: replay can only ever say what the executed chunk
        did, and this asks what a CURRENT-policy chunk would do from the same
        state.

        A dyna_success_frac share of seeds is drawn from replay windows that
        contain above-dyna_seed_thresh reward, placed 1..chunk_len steps
        before the reward moment, so imagined chunks overlap the rare signal.

        dyna_batch=0 skips the world model entirely -- this reduces to
        QC-FQL exactly. That is the control run. """
    batch_np = replay.sample_batch(wm_batch, seq_len, rng=rng)
    data = real_chunk_transitions(batch_np, chunk_len, gamma,
                                  obs_key=OBS_KEY, action_key=ACTION_KEY)
    if data is None or len(data['idx']) == 0:
        return None

    take = min(chunk_config.batch_size, len(data['idx']))
    sel = rng.choice(len(data['idx']), size=take, replace=False)

    to = lambda x: torch.as_tensor(x, device=device).float()
    obs = to(data['obs'][sel])
    next_obs = to(data['next_obs'][sel])
    chunk = to(data['chunk'][sel])
    real_reward = to(data['reward'][sel])
    real_mask = to(data['mask'][sel])
    valid = to(data['valid'][sel])
    step_valid = to(data['step_valid'][sel])

    dyna_n = int(chunk_config.dyna_batch)
    img = None
    succ_frac = 0.0
    with torch.no_grad():
        target = real_reward + gamma_h * real_mask * \
            policy.chunk_target_values(next_obs)

        if dyna_n > 0:
            obs_np = np.asarray(batch_np[OBS_KEY], dtype=np.float32)
            n_seq, t_len, obs_dim = obs_np.shape
            obs_flat = obs_np.reshape(-1, obs_dim)
            pool = bridge.seed_pool(OGBenchMethods.to_jax(batch_np), wm_batch)

            # Success-biased half: a separate biased sample so its windows are
            # GUARANTEED to contain above-threshold reward. Never fed to any
            # training loss directly -- it only donates seed latents.
            n_succ_want = int(round(dyna_n * chunk_config.dyna_success_frac))
            succ_idx = np.zeros((0,), dtype=np.int64)
            succ_pool, succ_obs_flat = None, None
            if n_succ_want > 0:
                bias_np = replay.sample_batch(
                    wm_batch, seq_len, rng=rng, bias_start_to_reward=True,
                    bias_reward_thresh=chunk_config.dyna_seed_thresh)
                succ_idx = success_seed_indices(
                    bias_np, n_succ_want, chunk_len, rng,
                    chunk_config.dyna_seed_thresh, obs_key=OBS_KEY)
                if len(succ_idx) > 0:
                    succ_pool = bridge.seed_pool(
                        OGBenchMethods.to_jax(bias_np), wm_batch)
                    succ_obs_flat = np.asarray(
                        bias_np[OBS_KEY], dtype=np.float32).reshape(-1, obs_dim)

            n_uni = dyna_n - len(succ_idx)
            uni_idx = rng.choice(n_seq * t_len, size=n_uni, replace=False) \
                .astype(np.int64)
            succ_frac = len(succ_idx) / max(dyna_n, 1)

            parts = [imagine_transitions(
                bridge, policy, pool, uni_idx, obs_flat, chunk_len, gamma,
                device, obs_key=OBS_KEY,
                reward_shift=chunk_config.reward_shift)]
            if len(succ_idx) > 0:
                parts.append(imagine_transitions(
                    bridge, policy, succ_pool, succ_idx, succ_obs_flat,
                    chunk_len, gamma, device, obs_key=OBS_KEY,
                    reward_shift=chunk_config.reward_shift))
            img = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            img_target = img['reward'] + gamma_h * img['cont'] * \
                policy.chunk_target_values(img['next_obs'])

    if img is not None:
        critic_obs = torch.cat([obs, img['obs']])
        critic_chunk = torch.cat([chunk, img['chunk']])
        critic_target = torch.cat([target, img_target])
        critic_weight = torch.cat(
            [valid, chunk_config.dyna_weight * torch.ones_like(img['reward'])])
    else:
        critic_obs, critic_chunk = obs, chunk
        critic_target, critic_weight = target, valid

    metrics = {}
    # n_real=take activates update_critic's per-source diagnostics
    # (critic_mse_real vs critic_mse_imagined) -- the poisoning alarm.
    metrics.update(_prefixed(policy.update_critic(
        critic_obs, critic_chunk, critic_target, critic_weight,
        n_real=take, metrics_on=metrics_on), 'sac'))
    # Actor is plain QC-FQL on REAL rows only: one batch drives distill/Q and
    # the flow-matching term, exactly as agents/acfql.py does. Imagined chunks
    # are not behavior data and never reach the actor.
    metrics.update(_prefixed(policy.update_actor(
        obs, torch.ones_like(valid), bc_feat=obs, bc_chunk=chunk,
        bc_valid=step_valid, metrics_on=metrics_on), 'sac'))
    policy.update_target()

    if not metrics_on:
        # chunk_diversity runs extra forward passes and every .item() below is
        # a blocking GPU sync -- both wasted on a step whose metrics are
        # discarded. Training math is unchanged; only reporting is skipped.
        return metrics

    metrics['sac/mean_chunk_reward'] = real_reward.mean().item()
    metrics['sac/mean_chunk_mask'] = real_mask.mean().item()
    metrics['sac/valid_frac'] = valid.mean().item()
    metrics['sac/chunk_diversity'] = policy.chunk_diversity(obs)
    metrics['diagnosis/batch_reward_max'] = real_reward.max().item()
    if img is not None:
        metrics['dyna/imagined_reward_mean'] = img['reward'].mean().item()
        # THE metric for this method: does imagination ever produce
        # above-floor reward? If this never rises above the floor, the model
        # is not generating signal and the run is QC-FQL plus noise.
        metrics['dyna/imagined_reward_max'] = img['reward'].max().item()
        metrics['dyna/imagined_cont_mean'] = img['cont'].mean().item()
        metrics['dyna/imagined_target_mean'] = img_target.mean().item()
        metrics['dyna/success_seed_frac'] = succ_frac
    return metrics

def train(config):
    general_config = config.train_sac_chunked_wm.general
    dreamer_config = config.train_sac_chunked_wm.dreamer
    chunk_config = config.train_sac_chunked_wm.chunk

    out_dir = pathlib.Path(general_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wm_batch = config.batch_size
    seq_len = config.batch_length
    chunk_len = chunk_config.chunk_len
    dyna_batch = chunk_config.dyna_batch
    gamma = chunk_config.gamma
    gamma_h = gamma ** chunk_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TF32 tensor cores for fp32 matmuls; off by default in torch. Large free
    # speedup on Ampere+, at a precision loss irrelevant beside the RSSM's own
    # sampling noise.
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device} | JAX devices: {jax.devices()}')
    if dyna_batch > 0:
        print(f'Dyna: critic trains on {chunk_config.batch_size} real + '
              f'{dyna_batch} imagined chunk transitions per update '
              f'(weight {chunk_config.dyna_weight}, '
              f'{chunk_config.dyna_success_frac:.0%} success-seeded above '
              f'reward {chunk_config.dyna_seed_thresh}) | actor and acting '
              f'are plain QC-FQL')
    else:
        print('dyna_batch=0: world model NOT used for critic data. This run '
              'is plain QC-FQL and should track the no-world-model arm.')
    print(f'wm report: {chunk_config.wm_diag_states} windows x '
          f'{chunk_config.wm_diag_samples} prior draws at depth '
          f'{chunk_config.wm_diag_depth}, offline against replay, 0 env steps')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(obs_dim, action_dim, OBS_KEY, ACTION_KEY)

    agent_config = build_agent_config(config, wm_batch, seq_len, out_dir / 'wm_ckpts')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)
    if general_config.wm_ckpt:
        print(f'Loading world model checkpoint: {general_config.wm_ckpt}')
        raw = np.load(general_config.wm_ckpt, allow_pickle=True)
        wm_agent.load({k: unwrap(raw[k]) for k in raw.files})
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)

    replay = OnlineReplay(obs_key=OBS_KEY, action_key=ACTION_KEY, max_episodes=dreamer_config.max_episodes)
    if train_dataset is not None:
        offline_episodes = OGBenchMethods.make_dreamer_episodes(
            train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
        replay.seed_from_offline(offline_episodes, rng=rng)
        print(f'Seeded replay buffer with {len(replay.offline_episodes)} offline episodes')

    # repr_dim is the RAW observation dim. The world model never sits between
    # the environment and the policy; it only manufactures extra critic
    # training data inside _agent_update.
    policy = ChunkAgent(
        repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
        compile_nets=chunk_config.compile_nets,
    )

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm='world_model_dyna',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    def run_eval(step, n_updates):
        results = eval_chunk_in_env(
            env, None, policy, action_dim, general_config.eval_episodes,
            device, OBS_KEY, chunk_len, eef_slice=eef_slice, record_video=True)
        print(f'step {step:7d} | return {results["mean_return"]:.2f} | '
              f'success {results["success_rate"]:.2f} | coherence {results["coherence"]:.4f}')
        eval_csv.append(step, n_updates, time.time() - start_time, results)
        log_dict = {
            'eval/mean_return': results['mean_return'],
            'eval/success_rate': results['success_rate'],
            'eval/coherence': results['coherence'],
            'eval/mean_episode_len': results['mean_episode_len'],
        }
        if chunk_config.wm_diag_states > 0 and dyna_batch > 0:
            wm_m = wm_report(
                bridge, replay, chunk_config, chunk_len,
                chunk_config.wm_diag_depth, gamma, device, rng, wm_batch,
                seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY,
                num_states=chunk_config.wm_diag_states,
                model_samples=chunk_config.wm_diag_samples)
            print_wm_report(wm_m, chunk_config.wm_diag_depth)
            # numeric_metrics ONLY on the diagnostics -- it does float(v) and
            # silently drops anything non-numeric, which would throw away the
            # wandb.Video object below. Filter here, never the whole log_dict.
            log_dict.update(numeric_metrics(wm_m))
        if results['video'] is not None:
            log_dict['eval/video'] = wandb.Video(results['video'], fps=20, format='mp4')
        wandb.log(log_dict, step=step)

    n_updates = 0
    offline_steps = general_config.num_offline_steps
    for i in range(1, offline_steps + 1):
        if not replay.ready(seq_len):
            continue
        metrics = {}
        if i % dreamer_config.train_every == 0:
            metrics.update(_prefixed(_wm_update(wm_agent, replay, wm_batch, seq_len, rng, i), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(bridge, policy, replay, wm_batch, seq_len,
                              chunk_config, chunk_len, device, rng, gamma, gamma_h,
                              metrics_on=(i % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1
        if metrics and i % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = _param_norm(wm_agent.params)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 0
            wandb.log(numeric_metrics(metrics), step=i)
        if i % general_config.eval_every == 0:
            run_eval(i, n_updates)
            env.reset()
    if offline_steps > 0:
        torch.save(policy.state_dict_all(), out_dir / 'chunk_offline.pt')
        print(f'Offline phase done: {n_updates} policy updates')

    obs, info = env.reset(seed=config.seed)
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    print('Starting online phase (QC-FQL acting, Dyna critic data)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        if global_step < general_config.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
            # The chunk is executed fully before the actor is queried again,
            # matching main.py's action_queue.
            if chunk_pos >= chunk_len:
                chunk_buffer = policy.act(state, eval_mode=False)
                chunk_pos = 0
            action = chunk_buffer[chunk_pos]
            chunk_pos += 1

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        replay.add_step(state, action, reward, np.asarray(next_obs, dtype=np.float32), terminated, truncated)

        done = bool(terminated or truncated)
        obs = next_obs
        if done:
            obs, info = env.reset()
            chunk_pos = chunk_len

        global_step += 1
        log_step = offline_steps + global_step
        metrics = {}
        ready = replay.ready(seq_len)

        if ready and global_step % dreamer_config.train_every == 0:
            metrics.update(_prefixed(
                _wm_update(wm_agent, replay, wm_batch, seq_len, rng, log_step), 'wm'))

        if (ready and global_step % chunk_config.train_every == 0
                and global_step >= general_config.start_training):
            m = _agent_update(bridge, policy, replay, wm_batch, seq_len,
                              chunk_config, chunk_len, device, rng, gamma, gamma_h,
                              metrics_on=(global_step % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1

        if metrics and global_step % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = _param_norm(wm_agent.params)
            metrics['diagnosis/replay_transitions'] = len(replay)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 1
            _succ = replay.success_stats
            metrics['replay/success_frac_total'] = _succ['total_frac']
            metrics['replay/success_frac_online'] = _succ['online_frac']
            metrics['replay/success_episodes_online'] = _succ['online_success']
            wandb.log(numeric_metrics(metrics), step=log_step)

        if global_step % general_config.eval_every == 0:
            run_eval(log_step, n_updates)
            obs, info = env.reset()
            chunk_pos = chunk_len

        if global_step % general_config.save_every == 0:
            torch.save(policy.state_dict_all(), out_dir / 'chunk_latest.pt')
            wm_cp = elements.Checkpoint(out_dir / 'wm_latest.pkl')
            wm_cp.agent = wm_agent
            wm_cp.save()

    torch.save(policy.state_dict_all(), out_dir / 'chunk_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac_chunked_wm.py --train_sac_chunked_wm.general.env_name=cube-triple-play-singletask-v0
# control run (world model unused for critic data):
# python train_sac_chunked_wm.py --train_sac_chunked_wm.chunk.dyna_batch=0