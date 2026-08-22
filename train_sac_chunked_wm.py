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
from sac_chunked.chunk_utils import chunk_pair_indices, real_chunk_transitions, mve_target
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from wm.imagination_chunk import imagine_chunk_rollout
from helpers.interop import jax_to_torch, numeric_metrics, unwrap
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
    leaves = jax.tree_util.tree_leaves(params)
    squares = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves]
    total = jax.numpy.sum(jax.numpy.stack(squares))
    return float(jax.device_get(total)) ** 0.5

def _prefixed(d, default_prefix):
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

def _wm_update(wm_agent, replay, batch_size, seq_len, rng, global_step):
    """ DreamerV3 world-model update on real replay sequences. Co-trained with
        the policy so the model keeps tracking the state distribution the
        policy is currently visiting. """
    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
    batch = OGBenchMethods.to_jax(batch_np)
    batch.pop('discount', None)
    batch['seed'] = wm_agent._seeds(global_step, wm_agent.train_mirrored)
    wm_carry = wm_agent.init_train(batch_size)
    wm_carry, outs, wm_mets = wm_agent.train(wm_carry, batch)
    return wm_mets

def _agent_update(bridge, policy, replay, config_batch, seq_len, chunk_config,
                  chunk_len, num_chunks, device, rng, gamma, gamma_h):
    """ QC-FQL with a model-based value expansion (MVE) target.

        The critic and actor train on REAL states and REAL chunks only -- the
        batch is exactly what plain QC-FQL would use, and the actor loss is
        untouched. The world model changes one thing: how target_Q is computed.

        QC-FQL:  target = R_real + gamma^h * mask * Q(L_0)
        Here:    target = R_real + gamma^h * mask * [imagined continuation]

        where the continuation is num_chunks chunks sampled from the CURRENT
        actor and rolled through the model, starting from L_0 -- the latent at
        the state the real chunk ended at. Replay's own following actions
        cannot be used for this because they came from the policy that
        collected the data, so their rewards would estimate that old policy's
        return rather than the current one's. """
    seed_batch_np = replay.sample_batch(config_batch, seq_len, rng=rng)
    seed_batch = OGBenchMethods.to_jax(seed_batch_np)
    pool = bridge.seed_pool(seed_batch, config_batch)

    flat_idx, next_flat_idx, chunks_np, chunk_r_np, chunk_m_np, valid_np = \
        real_chunk_transitions(seed_batch_np, chunk_len, gamma, action_key=ACTION_KEY)
    if len(flat_idx) == 0:
        return None
    take = min(chunk_config.batch_size, len(flat_idx))
    sel = rng.choice(len(flat_idx), size=take, replace=False)

    # One place_seed for both the chunk-start latents (critic/actor input) and
    # the chunk-end latents (imagination seeds), so the model is touched once.
    both_idx = np.concatenate([flat_idx[sel], next_flat_idx[sel]])
    both_pool = {k: v[both_idx] for k, v in pool.items()}
    both_carry = bridge.place_seed(both_pool)
    both_feat = jax_to_torch(bridge.get_feat(both_carry), device)
    feat = both_feat[:take]

    to = lambda x: torch.as_tensor(x, device=device).float()
    chunk = to(chunks_np[sel])
    real_reward = to(chunk_r_np[sel])
    real_mask = to(chunk_m_np[sel])
    valid = to(valid_np[sel])

    # Imagination seeds: the chunk-END latents only.
    seed_carry = bridge.place_seed({k: v[next_flat_idx[sel]] for k, v in pool.items()})
    img_rewards, img_conts, img_next_feats, step_rewards = imagine_chunk_rollout(
        bridge, policy, seed_carry, num_chunks, chunk_len,
        device, gamma, reward_shift=chunk_config.reward_shift)

    with torch.no_grad():
        img_values = policy.chunk_target_values(img_next_feats)
        target = mve_target(real_reward, real_mask, img_rewards, img_conts,
                            img_values, gamma_h, chunk_config.mve_lambda, num_chunks)
        # QC-FQL's own 1-chunk target, logged for comparison only -- the
        # critic is never trained on it.
        qc_target = real_reward + gamma_h * real_mask * policy.chunk_target_values(
            both_feat[take:])

    metrics = {}
    metrics.update(_prefixed(policy.update_critic(feat, chunk, target, valid), 'sac'))
    # Actor is plain QC-FQL: same batch drives distill/Q and the flow-matching
    # term, exactly as agents/acfql.py does.
    bc_idx, bc_chunks_np, bc_valid_np = chunk_pair_indices(
        seed_batch_np, chunk_len, action_key=ACTION_KEY)
    if len(bc_idx) > 0:
        take_bc = min(chunk_config.bc_batch, len(bc_idx))
        sel_bc = rng.choice(len(bc_idx), size=take_bc, replace=False)
        bc_feat = jax_to_torch(bridge.get_feat(bridge.place_seed(
            {k: v[bc_idx[sel_bc]] for k, v in pool.items()})), device)
        bc_chunks = to(bc_chunks_np[sel_bc])
        bc_valid = to(bc_valid_np[sel_bc])
    else:
        bc_feat, bc_chunks, bc_valid = feat, chunk, None
    metrics.update(_prefixed(policy.update_actor(
        feat, torch.ones_like(valid), bc_feat=bc_feat,
        bc_chunk=bc_chunks, bc_valid=bc_valid), 'sac'))
    policy.update_target()

    metrics['sac/mean_chunk_reward'] = real_reward.mean().item()
    metrics['sac/mean_target'] = target.mean().item()
    metrics['sac/chunk_diversity'] = policy.chunk_diversity(feat)
    # THE metric for this method: how much the imagined continuation moved the
    # target away from QC-FQL's 1-chunk version. Near zero means MVE is doing
    # nothing and the run is just slower QC-FQL.
    metrics['diagnosis/mve_target_delta'] = (target - qc_target).abs().mean().item()
    metrics['diagnosis/qc_target_mean'] = qc_target.mean().item()
    metrics['diagnosis/real_reward_mean'] = real_reward.mean().item()
    metrics['diagnosis/real_valid_frac'] = valid.mean().item()

    ir = img_rewards.reshape(num_chunks, -1)
    ic = img_conts.reshape(num_chunks, -1)
    metrics['diagnosis/imagined_reward_mean'] = img_rewards.mean().item()
    metrics['diagnosis/imagined_reward_first_chunk'] = ir[0].mean().item()
    metrics['diagnosis/imagined_reward_last_chunk'] = ir[-1].mean().item()
    metrics['diagnosis/imagined_reward_max'] = img_rewards.max().item()
    metrics['diagnosis/imagined_cont_last_chunk'] = ic[-1].mean().item()
    metrics['diagnosis/intra_chunk_reward_first'] = step_rewards[0].mean().item()
    metrics['diagnosis/intra_chunk_reward_last'] = step_rewards[-1].mean().item()
    return metrics

def train(config):
    general_config = config.train_sac_chunked_wm.general
    dreamer_config = config.train_sac_chunked_wm.dreamer
    chunk_config = config.train_sac_chunked_wm.chunk

    out_dir = pathlib.Path(general_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config.batch_size
    seq_len = config.batch_length
    chunk_len = chunk_config.chunk_len
    num_chunks = chunk_config.num_chunks
    gamma = chunk_config.gamma
    gamma_h = gamma ** chunk_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device} | JAX devices: {jax.devices()}')
    print(f'MVE: real chunk + {num_chunks} imagined chunks -> bootstrap at '
          f'{chunk_len * (num_chunks + 1)} env steps (QC-FQL: {chunk_len}) | '
          f'lambda={chunk_config.mve_lambda} alpha={chunk_config.alpha}')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(obs_dim, action_dim, OBS_KEY, ACTION_KEY)

    agent_config = build_agent_config(config, batch_size, seq_len, out_dir / 'wm_ckpts')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)
    if general_config.wm_ckpt:
        print(f'Loading world model checkpoint: {general_config.wm_ckpt}')
        raw = np.load(general_config.wm_ckpt, allow_pickle=True)
        wm_agent.load({k: unwrap(raw[k]) for k in raw.files})
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)
    rssm_cfg = agent_config.dyn.rssm
    feat_dim = int(rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes)
    print(f'World model feature dim: {feat_dim}')

    replay = OnlineReplay(obs_key=OBS_KEY, action_key=ACTION_KEY, max_episodes=dreamer_config.max_episodes)
    if train_dataset is not None:
        offline_episodes = OGBenchMethods.make_dreamer_episodes(
            train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
        replay.seed_from_offline(offline_episodes, rng=rng)
        print(f'Seeded replay buffer with {len(replay.offline_episodes)} offline episodes')

    policy = ChunkAgent(
        repr_dim=feat_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
    )

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm='world_model_mve',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    def run_eval(step, n_updates):
        results = eval_chunk_in_env(
            env, bridge, policy, action_dim, general_config.eval_episodes,
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
            metrics.update(_prefixed(_wm_update(wm_agent, replay, batch_size, seq_len, rng, i), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(bridge, policy, replay, batch_size, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng, gamma, gamma_h)
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
    enc_carry, dyn_carry = bridge.init_encode(1)
    prevact = np.zeros((1, action_dim), dtype=np.float32)
    is_first = np.array([True])
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    print('Starting online phase (QC-FQL + MVE target)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        enc_carry, dyn_carry, feat_jax = bridge.encode_step(enc_carry, dyn_carry, state, prevact, is_first)

        if global_step < general_config.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
            # Execution stays at chunk_len -- the MVE target extends the
            # critic's horizon without extending the open-loop commitment.
            if chunk_pos >= chunk_len:
                feat_np = np.asarray(jax.device_get(feat_jax))[0].copy()
                chunk_buffer = policy.act(feat_np, eval_mode=False)
                chunk_pos = 0
            action = chunk_buffer[chunk_pos]
            chunk_pos += 1

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        replay.add_step(state[0], action, reward, np.asarray(next_obs, dtype=np.float32), terminated, truncated)

        done = bool(terminated or truncated)
        prevact = action.reshape(1, -1).astype(np.float32)
        is_first = np.array([False])
        obs = next_obs
        if done:
            obs, info = env.reset()
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])
            chunk_pos = chunk_len

        global_step += 1
        log_step = offline_steps + global_step
        metrics = {}
        ready = replay.ready(seq_len)

        if ready and global_step % dreamer_config.train_every == 0:
            metrics.update(_prefixed(
                _wm_update(wm_agent, replay, batch_size, seq_len, rng, log_step), 'wm'))

        if (ready and global_step % chunk_config.train_every == 0
                and global_step >= general_config.start_training):
            m = _agent_update(bridge, policy, replay, batch_size, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng, gamma, gamma_h)
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
            wandb.log(numeric_metrics(metrics), step=log_step)

        if global_step % general_config.eval_every == 0:
            run_eval(log_step, n_updates)
            obs, info = env.reset()
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])
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