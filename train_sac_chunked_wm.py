import os
os.environ.setdefault('MUJOCO_GL', 'egl') # Headless for video rendering

import pathlib
import time
import elements
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import ogbench

from sac_chunked.sac_chunk_agent import ChunkAgent
from helpers.sac_wm_utils import set_seed_everywhere
from sac_chunked.chunk_utils import real_chunk_transitions
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from tdmpc.agent import TDMPC2Model
from tdmpc.data import latent_rollout_windows
from tdmpc.diagnostics import model_report, print_wm_report
from wm.chunk_selector import ChunkSelector
from helpers.interop import numeric_metrics
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

def build_real_env(env_name, load_offline_dataset):
    if load_offline_dataset:
        return OGBenchMethods.load_ogbench(env_name)
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    return env, None, None

def _prefixed(d, default_prefix):
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

def _model_update(model, replay, batch_size, seq_len, tdmpc_config, device, rng,
                  metrics_on=True):
    """ TD-MPC2 joint update on real replay windows: latent consistency,
        reward, value, then the policy prior.

        The ONLY consumers of this model are the ChunkSelector (chunk scoring)
        and the diagnostics -- the QC-FQL critic and actor never touch it.
        Co-trained so the model keeps tracking the state distribution the
        policy is currently visiting. """
    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
    w = latent_rollout_windows(batch_np, tdmpc_config.horizon,
                               obs_key=OBS_KEY, action_key=ACTION_KEY)
    if w is None or len(w['obs']) == 0:
        return {}
    take = min(tdmpc_config.batch_size, len(w['obs']))
    sel = rng.choice(len(w['obs']), size=take, replace=False)
    to = lambda x: torch.as_tensor(x[sel], device=device).float()
    return model.update(to(w['obs']), to(w['action']), to(w['reward']),
                        to(w['mask']), to(w['valid']), metrics_on=metrics_on)

def _agent_update(policy, replay, wm_batch, seq_len, chunk_config,
                  chunk_len, device, rng, gamma, gamma_h, metrics_on=True):
    """ Plain QC-FQL on real replay chunks. Deliberately takes no model
        argument: the training path CANNOT touch the latent model, by
        signature. The model influences this run only through which chunks the
        ChunkSelector chose to execute -- i.e. through the data.

            target = R_real + gamma^h * mask * Q(s_next) """
    batch_np = replay.sample_batch(wm_batch, seq_len, rng=rng)
    data = real_chunk_transitions(batch_np, chunk_len, gamma,
                                  obs_key=OBS_KEY, action_key=ACTION_KEY)
    if data is None or len(data['idx']) == 0:
        return None

    take = min(chunk_config.batch_size, len(data['idx']))
    sel = rng.choice(len(data['idx']), size=take, replace=False)

    to = lambda x: torch.as_tensor(x[sel], device=device).float()
    obs = to(data['obs'])
    next_obs = to(data['next_obs'])
    chunk = to(data['chunk'])
    real_reward = to(data['reward'])
    real_mask = to(data['mask'])
    valid = to(data['valid'])
    step_valid = to(data['step_valid'])

    with torch.no_grad():
        target = real_reward + gamma_h * real_mask * \
            policy.chunk_target_values(next_obs)

    metrics = {}
    metrics.update(_prefixed(policy.update_critic(
        obs, chunk, target, valid, metrics_on=metrics_on), 'sac'))
    # One batch drives distill/Q and the flow-matching term, exactly as
    # agents/acfql.py does.
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
    return metrics

def train(config):
    general_config = config.train_sac_chunked_wm.general
    tdmpc_config = config.train_sac_chunked_wm.tdmpc
    chunk_config = config.train_sac_chunked_wm.chunk

    out_dir = pathlib.Path(general_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wm_batch = config.batch_size
    seq_len = config.batch_length
    chunk_len = chunk_config.chunk_len
    select_n = chunk_config.select_n
    gamma = chunk_config.gamma
    gamma_h = gamma ** chunk_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TF32 tensor cores for fp32 matmuls; off by default in torch. Large free
    # speedup on Ampere+, at a precision loss irrelevant beside the model's
    # own prediction error.
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device}')
    if select_n > 1:
        print(f'Model-scored chunk selection: best-of-{select_n} at every '
              f'chunk boundary | training is plain QC-FQL, the TD-MPC2 model '
              f'only picks which chunk runs')
        print(f'  lookahead {chunk_config.select_rollout_chunks} chunk(s) = '
              f'{chunk_config.select_rollout_chunks * chunk_len} latent steps, '
              f'no decode anywhere')
    else:
        print('select_n<=1: latent model NOT used for action selection. This '
              'run is plain QC-FQL and should track the no-model arm.')
    print(f'wm report: {tdmpc_config.diag_windows} replay windows at depth '
          f'{tdmpc_config.diag_depth} chunk(s), offline, 0 env steps')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode,
               name=(general_config.run_name or None), config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay = OnlineReplay(obs_key=OBS_KEY, action_key=ACTION_KEY,
                          max_episodes=tdmpc_config.max_episodes)
    if train_dataset is not None:
        offline_episodes = OGBenchMethods.make_dreamer_episodes(
            train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
        replay.seed_from_offline(offline_episodes, rng=rng)
        print(f'Seeded replay buffer with {len(replay.offline_episodes)} offline episodes')

    # TD-MPC2 latent model: encoder -> latent, dynamics in latent, reward head
    # and Q both reading the latent. Nothing decodes back to observation
    # space. Scoring only.
    model = TDMPC2Model(obs_dim, action_dim, device, tdmpc_config, gamma)
    if general_config.model_ckpt:
        print(f'Loading latent model checkpoint: {general_config.model_ckpt}')
        model.load_state_dict_all(torch.load(general_config.model_ckpt, map_location=device))

    # repr_dim is the RAW observation dim. The latent model never sits between
    # the environment and the policy's INPUT; it only scores candidate chunks
    # inside the selector.
    policy = ChunkAgent(
        repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
        compile_nets=chunk_config.compile_nets,
    )
    selector = ChunkSelector(
        model, policy, action_dim, chunk_len, select_n, gamma, device,
        score_mode='model', rollout_chunks=chunk_config.select_rollout_chunks)

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm='latent_model_select',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    def run_eval(step, n_updates):
        results = eval_chunk_in_env(
            env, policy, general_config.eval_episodes, OBS_KEY, chunk_len,
            eef_slice=eef_slice, record_video=True, selector=selector)
        print(f'step {step:7d} | return {results["mean_return"]:.2f} | '
              f'success {results["success_rate"]:.2f} | coherence {results["coherence"]:.4f}')
        eval_csv.append(step, n_updates, time.time() - start_time, results)
        log_dict = {
            'eval/mean_return': results['mean_return'],
            'eval/success_rate': results['success_rate'],
            'eval/coherence': results['coherence'],
            'eval/mean_episode_len': results['mean_episode_len'],
        }
        if tdmpc_config.diag_windows > 0 and select_n > 1:
            wm_m = model_report(
                model, policy, replay, chunk_len, tdmpc_config.diag_depth,
                gamma, device, rng, wm_batch, seq_len, obs_key=OBS_KEY,
                action_key=ACTION_KEY, num_windows=tdmpc_config.diag_windows)
            print_wm_report(wm_m, tdmpc_config.diag_depth)
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
        if i % tdmpc_config.train_every == 0:
            metrics.update(_prefixed(_model_update(
                model, replay, wm_batch, seq_len, tdmpc_config, device, rng,
                metrics_on=(i % general_config.log_every == 0)), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(policy, replay, wm_batch, seq_len, chunk_config,
                              chunk_len, device, rng, gamma, gamma_h,
                              metrics_on=(i % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1
        if metrics and i % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = model.param_norm()
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
    print('Starting online phase (QC-FQL training, model-scored chunk selection)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        if global_step < general_config.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
            # The chunk is executed fully before the next decision, matching
            # main.py's action_queue. The selector chooses WHICH chunk runs;
            # the open-loop commitment length is unchanged.
            if chunk_pos >= chunk_len:
                chunk_buffer = selector.select(state)
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

        if ready and global_step % tdmpc_config.train_every == 0:
            metrics.update(_prefixed(_model_update(
                model, replay, wm_batch, seq_len, tdmpc_config, device, rng,
                metrics_on=(global_step % general_config.log_every == 0)), 'wm'))

        if (ready and global_step % chunk_config.train_every == 0
                and global_step >= general_config.start_training):
            m = _agent_update(policy, replay, wm_batch, seq_len, chunk_config,
                              chunk_len, device, rng, gamma, gamma_h,
                              metrics_on=(global_step % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1

        if metrics and global_step % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = model.param_norm()
            metrics['diagnosis/replay_transitions'] = len(replay)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 1
            metrics.update(selector.pop_stats())
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
            torch.save(model.state_dict_all(), out_dir / 'model_latest.pt')

    torch.save(policy.state_dict_all(), out_dir / 'chunk_final.pt')
    torch.save(model.state_dict_all(), out_dir / 'model_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac_chunked_wm.py --train_sac_chunked_wm.general.env_name=cube-triple-play-singletask-v0
# control run (latent model unused for selection)
# python train_sac_chunked_wm.py --train_sac_chunked_wm.chunk.select_n=1
