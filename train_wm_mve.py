# Import order matters: trainer_common sets XLA/MUJOCO env vars at import
# time, so it must load before anything that imports jax.
from helpers.trainer_common import (
    load_config, build_agent_config, build_real_env, param_norm, prefixed,
    wm_update, OBS_KEY, ACTION_KEY, ENV_ACTION_LOW, ENV_ACTION_HIGH)

import pathlib
import time
import elements
import jax
import numpy as np
import torch
import wandb

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from sac_chunked.sac_chunk_agent import ChunkAgent
from helpers.sac_wm_utils import set_seed_everywhere
from sac_chunked.chunk_utils import real_chunk_transitions, mve_continuation
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from sac_chunked.wm_diagnostics import wm_report, print_wm_report
from wm.imagination_chunk import imagine_chunk_rollout
from helpers.interop import numeric_metrics, unwrap
from helpers.ogbench_methods import OGBenchMethods
from helpers.online_replay import OnlineReplay

ARM = 'mve'

def _agent_update(bridge, policy, replay, wm_batch, seq_len, chunk_config,
                  chunk_len, num_chunks, device, rng, gamma, gamma_h,
                  metrics_on=True):
    """ QC-FQL with a model-based value expansion (MVE) target.

        The world model is used in EXACTLY one place: the bracketed
        continuation inside target_Q. The actor, the critic, the batch, the BC
        flow term and the action selection are all plain QC-FQL on raw
        observations.

            QC-FQL:  target = R_real + gamma^h * mask * Q(s_next)
            MVE:     target = R_real + gamma^h * mask *
                              [ R_1 + g*c_1 * [ ... + g*c_N * Q(s_N) ] ]

        The imagined chunks start at the latent encoding s_next and are
        sampled from the CURRENT actor, so the continuation stays on-policy.
        Replay's own following actions cannot be used for this: they came from
        the policy that collected the data, so their rewards would estimate
        that old policy's return instead of the current one's.

        num_chunks=0 short-circuits the model entirely -- the bridge is never
        touched and this reduces to QC-FQL exactly. That is the control run. """
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

    # Q(s_next) is QC-FQL's whole target. With MVE it is not part of the
    # target at all -- it is only needed to LOG how far MVE moved things -- so
    # on a step whose metrics are discarded it is not computed.
    lam = chunk_config.mve_lambda
    need_qc = (num_chunks == 0) or metrics_on
    img_rewards = img_conts = final_value = None
    with torch.no_grad():
        qc_value = policy.chunk_target_values(next_obs) if need_qc else None
        if num_chunks > 0:
            pool = bridge.seed_pool(OGBenchMethods.to_jax(batch_np), wm_batch)
            seed_carry = bridge.place_seed(
                {k: v[data['next_idx'][sel]] for k, v in pool.items()})
            img_rewards, img_conts, img_obs = imagine_chunk_rollout(
                bridge, policy, seed_carry, num_chunks, chunk_len, device,
                gamma, obs_key=OBS_KEY, reward_shift=chunk_config.reward_shift)
            if lam >= 1.0:
                # Pure nesting: only the deepest bootstrap appears in the
                # target, so the shallower ones are not worth their forward
                # passes.
                inter_values = None
                final_value = policy.chunk_target_values(img_obs[-1])
            else:
                inter_values = torch.stack(
                    [policy.chunk_target_values(o) for o in img_obs])
                final_value = inter_values[-1]
            cont_value = mve_continuation(img_rewards, img_conts, final_value,
                                          gamma_h, lam, inter_values)
        else:
            cont_value = qc_value
        target = real_reward + gamma_h * real_mask * cont_value
        # QC-FQL's own 1-chunk target, for comparison only. The critic is
        # never trained on it.
        qc_target = (real_reward + gamma_h * real_mask * qc_value
                     if need_qc else None)

    metrics = {}
    metrics.update(prefixed(policy.update_critic(
        obs, chunk, target, valid, metrics_on=metrics_on), 'sac'))
    # Actor is plain QC-FQL: one batch drives distill/Q and the flow-matching
    # term, exactly as agents/acfql.py does.
    metrics.update(prefixed(policy.update_actor(
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

    # THE metric for this method: how far the imagined continuation moved the
    # target away from QC-FQL's. Zero means the model changed nothing and the
    # run is QC-FQL with extra steps.
    metrics['mve/target_delta'] = (target - qc_target).abs().mean().item()
    metrics['mve/target_mean'] = target.mean().item()
    metrics['mve/qc_target_mean'] = qc_target.mean().item()
    if num_chunks > 0:
        for k in range(num_chunks):
            metrics[f'mve/img_reward_chunk{k+1}'] = img_rewards[k].mean().item()
            metrics[f'mve/img_cont_chunk{k+1}'] = img_conts[k].mean().item()
        # Share of the target's magnitude that is the deepest bootstrap rather
        # than imagined reward. On a floor-dominated task this sits pinned
        # near gamma^(h*(N+1)) whatever the model does -- a consistency check
        # only, never evidence the model is contributing.
        deep = (gamma_h ** (num_chunks + 1)) * final_value.abs().mean()
        metrics['mve/bootstrap_share'] = (deep / target.abs().mean().clamp_min(1e-8)).item()
    return metrics

def train(config):
    general_config = config.train_sac_chunked_wm.general
    dreamer_config = config.train_sac_chunked_wm.dreamer
    chunk_config = config.train_sac_chunked_wm.chunk

    out_dir = pathlib.Path(general_config.out_dir) / ARM
    out_dir.mkdir(parents=True, exist_ok=True)

    wm_batch = config.batch_size
    seq_len = config.batch_length
    chunk_len = chunk_config.chunk_len
    num_chunks = chunk_config.num_chunks
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
    if num_chunks > 0:
        print(f'MVE: real chunk + {num_chunks} imagined chunks -> bootstrap at '
              f'{chunk_len * (num_chunks + 1)} env steps (QC-FQL: {chunk_len}) | '
              f'lambda={chunk_config.mve_lambda} alpha={chunk_config.alpha}')
    else:
        print('num_chunks=0: world model NOT used in the target. This run is '
              'plain QC-FQL and should track the no-world-model arm.')
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

    # repr_dim is the RAW observation dim. The world model does not sit
    # between the environment and the policy anywhere in this file; it is
    # consumed only inside the (no_grad) target computation.
    policy = ChunkAgent(
        repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
        compile_nets=chunk_config.compile_nets,
    )

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm=f'world_model_{ARM}',
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
        if chunk_config.wm_diag_states > 0 and num_chunks > 0:
            # Report depth = num_chunks: measure exactly the rollout the
            # target consumes, not the config's generic depth.
            wm_m = wm_report(
                bridge, replay, chunk_config, chunk_len,
                num_chunks, gamma, device, rng, wm_batch,
                seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY,
                num_states=chunk_config.wm_diag_states,
                model_samples=chunk_config.wm_diag_samples)
            print_wm_report(wm_m, num_chunks)
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
            metrics.update(prefixed(wm_update(wm_agent, replay, wm_batch, seq_len, rng, i), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(bridge, policy, replay, wm_batch, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng,
                              gamma, gamma_h,
                              metrics_on=(i % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1
        if metrics and i % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = param_norm(wm_agent.params)
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
    print('Starting online phase (QC-FQL policy, MVE target)')

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
            metrics.update(prefixed(
                wm_update(wm_agent, replay, wm_batch, seq_len, rng, log_step), 'wm'))

        if (ready and global_step % chunk_config.train_every == 0
                and global_step >= general_config.start_training):
            m = _agent_update(bridge, policy, replay, wm_batch, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng,
                              gamma, gamma_h,
                              metrics_on=(global_step % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1

        if metrics and global_step % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = param_norm(wm_agent.params)
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

# python train_wm_mve.py --train_sac_chunked_wm.general.env_name=cube-triple-play-singletask-v0
# control run (world model unused in the target):
# python train_wm_mve.py --train_sac_chunked_wm.chunk.num_chunks=0