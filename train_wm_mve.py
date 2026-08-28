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
from wm.imagination_latent import imagine_chunk_rollout_latent
from helpers.interop import numeric_metrics, unwrap, jax_to_torch
from helpers.ogbench_methods import OGBenchMethods
from helpers.online_replay import OnlineReplay

ARM = 'mve'

def _agent_update(bridge, policy, replay, wm_batch, seq_len, chunk_config,
                  chunk_len, num_chunks, device, rng, gamma, gamma_h,
                  mve_bootstrap='q', metrics_on=True):
    """ QC-FQL on LATENT FEATURES with a latent MVE target -- the original
        codebase's architecture, isolated. The evidence for this combination:
        the original run (latent bootstrap) climbed to ~-1700 return while the
        decoded-bootstrap variant sat dead at -3000. The decoder appears
        nowhere in this trainer.

        Everything the critic and actor consume is bridge.get_feat of a
        posterior latent of a REAL state; the world model changes exactly one
        thing beyond the representation: the bracketed continuation in
        target_Q.

            QC-FQL:  target = R_real + gamma^h * mask * Q(L_next)
            MVE:     target = R_real + gamma^h * mask *
                              [ R_1 + g*c_1 * [ ... + g*c_N * Q(L_N) ] ]

        Imagined chunks are sampled from the CURRENT actor at imagined
        features, so the continuation stays on-policy; L_N is a PRIOR latent
        (imagined), evaluated by a target critic trained on POSTERIOR latents.
        That distribution gap is this design's known open risk -- the price of
        having no decoder -- and diagnosis/critic_target_q_range is its gauge
        (the original run showed it compressed to ~130 vs qc-fql's ~270).

        num_chunks=0 skips imagination: target = QC-FQL's, computed on latent
        features. That control is NOT the no-world-model arm -- it isolates
        the REPRESENTATION. If it also caps below qc-fql, latents are the
        ceiling and no target formula fixes it; if it tracks qc-fql, the MVE
        comparison against it is clean. """
    batch_np = replay.sample_batch(wm_batch, seq_len, rng=rng)
    pool = bridge.seed_pool(OGBenchMethods.to_jax(batch_np), wm_batch)
    data = real_chunk_transitions(batch_np, chunk_len, gamma,
                                  obs_key=OBS_KEY, action_key=ACTION_KEY)
    if data is None or len(data['idx']) == 0:
        return None

    take = min(chunk_config.batch_size, len(data['idx']))
    sel = rng.choice(len(data['idx']), size=take, replace=False)

    # One place_seed for both the chunk-start latents (critic/actor input) and
    # the chunk-end latents (bootstrap / imagination seeds), so the model is
    # touched once for the real-side features.
    both_idx = np.concatenate([data['idx'][sel], data['next_idx'][sel]])
    both_carry = bridge.place_seed({k: v[both_idx] for k, v in pool.items()})
    both_feat = jax_to_torch(bridge.get_feat(both_carry), device)
    feat = both_feat[:take]
    next_feat = both_feat[take:]

    to = lambda x: torch.as_tensor(x, device=device).float()
    chunk = to(data['chunk'][sel])
    real_reward = to(data['reward'][sel])
    real_mask = to(data['mask'][sel])
    valid = to(data['valid'][sel])
    step_valid = to(data['step_valid'][sel])

    # Q(L_next) is QC-FQL's whole target. With MVE it is only needed to LOG
    # how far the imagined continuation moved things, so on a step whose
    # metrics are discarded it is not computed.
    lam = chunk_config.mve_lambda
    need_qc = (num_chunks == 0) or metrics_on
    img_rewards = img_conts = final_value = None
    with torch.no_grad():
        qc_value = policy.chunk_target_values(next_feat) if need_qc else None
        if num_chunks > 0:
            seed_carry = bridge.place_seed(
                {k: v[data['next_idx'][sel]] for k, v in pool.items()})
            img_rewards, img_conts, img_feats = imagine_chunk_rollout_latent(
                bridge, policy, seed_carry, num_chunks, chunk_len, device,
                gamma, reward_shift=chunk_config.reward_shift)
            if mve_bootstrap == 'none':
                # PURE-WM ablation: the critic appears NOWHERE in the target.
                # target = R_real + gamma^h*mask*[imagined rewards only], a
                # 55-step truncated return at num_chunks=10. Everything past
                # the horizon contributes zero, so targets live on a ~-130
                # scale instead of Q's ~-280 and the induced policy is
                # short-horizon by construction. This is an ATTRIBUTION run
                # (is the bootstrap load-bearing?), not a candidate method;
                # read the gap to the 'q' run, not the absolute curve.
                inter_values = None
                final_value = torch.zeros_like(img_rewards[-1])
            elif lam >= 1.0:
                # Pure nesting: only the deepest bootstrap appears in the
                # target, so the shallower ones are not worth their passes.
                inter_values = None
                final_value = policy.chunk_target_values(img_feats[-1])
            else:
                inter_values = torch.stack(
                    [policy.chunk_target_values(f) for f in img_feats])
                final_value = inter_values[-1]
            eff_lam = 1.0 if mve_bootstrap == 'none' else lam
            cont_value = mve_continuation(img_rewards, img_conts, final_value,
                                          gamma_h, eff_lam, inter_values)
        else:
            cont_value = qc_value
        target = real_reward + gamma_h * real_mask * cont_value
        # QC-FQL's own 1-chunk target, for comparison only. The critic is
        # never trained on it.
        qc_target = (real_reward + gamma_h * real_mask * qc_value
                     if need_qc else None)

    metrics = {}
    metrics.update(prefixed(policy.update_critic(
        feat, chunk, target, valid, metrics_on=metrics_on), 'sac'))
    # Actor is plain QC-FQL on the same latent batch: one selection drives
    # distill/Q and the flow-matching term, exactly as agents/acfql.py does.
    metrics.update(prefixed(policy.update_actor(
        feat, torch.ones_like(valid), bc_feat=feat, bc_chunk=chunk,
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
    metrics['sac/chunk_diversity'] = policy.chunk_diversity(feat)
    metrics['diagnosis/batch_reward_max'] = real_reward.max().item()

    # THE metric for this method: how far the imagined continuation moved the
    # target away from QC-FQL's (on the same latent features). Zero means the
    # model changed nothing and the run is latent QC-FQL with extra steps.
    metrics['mve/target_delta'] = (target - qc_target).abs().mean().item()
    metrics['mve/target_mean'] = target.mean().item()
    metrics['mve/qc_target_mean'] = qc_target.mean().item()
    # Spread of the target across the batch. The decoded-bootstrap failure
    # showed up as target FLATTENING (critic_target_q_range compressed), so
    # this is watched directly now.
    metrics['mve/target_std'] = target.std().item()
    metrics['mve/qc_target_std'] = qc_target.std().item()
    if num_chunks > 0:
        for k in range(num_chunks):
            metrics[f'mve/img_reward_chunk{k+1}'] = img_rewards[k].mean().item()
            metrics[f'mve/img_cont_chunk{k+1}'] = img_conts[k].mean().item()
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
    mve_bootstrap = chunk_config.mve_bootstrap
    if mve_bootstrap not in ('q', 'none'):
        raise ValueError(f'mve_bootstrap must be q or none, got {mve_bootstrap}')
    if mve_bootstrap == 'none' and num_chunks <= 0:
        raise ValueError('mve_bootstrap=none needs num_chunks>0: with no '
                         'imagination AND no bootstrap the target is just the '
                         'real chunk reward, which trains nothing useful')
    wm_online_every = dreamer_config.online_train_every
    wm_freeze_after = dreamer_config.wm_freeze_after

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TF32 tensor cores for fp32 matmuls; off by default in torch. Large free
    # speedup on Ampere+, at a precision loss irrelevant beside the RSSM's own
    # sampling noise.
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device} | JAX devices: {jax.devices()}')
    if num_chunks > 0 and mve_bootstrap == 'none':
        print(f'PURE-WM TARGET (ablation): real chunk + {num_chunks} imagined '
              f'chunks, NO critic bootstrap -- a {chunk_len * (num_chunks + 1)}'
              f'-step truncated return. Attribution run: read the gap to the '
              f'bootstrapped arm, not the absolute curve.')
    elif num_chunks > 0:
        print(f'Latent MVE: critic/actor on RSSM features, real chunk + '
              f'{num_chunks} imagined chunks -> latent bootstrap at '
              f'{chunk_len * (num_chunks + 1)} env steps | '
              f'lambda={chunk_config.mve_lambda} | no decoder anywhere')
    else:
        print('num_chunks=0: QC-FQL computed on LATENT features. This is the '
              'REPRESENTATION control, not the no-world-model arm -- compare '
              'it against qc-fql to see whether latents themselves cap '
              'performance.')
    if wm_online_every > 0 and wm_freeze_after > 0:
        print(f'WM schedule: every {dreamer_config.train_every} steps offline, '
              f'every {wm_online_every} online until total step '
              f'{wm_freeze_after}, then FROZEN -- the critic\'s input '
              f'representation is stationary from there on (the original run '
              f'trained it forever; its 3-8x critic loss was the suspected '
              f'cost)')
    elif wm_online_every > 0:
        print(f'WM schedule: every {dreamer_config.train_every} steps offline, '
              f'every {wm_online_every} online, never frozen')
    else:
        print(f'WM schedule: every {dreamer_config.train_every} steps offline, '
              f'FROZEN for the whole online phase')
    print(f'wm report: {chunk_config.wm_diag_states} windows x '
          f'{chunk_config.wm_diag_samples} prior draws at depth '
          f'{max(num_chunks, 1)}, offline against replay, 0 env steps')
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
    rssm_cfg = agent_config.dyn.rssm
    feat_dim = int(rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes)
    print(f'World model feature dim: {feat_dim}')

    replay = OnlineReplay(obs_key=OBS_KEY, action_key=ACTION_KEY, max_episodes=dreamer_config.max_episodes)
    if train_dataset is not None:
        offline_episodes = OGBenchMethods.make_dreamer_episodes(
            train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
        replay.seed_from_offline(offline_episodes, rng=rng)
        print(f'Seeded replay buffer with {len(replay.offline_episodes)} offline episodes')

    # repr_dim is the RSSM FEATURE dim: the critic, actor and BC flow all live
    # in latent space in this trainer. That is load-bearing for the latent
    # bootstrap Q(L_N) -- there is no observation-space critic here.
    policy = ChunkAgent(
        repr_dim=feat_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
        compile_nets=chunk_config.compile_nets,
    )

    arm_name = (f'world_model_{ARM}_latent_nobootstrap'
                if mve_bootstrap == 'none' else f'world_model_{ARM}_latent')
    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm=arm_name,
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    def run_eval(step, n_updates):
        # bridge is passed: eval encodes every step and the policy acts on
        # features -- the deployed behavior of this architecture.
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
        if chunk_config.wm_diag_states > 0 and num_chunks > 0:
            # Report depth = num_chunks: measure exactly the rollout the
            # target consumes.
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
        if (i % dreamer_config.train_every == 0
                and (wm_freeze_after <= 0 or i <= wm_freeze_after)):
            metrics.update(prefixed(wm_update(wm_agent, replay, wm_batch, seq_len, rng, i), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(bridge, policy, replay, wm_batch, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng,
                              gamma, gamma_h, mve_bootstrap=mve_bootstrap,
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
        if wm_online_every <= 0:
            print('World model FROZEN for the online phase.')
        elif 0 < wm_freeze_after <= offline_steps:
            print(f'World model already frozen (wm_freeze_after='
                  f'{wm_freeze_after} <= offline steps).')

    obs, info = env.reset(seed=config.seed)
    enc_carry, dyn_carry = bridge.init_encode(1)
    prevact = np.zeros((1, action_dim), dtype=np.float32)
    is_first = np.array([True])
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    print('Starting online phase (latent QC-FQL policy, latent MVE target)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        # The RSSM filters the posterior on EVERY step, including mid-chunk --
        # committing to actions is not a reason to stop looking. The policy's
        # input IS this feature; that is the architecture under test.
        enc_carry, dyn_carry, feat_j = bridge.encode_step(
            enc_carry, dyn_carry, state, prevact, is_first)
        is_first = np.array([False])

        if global_step < general_config.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
            # The chunk is executed fully before the actor is queried again,
            # matching main.py's action_queue.
            if chunk_pos >= chunk_len:
                feat_np = np.asarray(jax.device_get(feat_j))[0].copy()
                chunk_buffer = policy.act(feat_np, eval_mode=False)
                chunk_pos = 0
            action = chunk_buffer[chunk_pos]
            chunk_pos += 1

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        replay.add_step(state[0], action, reward, np.asarray(next_obs, dtype=np.float32), terminated, truncated)
        prevact = np.asarray(action, dtype=np.float32).reshape(1, -1)

        done = bool(terminated or truncated)
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

        if (ready and wm_online_every > 0
                and (wm_freeze_after <= 0 or log_step <= wm_freeze_after)
                and global_step % wm_online_every == 0):
            metrics.update(prefixed(
                wm_update(wm_agent, replay, wm_batch, seq_len, rng, log_step), 'wm'))
            if wm_freeze_after > 0 and log_step + wm_online_every > wm_freeze_after:
                print(f'World model FROZEN at total step {log_step}.')

        if (ready and global_step % chunk_config.train_every == 0
                and global_step >= general_config.start_training):
            m = _agent_update(bridge, policy, replay, wm_batch, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng,
                              gamma, gamma_h, mve_bootstrap=mve_bootstrap,
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

# python train_wm_mve.py --train_sac_chunked_wm.general.env_name=cube-triple-play-singletask-v0
# representation control (QC-FQL on latent features, no imagination):
# python train_wm_mve.py --train_sac_chunked_wm.chunk.num_chunks=0
# training without critic in the target computation
# python train_wm_mve.py --train_sac_chunked_wm.general.env_name=cube-triple-play-singletask-v0 --train_sac_chunked_wm.chunk.mve_bootstrap=none