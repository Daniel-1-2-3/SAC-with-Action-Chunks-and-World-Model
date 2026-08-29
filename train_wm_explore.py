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
from sac_chunked.chunk_utils import real_chunk_transitions
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from sac_chunked.wm_diagnostics import wm_report, print_wm_report
from wm.explore import ExploreSelector
from wm.rnd import RNDNovelty
from helpers.interop import numeric_metrics, unwrap
from helpers.ogbench_methods import OGBenchMethods
from helpers.online_replay import OnlineReplay

ARM = 'explore'

def _agent_update(policy, replay, wm_batch, seq_len, chunk_config,
                  chunk_len, device, rng, gamma, gamma_h, metrics_on=True):
    """ Plain QC-FQL on real replay chunks. Deliberately takes no bridge and
        no world-model argument: the training path CANNOT touch the model, by
        signature. The world model influences this run only through which
        chunks the ExploreSelector chose to execute during COLLECTION -- i.e.
        through the data.

            target = R_real + gamma^h * mask * Q(s_next) """
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

    with torch.no_grad():
        target = real_reward + gamma_h * real_mask * \
            policy.chunk_target_values(next_obs)

    # v2 self-imitation BC: replace sil_frac of the flow-matching batch with
    # the top pooled-reward chunk windows from reward-bearing episodes, so
    # the behavior flow -- and through distillation the 16 candidates --
    # tracks the policy's best behavior instead of the buffer average.
    # ONLY the bc_* args change: the critic batch and the distill/Q batch
    # stay uniform, so value estimates keep covering ordinary states.
    # Inert (uniform) until the first reward-bearing episode exists.
    bc_feat, bc_chunk, bc_valid = obs, chunk, step_valid
    sil_frac = getattr(chunk_config, 'sil_frac', 0.0)
    sil_n_used = 0
    if sil_frac > 0.0:
        sil_batch = replay.sample_reward_batch(
            wm_batch, seq_len, chunk_config.sil_reward_thresh, rng=rng)
        if sil_batch is not None:
            sil_data = real_chunk_transitions(
                sil_batch, chunk_len, gamma, obs_key=OBS_KEY,
                action_key=ACTION_KEY)
            if sil_data is not None and len(sil_data['idx']) > 0:
                n_sil = min(int(round(take * sil_frac)), len(sil_data['idx']))
                # Top-k by pooled chunk reward: the best windows of the best
                # episodes. Degrades toward uniform when rewards are equal.
                top = np.argsort(sil_data['reward'][:, 0])[-n_sil:]
                n_uni = take - n_sil
                bc_feat = torch.cat([obs[:n_uni], to(sil_data['obs'][top])])
                bc_chunk = torch.cat([chunk[:n_uni], to(sil_data['chunk'][top])])
                bc_valid = torch.cat([step_valid[:n_uni],
                                      to(sil_data['step_valid'][top])])
                sil_n_used = n_sil

    metrics = {}
    metrics.update(prefixed(policy.update_critic(
        obs, chunk, target, valid, metrics_on=metrics_on), 'sac'))
    # One batch drives distill/Q and the flow-matching term, exactly as
    # agents/acfql.py does.
    metrics.update(prefixed(policy.update_actor(
        obs, torch.ones_like(valid), bc_feat=bc_feat, bc_chunk=bc_chunk,
        bc_valid=bc_valid, metrics_on=metrics_on), 'sac'))
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
    metrics['sac/sil_frac_actual'] = sil_n_used / max(take, 1)
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
    explore_n = chunk_config.explore_n
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
    if explore_n > 1 and chunk_config.explore_beta != 0.0:
        _nov = getattr(chunk_config, 'explore_novelty', 'draws')
        _sig = ('RND error at the imagined end state (self-annealing)'
                if _nov == 'rnd' else
                f'model disagreement over {chunk_config.explore_draws} draws')
        print(f'Exploration bonus: best-of-{explore_n} at every COLLECTION '
              f'chunk boundary, score = Q + {chunk_config.explore_beta} * '
              f'z({_sig}) | training is plain QC-FQL | eval acts with the '
              f'bare policy')
    elif explore_n > 1:
        print(f'explore_beta=0: critic-only best-of-{explore_n} (the QC '
              f'paper\'s method). World model unused for scoring -- this is '
              f'the bonus\'s control run.')
    else:
        print('explore_n<=1: no candidate selection at all. This run is '
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

    # repr_dim is the RAW observation dim. The world model never sits between
    # the environment and the policy's INPUT; it only supplies the novelty
    # term inside the collection-time selector.
    policy = ChunkAgent(
        repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
        compile_nets=chunk_config.compile_nets,
    )
    rnd = None
    if getattr(chunk_config, 'explore_novelty', 'draws') == 'rnd' \
            and chunk_config.explore_beta != 0.0 and explore_n > 1:
        rnd = RNDNovelty(
            device, hidden=chunk_config.rnd_hidden, out=chunk_config.rnd_out,
            lr=chunk_config.rnd_lr, buffer_size=chunk_config.rnd_buffer,
            batch=chunk_config.rnd_batch,
            train_every=chunk_config.rnd_train_every)
    selector = ExploreSelector(
        bridge, policy, action_dim, chunk_len, explore_n,
        chunk_config.explore_beta, chunk_config.explore_draws, device,
        novelty_mode=getattr(chunk_config, 'explore_novelty', 'draws'),
        norm_freeze=getattr(chunk_config, 'explore_norm_freeze', 0),
        rnd=rnd)

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm=f'world_model_{ARM}',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    def run_eval(step, n_updates):
        # Eval acts with the BARE policy: the exploration bonus is a
        # collection-time device, and eval must measure what was learned, not
        # the search wrapper.
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
        if chunk_config.wm_diag_states > 0 and explore_n > 1 \
                and chunk_config.explore_beta != 0.0:
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
            metrics.update(prefixed(wm_update(wm_agent, replay, wm_batch, seq_len, rng, i), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(policy, replay, wm_batch, seq_len, chunk_config,
                              chunk_len, device, rng, gamma, gamma_h,
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
    selector.reset()
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    print('Starting online phase (QC-FQL training, disagreement-bonus collection)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)
        selector.observe(state)

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
        selector.record_action(action)

        done = bool(terminated or truncated)
        obs = next_obs
        if done:
            obs, info = env.reset()
            selector.reset()
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
            m = _agent_update(policy, replay, wm_batch, seq_len, chunk_config,
                              chunk_len, device, rng, gamma, gamma_h,
                              metrics_on=(global_step % general_config.log_every == 0))
            if m is not None:
                metrics.update(m)
                n_updates += 1

        if metrics and global_step % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = param_norm(wm_agent.params)
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
            selector.reset()
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

# python train_wm_explore.py --train_sac_chunked_wm.general.env_name=cube-triple-play-singletask-v0
# controls:
#   critic-only best-of-N (QC's method, model unused):
#     python train_wm_explore.py --train_sac_chunked_wm.chunk.explore_beta=0
#   exact QC-FQL (no candidates at all):
#     python train_wm_explore.py --train_sac_chunked_wm.chunk.explore_n=1