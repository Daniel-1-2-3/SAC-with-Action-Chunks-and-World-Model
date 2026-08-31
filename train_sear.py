""" SEAR + Plan2Explore trainer: from scratch, single budget.

    Two agents, each faithful to its paper:
      task agent -- SEAR (Nagy et al. 2026): chunked MaxEnt policy, causal
                    transformer critic with multi-horizon targets, random
                    replanning at collection, receding-horizon eval.
                    Trained on real replay with environment reward. The
                    ONLY policy evaluated.
      explorer   -- Plan2Explore (Sekar et al. 2020): single-step actor and
                    value on latent model states, trained purely in
                    imagination to maximize ensemble disagreement
                    (expected future novelty). Executed in the environment
                    to collect informative real data.

    The integration seam (ours, and the only non-paper components):
    (1) the explorer collects (all or nearly all) episodes and SEAR learns
    off-policy from the shared per-step replay -- mirroring P2E, whose task
    policy also never collects; there is NO event-driven schedule; and
    (2) a small fraction of each training batch is drawn from
    reward-bearing episodes (biased sampling, no importance correction,
    fraction kept small), so rare found rewards are amplified instead of
    diluted in the growing buffer.

    use_world_model=False collapses to the SEAR-only baseline.

    Run:
      python train_sear.py --train_sear.env_name=cube-double-play-singletask-v0
"""
import os
# Headless rendering: MuJoCo defaults to GLFW, which needs X11 and core-dumps
# on display-less pods. EGL renders on the GPU without a display. Set before
# anything imports mujoco. Override with MUJOCO_GL=osmesa for CPU-only pods.
os.environ.setdefault('MUJOCO_GL', 'egl')
import pathlib
import sys

import numpy as np
import ogbench
import torch
import wandb

from explore.p2e_explorer import P2EExplorer
from helpers.common import (load_config, prefixed, set_seed_everywhere,
                            temporal_coherence)
from helpers.curriculum import Curriculum
from helpers.her import GoalTools
from helpers.sear_replay import EpisodeReplay
from sear.agent import SEARAgent
from sear.windows import build_windows
from torch_wm.ensemble import LatentDisagreementEnsemble
from torch_wm.world_model import WorldModel

OBS_KEY, ACTION_KEY = 'state', 'action'


def run_eval(env, agent, cfg, eef_slice, gc=lambda o: o):
    returns, lens, succ, coh = [], [], [], []
    frames = []
    for ep_i in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        chunk, pos = None, cfg.chunk_len
        eefs = []
        while not done and steps < cfg.eval_max_steps:
            if pos >= cfg.eval_receding:
                chunk = agent.act(gc(obs), deterministic=True)
                pos = 0
            a = chunk[pos]
            pos += 1
            obs, r, term, trunc, info = env.step(a)
            if ep_i == 0 and steps % 2 == 0:
                try:
                    f = env.render()
                    if f is not None:
                        frames.append(f)
                except Exception:
                    pass
            eefs.append(obs[eef_slice[0]:eef_slice[1]])
            ep_ret += r
            steps += 1
            done = term or trunc
        returns.append(ep_ret)
        lens.append(steps)
        succ.append(float(info.get('success', 0.0)))
        coh.append(temporal_coherence(np.asarray(eefs)))
    metrics = {'mean_return': float(np.mean(returns)),
               'success_rate': float(np.mean(succ)),
               'mean_episode_len': float(np.mean(lens)),
               'coherence': float(np.mean(coh))}
    return metrics, frames


def feat_to_carry(wm, feats):
    """ Reconstruct RSSM carries from flat features (feat = deter ++ stoch),
        for imagination start states per P2E: 'latent states obtained by
        encoding [observations] from the replay buffer'. """
    deter = feats[:, :wm.deter]
    stoch = feats[:, wm.deter:].reshape(-1, wm.stoch, wm.classes)
    return dict(deter=deter, stoch=stoch)


def train(config):
    cfg = config.train_sear
    set_seed_everywhere(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(config.seed)
    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode,
               config=dict(config))

    try:
        env = ogbench.make_env_and_datasets(
            cfg.env_name, env_only=True, render_mode='rgb_array')
    except TypeError:
        env = ogbench.make_env_and_datasets(cfg.env_name, env_only=True)
    obs, _ = env.reset(seed=config.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    N = cfg.chunk_len
    goal_tools = GoalTools(env, thresh=cfg.her_thresh) if cfg.use_her \
        else None
    goal_dim = goal_tools.goal_dim if goal_tools else 0
    if goal_tools is not None:
        goal_tools.calibrate(obs)      # solve obs-space center first
    task_goal = goal_tools.task_goal() if goal_tools else None
    replay = None  # forward decl for set_goal below
    gc = (lambda o: np.concatenate([o, task_goal]).astype(np.float32)) \
        if goal_tools else (lambda o: o)

    curriculum = Curriculum(
        env, spawn_frac=cfg.spawn_frac, pool_frac=cfg.reset_pool_frac,
        pool_size=cfg.reset_pool_size,
        reward_thresh=cfg.success_reward_thresh, rng=rng)

    replay = EpisodeReplay(OBS_KEY, ACTION_KEY, cfg.max_episodes)
    replay.set_goal(task_goal)
    task = SEARAgent(obs_dim + goal_dim, act_dim, N, gamma=cfg.gamma,
                     device=device, alpha_min=cfg.alpha_min,
                     critic_kw=dict(vmin=cfg.critic_vmin,
                                    vmax=cfg.critic_vmax))
    wm = ensemble = explorer = None
    if cfg.use_world_model:
        wm = WorldModel(obs_dim, act_dim, device=device, lr=cfg.wm_lr,
                        use_compile=cfg.wm_compile)
        ensemble = LatentDisagreementEnsemble(
            wm.feat_dim, act_dim, wm.embed_dim, k=cfg.ensemble_k,
            device=device)
        if cfg.explore_frac > 0:
            explorer = P2EExplorer(wm.feat_dim, act_dim,
                                   horizon=cfg.imagine_horizon,
                                   gamma=cfg.gamma,
                                   reward_mix=cfg.explore_reward_mix,
                                   device=device)

    mode = ('explorer=P2E (latent disagreement, imagination-trained)'
            if explorer else
            ('passive WM (SEAR collects; model trains for eval planning)'
             if wm is not None else 'NO world model (SEAR-only baseline)'))
    print(f'sear-p2e | task=SEAR(chunk {N}) | {mode} | '
          f'eval=task policy, receding horizon {cfg.eval_receding}')

    trigger_step = None
    acting = 'explorer' if explorer else 'task'
    chunk, pos, prefix_len = None, N, N        # task-policy chunk state
    ex_carry, prev_a, ep_start = None, None, True   # explorer latent state
    ep_steps = 0
    metrics_acc = {}

    def reset_episode_state():
        nonlocal chunk, pos, prefix_len, ex_carry, prev_a, ep_start, \
            ep_steps
        chunk, pos, prefix_len = None, N, 0
        ex_carry = wm.init(1) if (explorer and wm) else None
        prev_a = np.zeros(act_dim, np.float32)
        ep_start, ep_steps = True, 0

    reset_episode_state()
    for step in range(1, cfg.num_online_steps + 1):
        # ---------------- collection ----------------
        if step <= cfg.num_seed_steps:
            a = rng.uniform(-1, 1, act_dim).astype(np.float32)
        elif acting == 'explorer':
            ex_carry, feat = wm.encode_step(
                ex_carry, obs[None], prev_a[None],
                np.asarray([ep_start], bool))
            a = explorer.act(feat)[0]
        else:
            # SEAR random replanning (Sec 4.4): 'we only execute a random
            # prefix of each chunk' -- draw a fresh chunk, run a uniform
            # random prefix length of it, replan.
            if pos >= prefix_len:
                chunk = task.act(gc(obs), deterministic=False)
                prefix_len = int(rng.integers(1, N + 1))
                pos = 0
            a = chunk[pos]
            pos += 1
        ach = goal_tools.achieved() if goal_tools else None
        next_obs, r, term, trunc, _ = env.step(a)
        next_ach = goal_tools.achieved() if goal_tools else None
        ep_steps += 1
        ep_start = False
        prev_a = a
        if ep_steps >= cfg.max_episode_steps:
            trunc = True
        curriculum.maybe_pool(r)
        replay.add_step(obs, a, r, next_obs, term, trunc,
                        achieved=ach, next_achieved=next_ach)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
            obs = curriculum.on_reset(obs)
            if goal_tools is not None:
                goal_tools.calibrate(obs)
                task_goal = goal_tools.task_goal()
                replay.set_goal(task_goal)
            reset_episode_state()
            if explorer:
                acting = 'explorer' if rng.random() < cfg.explore_frac \
                    else 'task'

        # first-partial is a logged milestone only -- it changes nothing
        if trigger_step is None and \
                replay.best_online_reward > cfg.success_reward_thresh:
            trigger_step = step
            print(f'first partial at step {step} '
                  f'(best online reward {replay.best_online_reward:.2f})')

        # ---------------- updates ----------------
        if step > cfg.start_training and replay.ready(cfg.seq_len):
            seqs = replay.sample_seqs(
                cfg.wm_batch, cfg.seq_len, rng,
                reward_frac=cfg.reward_batch_frac,
                reward_thresh=cfg.success_reward_thresh)
            her = (dict(goal_tools=goal_tools, task_goal=task_goal,
                        frac=cfg.her_frac) if goal_tools else None)
            win = build_windows(seqs, N, OBS_KEY, ACTION_KEY,
                                take=cfg.batch_size, rng=rng, device=device,
                                her=her)
            if win is not None:
                metrics_acc.update(prefixed(task.update(win), 'task'))
            if wm is not None and step % cfg.wm_every == 0:
                metrics_acc.update(wm.train_batch(seqs))
                metrics_acc.update(prefixed(ensemble.train_from_wm(
                    wm.last_feats, wm.last_actions, wm.last_embeds), 'wm'))
            if explorer is not None and step % cfg.explore_every == 0 \
                    and wm.last_feats is not None:
                flat = wm.last_feats.flatten(0, 1)
                idx = torch.randint(0, flat.shape[0],
                                    (cfg.imagine_batch,), device=flat.device)
                starts = feat_to_carry(wm, flat[idx])
                metrics_acc.update(prefixed(
                    explorer.update(wm, ensemble, starts), 'explorer'))
                if cfg.owm_alpha > 0:
                    rc = explorer.last_rollout
                    metrics_acc.update(prefixed(wm.optimism_update(
                        rc['deters'], rc['zs'], rc['advantages'],
                        cfg.owm_alpha, cfg.owm_eta), 'wm'))

        # ---------------- logging / eval / ckpt ----------------
        if step % cfg.log_every == 0:
            stats = replay.success_stats(cfg.success_reward_thresh)
            metrics_acc.update(prefixed(stats, 'replay'))
            metrics_acc['diagnosis/first_partial_seen'] = float(
                trigger_step is not None)
            metrics_acc.update(prefixed(curriculum.metrics(), 'curriculum'))
            wandb.log(metrics_acc, step=step)
            metrics_acc = {}
        if step % cfg.eval_every == 0:
            em, frames = run_eval(env, task, cfg, cfg.eef_slice, gc)
            log = prefixed(em, 'eval')
            if frames:
                arr = np.stack(frames).transpose(0, 3, 1, 2)
                log['eval/video'] = wandb.Video(arr, fps=15, format='mp4')
            wandb.log(log, step=step)
            obs, _ = env.reset()
            obs = curriculum.on_reset(obs)   # training episode resumes
            if goal_tools is not None:
                goal_tools.calibrate(obs)
                task_goal = goal_tools.task_goal()
                replay.set_goal(task_goal)
            reset_episode_state()
            replay.end_episode()
        if step % cfg.save_every == 0:
            torch.save({'task': task.state_dict(),
                        'explorer': explorer.state_dict() if explorer
                        else None,
                        'wm': wm.state_dict() if wm else None,
                        'step': step},
                       out_dir / 'sear_latest.pt')
    wandb.finish()


if __name__ == '__main__':
    train(load_config('train_sear', argv=sys.argv[1:]))