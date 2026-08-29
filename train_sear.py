""" SEAR + Plan2Explore trainer: from scratch, single budget.

    Two SEAR learners share one architecture:
      task agent   -- trained on real replay with environment reward;
                      it is the ONLY policy evaluated.
      explorer     -- trained on imagined rollouts labeled with ensemble
                      disagreement (Plan2Explore, off-policy adaptation);
                      its job is to collect informative real data.

    Collection alternates whole episodes between the two policies. Before
    the first online partial success the mix is explorer-heavy; after the
    trigger it shifts task-heavy (the project's event-trigger pattern,
    promoted from patch to scheduler). Both policies collect with SEAR's
    random replanning. Eval runs the task policy with a receding horizon.

    Run:
      python train_sear.py --train_sear.env_name=cube-double-play-singletask-v0
"""
import os
import pathlib
import sys
import time
from collections import deque

import numpy as np
import ogbench
import torch
import wandb

from explore.imagination import imagine_episodes
from helpers.common import (load_config, prefixed, sample_sequences,
                            set_seed_everywhere, temporal_coherence)
from helpers.sear_replay import EpisodeReplay
from sear.agent import SEARAgent
from sear.windows import build_windows
from torch_wm.ensemble import DisagreementEnsemble
from torch_wm.world_model import WorldModel

OBS_KEY, ACTION_KEY = 'state', 'action'


def run_eval(env, agent, cfg, eef_slice):
    returns, lens, succ, coh = [], [], [], []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        chunk, pos = None, cfg.chunk_len
        eefs = []
        while not done and steps < cfg.eval_max_steps:
            # receding horizon: replan every eval_receding steps
            if pos >= cfg.eval_receding:
                chunk = agent.act(obs, deterministic=True)
                pos = 0
            a = chunk[pos]
            pos += 1
            obs, r, term, trunc, info = env.step(a)
            eefs.append(obs[eef_slice[0]:eef_slice[1]])
            ep_ret += r
            steps += 1
            done = term or trunc
        returns.append(ep_ret)
        lens.append(steps)
        succ.append(float(info.get('success', 0.0)))
        coh.append(temporal_coherence(np.asarray(eefs)))
    return {'mean_return': float(np.mean(returns)),
            'success_rate': float(np.mean(succ)),
            'mean_episode_len': float(np.mean(lens)),
            'coherence': float(np.mean(coh))}


def train(config):
    cfg = config.train_sear
    set_seed_everywhere(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(config.seed)
    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode,
               config=dict(config))

    env = ogbench.make_env_and_datasets(cfg.env_name, env_only=True)
    obs, _ = env.reset(seed=config.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    N = cfg.chunk_len

    replay = EpisodeReplay(OBS_KEY, ACTION_KEY, cfg.max_episodes)
    wm = WorldModel(obs_dim, act_dim, device=device,
                    lr=cfg.wm_lr) if cfg.use_world_model else None
    ensemble = DisagreementEnsemble(
        obs_dim, act_dim, k=cfg.ensemble_k,
        obs_slice=cfg.ensemble_obs_slice, device=device) \
        if cfg.use_world_model else None
    task = SEARAgent(obs_dim, act_dim, N, gamma=cfg.gamma, device=device)
    explorer = SEARAgent(obs_dim, act_dim, N, gamma=cfg.gamma,
                         device=device) if cfg.use_world_model else None
    imagined = deque(maxlen=cfg.imagined_capacity)

    print(f'sear-p2e | task=SEAR(chunk {N}) | '
          f'{"explorer=SEAR-on-imagined-disagreement (P2E)" if explorer else "NO explorer (SEAR-only baseline)"} | '
          f'eval=task policy, receding horizon {cfg.eval_receding}')

    trigger_step = None
    acting = 'explorer' if explorer else 'task'
    chunk, pos = None, N
    ep_steps = 0
    metrics_acc = {}

    for step in range(1, cfg.num_online_steps + 1):
        # ---------------- collection ----------------
        replan = (pos >= N) or (rng.random() < cfg.replan_prob)
        if replan:
            if step <= cfg.num_seed_steps:
                chunk = rng.uniform(-1, 1, size=(N, act_dim)).astype(
                    np.float32)
            else:
                actor = explorer if (acting == 'explorer' and explorer) \
                    else task
                chunk = actor.act(obs, deterministic=False)
            pos = 0
        a = chunk[pos]
        pos += 1
        next_obs, r, term, trunc, _ = env.step(a)
        ep_steps += 1
        if ep_steps >= cfg.max_episode_steps:
            trunc = True
        replay.add_step(obs, a, r, next_obs, term, trunc)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
            ep_steps, pos = 0, N
            # scheduler: pick next episode's policy
            if explorer:
                frac = cfg.explore_frac_pre if trigger_step is None \
                    else cfg.explore_frac_post
                acting = 'explorer' if rng.random() < frac else 'task'

        # event trigger: first online partial success
        if trigger_step is None and \
                replay.best_online_reward > cfg.success_reward_thresh:
            trigger_step = step
            print(f'trigger at step {step} '
                  f'(best online reward {replay.best_online_reward:.2f})')

        # ---------------- updates ----------------
        if step > cfg.start_training and replay.ready(cfg.seq_len):
            seqs = replay.sample_seqs(cfg.wm_batch, cfg.seq_len, rng)
            win = build_windows(seqs, N, OBS_KEY, ACTION_KEY,
                                take=cfg.batch_size, rng=rng, device=device)
            if win is not None:
                metrics_acc.update(prefixed(task.update(win), 'task'))
            if wm is not None and step % cfg.wm_every == 0:
                metrics_acc.update(wm.train_batch(seqs))
                o, a_, no = replay.sample_pairs(cfg.batch_size, rng)
                metrics_acc.update(prefixed(
                    ensemble.train_batch(o, a_, no), 'wm'))
            if explorer is not None and step % cfg.imagine_every == 0:
                starts = replay.sample_start_obs(cfg.imagine_batch, rng)
                imagined.extend(imagine_episodes(
                    wm, explorer, ensemble, starts,
                    cfg.imagine_horizon_chunks, OBS_KEY, ACTION_KEY))
                for _ in range(cfg.explorer_updates):
                    iseqs = sample_sequences(
                        list(imagined), cfg.batch_size // 4,
                        min(cfg.seq_len,
                            cfg.imagine_horizon_chunks * N + 1),
                        OBS_KEY, ACTION_KEY, rng)
                    iwin = build_windows(iseqs, N, OBS_KEY, ACTION_KEY,
                                         take=cfg.batch_size, rng=rng,
                                         device=device)
                    if iwin is not None:
                        metrics_acc.update(prefixed(
                            explorer.update(iwin), 'explorer'))

        # ---------------- logging / eval / ckpt ----------------
        if step % cfg.log_every == 0:
            stats = replay.success_stats(cfg.success_reward_thresh)
            metrics_acc.update(prefixed(stats, 'replay'))
            metrics_acc['diagnosis/trigger_fired'] = float(
                trigger_step is not None)
            wandb.log(metrics_acc, step=step)
            metrics_acc = {}
        if step % cfg.eval_every == 0:
            em = run_eval(env, task, cfg, cfg.eef_slice)
            wandb.log(prefixed(em, 'eval'), step=step)
            obs, _ = env.reset()
            ep_steps, pos = 0, N
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
