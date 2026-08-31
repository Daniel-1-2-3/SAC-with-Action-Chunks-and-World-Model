""" Eval-time planning on a trained checkpoint: the world model rolls
    candidate chunk sequences in latent space, scores them with the reward
    head plus the SEAR critic, and MPPI refines toward the best (TD-MPC2's
    recipe, adapted to action chunks).

    Runs a three-way comparison on the SAME frozen weights:
      policy   -- bare SEAR actor (what normal eval does)
      bestofn  -- K policy samples scored by critic Q only (no model)
      planner  -- full MPPI with latent rollouts, reward head, terminal Q

    Any separation between bestofn and planner is the world model's
    marginal value at eval, measured directly.

    Usage:
      python plan_eval.py --checkpoint sear_runs/.../sear_latest.pt \
          --env_name cube-double-play-singletask-v0 --episodes 20
"""
import argparse
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys

import numpy as np
import ogbench
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.common import load_config, set_seed_everywhere  # noqa: E402
from sear.agent import SEARAgent                             # noqa: E402
from torch_wm.world_model import WorldModel                  # noqa: E402


@torch.no_grad()
def mppi_plan(wm, task, carry, obs, cfg, args, device):
    """ One planning call at the current (real) posterior carry.
        Optimizes a sequence of plan_chunks chunks; returns the first
        chunk (numpy, (N, act_dim)). """
    N, A = task.chunk_len, task.act_dim
    H = args.plan_chunks * N
    K = args.candidates
    mu = torch.zeros(H, A, device=device)
    sigma = torch.full((H, A), args.init_std, device=device)

    # policy-prior candidates: policy chunk at the current obs, tiled
    obs_t = torch.as_tensor(obs, dtype=torch.float32,
                            device=device).reshape(1, -1)
    prior_chunks, _ = task.policy.sample(obs_t.repeat(args.prior_k, 1))

    base = {k: v.repeat(K, *([1] * (v.dim() - 1))) for k, v in carry.items()}
    for it in range(args.iterations):
        cand = (mu.unsqueeze(0) + sigma.unsqueeze(0)
                * torch.randn(K, H, A, device=device)).clamp(-1, 1)
        cand[:args.prior_k, :N] = prior_chunks
        # roll all K candidates through the model in one batch
        c = {k: v.clone() for k, v in base.items()}
        score = torch.zeros(K, device=device)
        feats_last = None
        for t in range(H):
            c = wm.img_step(c, cand[:, t])
            feat = wm.feat(c)
            score = score + (task.gamma ** t) * wm.pred_reward(feat)
            feats_last = feat
        # terminal value: critic Q^(N) at the imagined end state with a
        # fresh policy chunk there (decoded obs -> policy -> critic)
        end_obs = wm.decode(feats_last)
        end_chunk, _ = task.policy.sample(end_obs)
        q_end = task.critic.target_min(end_obs, end_chunk)[:, -1]
        score = score + (task.gamma ** H) * q_end
        # MPPI refit
        w = torch.softmax(score / args.temperature, dim=0)
        mu = (w.reshape(K, 1, 1) * cand).sum(0)
        sigma = ((w.reshape(K, 1, 1) * (cand - mu) ** 2).sum(0)
                 .sqrt().clamp(min=0.05))
    return mu[:N].cpu().numpy()


@torch.no_grad()
def run_mode(mode, env, wm, task, cfg, args, device):
    N, A = task.chunk_len, task.act_dim
    returns, succ = [], 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        carry = wm.init(1) if wm is not None else None
        prev_a = np.zeros(A, np.float32)
        first = True
        ep_ret, steps, done = 0.0, 0, False
        chunk, pos = None, 10 ** 9
        while not done and steps < args.max_steps:
            if wm is not None:
                carry, _ = wm.encode_step(carry, obs[None], prev_a[None],
                                          np.asarray([first], bool))
            first = False
            if pos >= cfg.eval_receding:
                if mode == 'policy':
                    chunk = task.act(obs, deterministic=True)
                elif mode == 'bestofn':
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    ).reshape(1, -1).repeat(args.candidates, 1)
                    cands, _ = task.policy.sample(obs_t)
                    q = task.critic.target_min(obs_t, cands)[:, -1]
                    chunk = cands[q.argmax()].cpu().numpy()
                else:  # planner
                    chunk = mppi_plan(wm, task, carry, obs, cfg, args,
                                      device)
                pos = 0
            a = chunk[pos]
            pos += 1
            obs, r, term, trunc, info = env.step(a)
            prev_a = a.astype(np.float32)
            ep_ret += r
            steps += 1
            done = term or trunc
        returns.append(ep_ret)
        succ += float(info.get('success', 0.0))
    return float(np.mean(returns)), succ / args.episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--env_name', default='cube-double-play-singletask-v0')
    p.add_argument('--config', default='configs.yaml')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--modes', default='policy,bestofn,planner')
    p.add_argument('--candidates', type=int, default=512)
    p.add_argument('--prior_k', type=int, default=64)
    p.add_argument('--plan_chunks', type=int, default=2)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--init_std', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfg = load_config('train_sear', path=args.config, argv=[]).train_sear
    set_seed_everywhere(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    except TypeError:
        env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    ckpt = torch.load(args.checkpoint, map_location=device)
    task = SEARAgent(obs_dim, act_dim, cfg.chunk_len, gamma=cfg.gamma,
                     device=device,
                     critic_kw=dict(vmin=cfg.critic_vmin,
                                    vmax=cfg.critic_vmax))
    task.load_state_dict(ckpt['task'])
    wm = None
    if ckpt.get('wm') is not None:
        wm = WorldModel(obs_dim, act_dim, device=device)
        wm.load_state_dict(ckpt['wm'])
    print(f'checkpoint step {ckpt.get("step")} | wm: {wm is not None}')

    for mode in args.modes.split(','):
        if mode != 'policy' and mode == 'planner' and wm is None:
            print(f'{mode:>8}: SKIPPED (checkpoint has no world model)')
            continue
        ret, sr = run_mode(mode, env, wm, task, cfg, args, device)
        print(f'{mode:>8}: mean_return {ret:9.2f} | success_rate {sr:.2f}')


if __name__ == '__main__':
    main()
