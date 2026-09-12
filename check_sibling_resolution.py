""" Sibling-resolution gate: can the trained wm's chunk-Q distinguish the
    16 near-identical candidates that best-of-N chooses between, and does
    it carry usable action-gradients -- compared against the trained critic
    on the same states and the same candidates?

      python check_sibling_resolution.py runs/t4_mve_fixed_s0

    Loads chunk_*.pt + model_*.pt (newest of each) from the run dir, draws
    real states from the dataset, 16 actor candidates per state, and prints
    per-state candidate spread, rank correlation, pick agreement, and mean
    |d value / d chunk| for both scorers. """

import glob
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from helpers.ogbench_methods import OGBenchMethods
from sac_chunked.experiment import load_config
from arms.mve import MVEArm

RUN = sys.argv[1] if len(sys.argv) > 1 else 'runs/t4_mve_fixed_s0'
ENV = sys.argv[2] if len(sys.argv) > 2 else 'cube-triple-play-singletask-task4-v0'
STATES, CANDS = 256, 16


def newest(pattern):
    files = glob.glob(os.path.join(RUN, pattern))
    assert files, f'no {pattern} in {RUN}'
    return max(files, key=os.path.getmtime)


def spearman(a, b):
    ra = a.argsort(-1).argsort(-1).double()
    rb = b.argsort(-1).argsort(-1).double()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return ((ra * rb).mean(-1)
            / (ra.std(-1, correction=0) * rb.std(-1, correction=0) + 1e-8))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env, data, _ = OGBenchMethods.load_ogbench(ENV)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    config = load_config('.', argv=[f'--general.env_name={ENV}'])
    arm = MVEArm(config, obs_dim, action_dim, device, np.random.default_rng(0))
    if getattr(arm, 'model', None) is None:
        arm.build_model()
    arm.policy.load_state_dict_all(torch.load(newest('chunk_*.pt'),
                                              map_location=device))
    arm.model.load_state_dict_all(torch.load(newest('model_*.pt'),
                                             map_location=device))
    policy, model = arm.policy, arm.model

    obs_all = np.asarray(data['observations'])
    idx = np.random.default_rng(0).integers(0, len(obs_all), STATES)
    obs = torch.as_tensor(obs_all[idx], dtype=torch.float32, device=device)
    rep = obs.repeat_interleave(CANDS, dim=0)

    with torch.no_grad():
        cands = policy.sample_chunk(rep)
        q_c = policy._agg(policy.critic(rep, cands)).reshape(STATES, CANDS)
        q_w = model.chunk_q(model.encode(rep), cands).reshape(STATES, CANDS)

    spread_c = q_c.std(-1)
    spread_w = q_w.std(-1)
    rank = spearman(q_c, q_w)
    agree = (q_c.argmax(-1) == q_w.argmax(-1)).double().mean()

    cands_g = cands.clone().requires_grad_(True)
    policy._agg(policy.critic(rep, cands_g)).sum().backward()
    g_c = cands_g.grad.abs().mean()
    cands_g = cands.clone().requires_grad_(True)
    # model.chunk_q / model.encode are no-grad scoring APIs; the raw net
    # path (same one the wm's own training loss uses) carries gradients.
    z_g = model.net.encode(rep)
    model.net.q_values(z_g, cands_g).mean(0).sum().backward()
    g_w = cands_g.grad.abs().mean()

    print(f'run {RUN} | {STATES} states x {CANDS} candidates')
    bin_real = ((config.tdmpc.vmax - config.tdmpc.vmin) / (config.tdmpc.num_bins - 1)
                * (1.0 + float(q_w.abs().mean())))
    print(f'sibling spread   critic {spread_c.mean():.4f}  '
          f'wm {spread_w.mean():.4f}  ratio wm/critic '
          f'{(spread_w.mean() / (spread_c.mean() + 1e-8)):.3f}')
    print(f'wm two-hot bin width at its value level: ~{bin_real:.1f} real units '
          f'-> wm spread = {float(spread_w.mean()) / (bin_real + 1e-8):.4f} bins')
    print(f'rank corr (per state, mean) {rank.mean():.3f} | '
          f'pick agreement {agree:.3f} (chance {1 / CANDS:.3f})')
    print(f'|d value / d chunk|   critic {g_c:.6f}  wm {g_w:.6f}  '
          f'ratio wm/critic {(g_w / (g_c + 1e-8)):.3f}')


if __name__ == '__main__':
    main()