import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import elements
import ruamel.yaml as yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_ckpt', required=True)
    p.add_argument('--env_name', default='cube-triple-play-singletask-task4-v0')
    p.add_argument('--num_dyn', type=int, default=4)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--windows', type=int, default=1024)
    a = p.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    cfgs = yaml.YAML(typ='safe').load(open(root / 'configs.yaml'))
    cfg = elements.Config(cfgs['defaults'])
    tdmpc = cfg.tdmpc.update({'compile_nets': False})
    chunk_len = cfg.chunk.chunk_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from helpers.ogbench_methods import OGBenchMethods
    env, train_dataset, _ = OGBenchMethods.load_ogbench(a.env_name)
    obs_dim = train_dataset['observations'].shape[1]
    action_dim = train_dataset['actions'].shape[1]

    from tdmpc.agent import TDMPC2Model
    model = TDMPC2Model(obs_dim, action_dim, device, tdmpc, cfg.chunk.gamma,
                        num_dyn=a.num_dyn, chunk_len=chunk_len)
    model.load_state_dict_all(torch.load(a.model_ckpt, map_location=device))

    from sac_chunked.replay import ChunkTransitionReplay
    replay = ChunkTransitionReplay(
        obs_dim, action_dim, chunk_len,
        capacity=len(train_dataset['observations']) + 1)
    replay.seed_from_offline(train_dataset)

    from wm.visit_memory import VisitMemory
    rng = np.random.default_rng(0)
    memory = VisitMemory(obs_dim, device, offline_size=25000, k=5)
    memory.fill_offline(replay, rng)
    memory.refresh(model.encode)
    typical = memory.typical_distance

    span = a.depth * chunk_len
    w = replay.sample_model_windows(a.windows, span, device, rng,
                                    online_frac=0.0)
    keep = (w['valid'][:, :, 0].min(dim=1).values > 0.5)
    obs, next_obs, act = w['obs'][keep], w['next_obs'][keep], w['action'][keep]
    print(f'{int(keep.sum())} windows fully inside one episode | '
          f'typical distance {typical:.4f}')

    deepest = 0
    with torch.no_grad():
        z = model.encode(obs[:, 0])
        for d in range(1, a.depth + 1):
            lo = (d - 1) * chunk_len
            z, dis = model.imagine_chunk(z, act[:, lo:lo + chunk_len])
            z_real = model.encode(next_obs[:, d * chunk_len - 1])
            drift = (z - z_real).pow(2).sum(-1).sqrt()
            ratio = (drift / typical).cpu().numpy()
            dis_np = dis.cpu().numpy()
            if dis_np.std() > 1e-12 and ratio.std() > 1e-12:
                corr = float(np.corrcoef(dis_np, ratio)[0, 1])
            else:
                corr = float('nan')
            print(f'depth {d} | drift ratio mean {ratio.mean():.3f} '
                  f'90th {np.percentile(ratio, 90):.3f} | '
                  f'disagreement mean {dis_np.mean():.6f} | '
                  f'correlation(disagreement, drift) {corr:.3f}')
            if ratio.mean() < 1.0:
                deepest = d
    print(f'deepest depth with mean drift ratio < 1: {deepest}')


if __name__ == '__main__':
    main()
