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

from wm_sac.sac_chunk_agent import ChunkAgent
from wm_sac.flow_bc import FlowBC
from wm_sac.sac_wm_utils import set_seed_everywhere
from wm_sac.chunk_utils import pool_chunk_np
from wm_sac.evaluation_chunk import eval_chunk_in_env, EvalCSV
from wm_sac.interop import numeric_metrics
from wm_sac.ogbench_methods import OGBenchMethods

OBS_KEY = 'state'
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

class ChunkTransitionReplay:
    """ Flat ring buffer that can also serve chunk-level transitions.

        Alongside the usual (s, a, r, s', not_done) it stores an episode id per
        slot. A chunk window starting at i is valid only if all chunk_len slots
        share an episode id and the window does not straddle the write head --
        otherwise the "chunk" would splice together unrelated timesteps.

        This is where the world model's structural advantage shows up as
        absence: the only chunks available here are chunks that were actually
        executed. There is no way to ask what a different chunk would have
        done from the same state. """

    def __init__(self, obs_dim, action_dim, chunk_len, capacity=1_000_000):
        self.capacity = int(capacity)
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.not_done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ep_id = np.full((self.capacity,), -1, dtype=np.int64)
        self.idx = 0
        self.full = False
        self.cur_ep = 0
        self.offline_episodes = 0
        self.offline_success = 0
        self.online_episodes = 0
        self.online_success = 0
        self._ep_success = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, terminated, truncated=False):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.not_done[i] = 0.0 if terminated else 1.0
        self.ep_id[i] = self.cur_ep
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

        if reward > -1.0:
            self._ep_success = True
        if terminated or truncated:
            self.online_episodes += 1
            self.online_success += int(self._ep_success)
            self._ep_success = False
            self.cur_ep += 1

    def seed_from_offline(self, dataset):
        obs = np.asarray(dataset['observations'], dtype=np.float32)
        act = np.asarray(dataset['actions'], dtype=np.float32)
        rew = np.asarray(dataset['rewards'], dtype=np.float32).reshape(-1, 1)
        nobs = np.asarray(dataset['next_observations'], dtype=np.float32)
        term = np.asarray(dataset['terminals']).reshape(-1).astype(bool)
        if 'masks' in dataset:
            nd = np.asarray(dataset['masks'], dtype=np.float32).reshape(-1, 1)
        else:
            nd = (~term).astype(np.float32).reshape(-1, 1)

        n = min(len(obs), self.capacity)
        sl = slice(0, n)
        self.obs[sl] = obs[:n]
        self.action[sl] = act[:n]
        self.reward[sl] = rew[:n]
        self.next_obs[sl] = nobs[:n]
        self.not_done[sl] = nd[:n]
        # Episode ids come from the terminal flags, so chunk windows never
        # cross a boundary in the offline data either.
        self.ep_id[sl] = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
        self.cur_ep = int(self.ep_id[n - 1]) + 1
        self.idx = n % self.capacity
        if n >= self.capacity:
            self.full = True

        ep_ok = False
        for t in range(n):
            if rew[t, 0] > -1.0:
                ep_ok = True
            if term[t]:
                self.offline_episodes += 1
                self.offline_success += int(ep_ok)
                ep_ok = False
        return n

    def sample_chunks(self, batch_size, device, rng, gamma, reward_shift):
        h = self.chunk_len
        size = len(self)
        starts = rng.integers(0, max(size - h, 1), size=batch_size * 4)
        window = starts[:, None] + np.arange(h)[None, :]
        same_ep = (self.ep_id[window] == self.ep_id[starts][:, None]).all(-1)
        # Reject windows that contain the write head, whose slots belong to
        # two different points in time.
        if self.full:
            same_ep &= ~((window == self.idx).any(-1))
        starts = starts[same_ep][:batch_size]
        if len(starts) == 0:
            return None
        window = starts[:, None] + np.arange(h)[None, :]

        rewards = self.reward[window, 0] + reward_shift
        conts = self.not_done[window, 0]
        chunk_reward, chunk_cont = pool_chunk_np(rewards, conts, gamma)
        chunks = self.action[window].reshape(len(starts), h * self.action_dim)

        to = lambda x: torch.as_tensor(x, device=device).float()
        return (to(self.obs[starts]), to(chunks), to(chunk_reward),
                to(chunk_cont), to(self.next_obs[window[:, -1]]))

    @property
    def success_stats(self):
        n_off, n_on = self.offline_episodes, self.online_episodes
        off, on = self.offline_success, self.online_success
        return {
            'offline_frac': off / max(n_off, 1),
            'online_frac': on / max(n_on, 1),
            'total_frac': (off + on) / max(n_off + n_on, 1),
            'offline_success': off,
            'online_success': on,
        }

def train(config):
    general_config = config.train_sac_chunked.general
    chunk_config = config.train_sac_chunked.chunk

    out_dir = pathlib.Path(general_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = chunk_config.batch_size
    chunk_len = chunk_config.chunk_len
    gamma = chunk_config.gamma
    gamma_h = gamma ** chunk_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device} | chunk_len={chunk_len}')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay = ChunkTransitionReplay(obs_dim, action_dim, chunk_len,
                                   capacity=chunk_config.replay_capacity)
    if train_dataset is not None:
        n = replay.seed_from_offline(train_dataset)
        print(f'Seeded replay buffer with {n} offline transitions '
              f'({replay.offline_episodes} episodes)')

    policy = ChunkAgent(
        repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, feature_dim=chunk_config.feature_dim,
        hidden_dim=chunk_config.hidden_dim, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, bc_alpha=chunk_config.bc_alpha,
        normalize_q=chunk_config.normalize_q,
    )
    flow_bc = FlowBC(
        repr_dim=obs_dim, chunk_dim=action_dim * chunk_len, device=device,
        lr=chunk_config.lr, feature_dim=chunk_config.feature_dim,
        hidden_dim=chunk_config.hidden_dim, flow_steps=chunk_config.flow_steps,
    )

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm='no_world_model',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    obs, info = env.reset(seed=config.seed)
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    n_updates = 0
    print('Starting chunked-agent training loop (no world model)')

    while global_step < general_config.num_train_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        if global_step < general_config.num_seed_steps:
            action = env.action_space.sample()
        else:
            if chunk_pos >= chunk_len:
                chunk_buffer = policy.act(state, eval_mode=False)
                chunk_pos = 0
            action = chunk_buffer[chunk_pos]
            chunk_pos += 1

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        replay.add(state, action, reward, np.asarray(next_obs, dtype=np.float32),
                   terminated, truncated)

        done = bool(terminated or truncated)
        obs = next_obs
        if done:
            obs, info = env.reset()
            chunk_pos = chunk_len

        global_step += 1
        metrics = {}

        if len(replay) >= batch_size * 4 and global_step % chunk_config.train_every == 0:
            batch = replay.sample_chunks(batch_size, device, rng, gamma, chunk_config.reward_shift)
            if batch is not None:
                b_obs, b_chunk, b_rew, b_cont, b_next = batch

                # Chunk-level TD target. Unlike the world-model arm there is no
                # imagined horizon to run lambda-returns over, so this is a
                # single chunk_len-step backup -- which is exactly the unbiased
                # n-step backup the QC paper is built around, because the
                # critic scores the whole chunk that produced these rewards.
                with torch.no_grad():
                    next_values = policy.chunk_target_values(b_next)
                    targets = b_rew + gamma_h * b_cont * next_values

                weights = torch.ones_like(b_rew)
                metrics.update(_prefixed(policy.update_critic(b_obs, b_chunk, targets, weights), 'sac'))
                metrics.update(_prefixed(flow_bc.update(b_obs, b_chunk), 'sac'))
                metrics.update(_prefixed(policy.update_actor(b_obs, weights, flow_bc), 'sac'))
                policy.update_target()
                n_updates += 1

                metrics['sac/mean_chunk_reward'] = b_rew.mean().item()
                metrics['sac/mean_chunk_cont'] = b_cont.mean().item()
                metrics['sac/chunk_diversity'] = policy.chunk_diversity(b_obs)
                metrics['diagnosis/batch_reward_max'] = b_rew.max().item()

        if global_step % general_config.log_every == 0:
            metrics['diagnosis/replay_transitions'] = len(replay)
            metrics['diagnosis/gradient_updates'] = n_updates
            _succ = replay.success_stats
            metrics['replay/success_frac_total'] = _succ['total_frac']
            metrics['replay/success_frac_online'] = _succ['online_frac']
            metrics['replay/success_frac_offline'] = _succ['offline_frac']
            metrics['replay/success_episodes_online'] = _succ['online_success']
            metrics['replay/success_episodes_total'] = _succ['offline_success'] + _succ['online_success']

        if metrics and global_step % general_config.log_every == 0:
            wandb.log(numeric_metrics(metrics), step=global_step)

        if global_step % general_config.eval_every == 0 and global_step > 0:
            results = eval_chunk_in_env(
                env, None, policy, action_dim, general_config.eval_episodes,
                device, OBS_KEY, chunk_len, eef_slice=eef_slice, record_video=True)
            print(f'step {global_step:7d} | return {results["mean_return"]:.2f} | '
                  f'success {results["success_rate"]:.2f} | coherence {results["coherence"]:.4f}')
            eval_csv.append(global_step, n_updates, time.time() - start_time, results)
            log_dict = {
                'eval/mean_return': results['mean_return'],
                'eval/success_rate': results['success_rate'],
                'eval/coherence': results['coherence'],
                'eval/mean_episode_len': results['mean_episode_len'],
            }
            if results['video'] is not None:
                log_dict['eval/video'] = wandb.Video(results['video'], fps=20, format='mp4')
            wandb.log(log_dict, step=global_step)
            obs, info = env.reset()
            chunk_pos = chunk_len

        if global_step % general_config.save_every == 0 and global_step > 0:
            torch.save(policy.state_dict_all(), out_dir / 'chunk_latest.pt')
            torch.save(flow_bc.state_dict_all(), out_dir / 'flow_latest.pt')

    torch.save(policy.state_dict_all(), out_dir / 'chunk_final.pt')
    torch.save(flow_bc.state_dict_all(), out_dir / 'flow_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac_chunk.py --train_sac_chunked.general.env_name=cube-double-play-singletask-v0