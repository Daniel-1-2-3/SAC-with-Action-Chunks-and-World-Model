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
from sac_chunked.chunk_utils import pool_chunk_np, step_valid_np
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from helpers.interop import numeric_metrics
from helpers.ogbench_methods import OGBenchMethods
from wm.chunk_selector import ChunkSelector

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

        Windows are sampled uniformly and masked, not rejected -- see
        sample_chunks. This mirrors utils/datasets.py sample_sequence in the
        reference implementation.

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
        self.mask = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminal = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
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
        self.mask[i] = 0.0 if terminated else 1.0
        # A truncation ends the window for `valid` purposes even though it
        # does not zero the bootstrap mask.
        self.terminal[i] = 1.0 if (terminated or truncated) else 0.0
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

        if reward > -1.0:
            self._ep_success = True
        if terminated or truncated:
            self.online_episodes += 1
            self.online_success += int(self._ep_success)
            self._ep_success = False

    def seed_from_offline(self, dataset):
        obs = np.asarray(dataset['observations'], dtype=np.float32)
        act = np.asarray(dataset['actions'], dtype=np.float32)
        rew = np.asarray(dataset['rewards'], dtype=np.float32).reshape(-1, 1)
        nobs = np.asarray(dataset['next_observations'], dtype=np.float32)
        term = np.asarray(dataset['terminals']).reshape(-1).astype(bool)
        if 'masks' in dataset:
            mk = np.asarray(dataset['masks'], dtype=np.float32).reshape(-1, 1)
        else:
            mk = (~term).astype(np.float32).reshape(-1, 1)

        n = min(len(obs), self.capacity)
        sl = slice(0, n)
        self.obs[sl] = obs[:n]
        self.action[sl] = act[:n]
        self.reward[sl] = rew[:n]
        self.next_obs[sl] = nobs[:n]
        self.mask[sl] = mk[:n]
        self.terminal[sl] = term[:n].astype(np.float32).reshape(-1, 1)
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

    def sample_chunks(self, batch_size, device, rng, gamma):
        """ utils/datasets.py sample_sequence. Start indices are uniform over
            the buffer with no episode-boundary rejection; windows that cross
            a terminal are kept and masked. Returns the per-position validity
            too, for the BC flow term. """
        h = self.chunk_len
        size = len(self)
        if size <= h:
            return None
        starts = rng.integers(0, size - h + 1, size=batch_size)
        window = starts[:, None] + np.arange(h)[None, :]

        rewards = self.reward[window, 0]
        masks = self.mask[window, 0]
        terminals = self.terminal[window, 0]
        chunk_reward, chunk_mask, valid = pool_chunk_np(rewards, masks, terminals, gamma)
        step_valid = step_valid_np(terminals)
        chunks = self.action[window].reshape(batch_size, h * self.action_dim)

        to = lambda x: torch.as_tensor(x, device=device).float()
        return (to(self.obs[starts]), to(chunks), to(chunk_reward), to(chunk_mask),
                to(valid), to(step_valid), to(self.next_obs[window[:, -1]]))

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

def _agent_update(policy, replay, batch_size, device, rng, gamma, gamma_h):
    batch = replay.sample_chunks(batch_size, device, rng, gamma)
    if batch is None:
        return None
    b_obs, b_chunk, b_rew, b_mask, b_valid, b_step_valid, b_next = batch

    # Chunk-level TD target. Unlike the world-model arm there is no imagined
    # horizon to run lambda-returns over, so this is a single chunk_len-step
    # backup -- which is exactly the unbiased n-step backup the QC paper is
    # built around, because the critic scores the whole chunk that produced
    # these rewards.
    with torch.no_grad():
        next_values = policy.chunk_target_values(b_next)
        targets = b_rew + gamma_h * b_mask * next_values

    metrics = {}
    metrics.update(_prefixed(policy.update_critic(b_obs, b_chunk, targets, b_valid), 'sac'))
    # Reference passes one batch to both terms; weight is ones because the
    # actor loss in acfql.actor_loss is unweighted.
    ones = torch.ones_like(b_valid)
    metrics.update(_prefixed(policy.update_actor(
        b_obs, ones, bc_feat=b_obs, bc_chunk=b_chunk, bc_valid=b_step_valid), 'sac'))
    policy.update_target()

    metrics['sac/mean_chunk_reward'] = b_rew.mean().item()
    metrics['sac/mean_chunk_mask'] = b_mask.mean().item()
    metrics['sac/valid_frac'] = b_valid.mean().item()
    metrics['sac/chunk_diversity'] = policy.chunk_diversity(b_obs)
    metrics['diagnosis/batch_reward_max'] = b_rew.max().item()
    return metrics

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
    print(f'PyTorch device: {device} | chunk_len={chunk_len} alpha={chunk_config.alpha}')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode,
               name=(general_config.run_name or None), config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # main.py sizes the online buffer as max(buffer_size, dataset.size + 1),
    # so offline data is never evicted once online transitions start arriving.
    # A fixed capacity is not safe here: cube-double's offline dataset is
    # ~1M transitions, so a 1M-capacity ring buffer is already full the
    # instant it is seeded, and the first online transition would silently
    # start overwriting offline data.
    dataset_size = len(train_dataset['observations']) if train_dataset is not None else 0
    capacity = max(chunk_config.replay_capacity, dataset_size + general_config.num_online_steps + 1)
    replay = ChunkTransitionReplay(obs_dim, action_dim, chunk_len, capacity=capacity)
    if train_dataset is not None:
        n = replay.seed_from_offline(train_dataset)
        print(f'Seeded replay buffer with {n} offline transitions '
              f'({replay.offline_episodes} episodes) | buffer capacity {capacity}')

    policy = ChunkAgent(
        repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
    )

    # QC's own best-of-N: sample select_n candidate chunks, score each with
    # the online critic Q(s, chunk), execute the argmax. No learned model
    # anywhere. select_n=1 is plain QC-FQL. This is the CONTROL arm: the
    # model arm samples the same select_n candidates from the same policy and
    # differs only in what ranks them.
    selector = ChunkSelector(
        None, policy, action_dim, chunk_len, chunk_config.select_n, gamma,
        device, score_mode='critic')
    arm_name = ('critic_best_of_n' if chunk_config.select_n > 1
                else 'no_model')
    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm=arm_name,
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
        if results['video'] is not None:
            log_dict['eval/video'] = wandb.Video(results['video'], fps=20, format='mp4')
        wandb.log(log_dict, step=step)

    # Offline phase. main.py runs offline_steps gradient updates on the static
    # dataset before any environment interaction, then online_steps of
    # interaction. Every number in the paper's tables uses that protocol.
    n_updates = 0
    for i in range(1, general_config.num_offline_steps + 1):
        metrics = _agent_update(policy, replay, batch_size, device, rng, gamma, gamma_h)
        if metrics is None:
            continue
        n_updates += 1
        if i % general_config.log_every == 0:
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 0
            wandb.log(numeric_metrics(metrics), step=i)
        if i % general_config.eval_every == 0:
            run_eval(i, n_updates)
            env.reset()
    offline_steps = general_config.num_offline_steps
    if offline_steps > 0:
        torch.save(policy.state_dict_all(), out_dir / 'chunk_offline.pt')
        print(f'Offline phase done: {n_updates} updates')

    obs, info = env.reset(seed=config.seed)
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    print('Starting online phase (no world model)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        if global_step < general_config.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
            # The chunk is executed fully before the actor is queried again,
            # matching main.py's action_queue.
            if chunk_pos >= chunk_len:
                chunk_buffer = selector.select(state)
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
        log_step = offline_steps + global_step
        metrics = {}

        # main.py: `if i >= FLAGS.start_training`. No gradient updates for
        # the first start_training online steps -- just interaction, so the
        # buffer accumulates fresh online transitions before training resumes
        # on them. Applies to online-phase updates only; the offline loop
        # above is unaffected.
        for _ in range(chunk_config.utd_ratio if global_step >= general_config.start_training else 0):
            m = _agent_update(policy, replay, batch_size, device, rng, gamma, gamma_h)
            if m is not None:
                metrics = m
                n_updates += 1

        if global_step % general_config.log_every == 0:
            metrics['diagnosis/replay_transitions'] = len(replay)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 1
            # select/score_gap and select/score_std, logged under the same
            # names the model arm uses, so "how separated are the candidates"
            # is comparable between the two scorers.
            metrics.update(selector.pop_stats())
            _succ = replay.success_stats
            metrics['replay/success_frac_total'] = _succ['total_frac']
            metrics['replay/success_frac_online'] = _succ['online_frac']
            metrics['replay/success_frac_offline'] = _succ['offline_frac']
            metrics['replay/success_episodes_online'] = _succ['online_success']
            metrics['replay/success_episodes_total'] = _succ['offline_success'] + _succ['online_success']
            wandb.log(numeric_metrics(metrics), step=log_step)

        if global_step % general_config.eval_every == 0:
            run_eval(log_step, n_updates)
            obs, info = env.reset()
            chunk_pos = chunk_len

        if global_step % general_config.save_every == 0:
            torch.save(policy.state_dict_all(), out_dir / 'chunk_latest.pt')

    torch.save(policy.state_dict_all(), out_dir / 'chunk_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac_chunked.py --train_sac_chunked.general.env_name=cube-double-play-singletask-v0