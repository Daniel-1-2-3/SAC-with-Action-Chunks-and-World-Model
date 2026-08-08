import os
os.environ.setdefault('MUJOCO_GL', 'egl') # Headless for video rendering

import pathlib
import elements
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import ogbench

from sac_wm_agent import SACWorldModelAgent, sample_squashed
from sac_wm_utils import set_seed_everywhere
from interop import numeric_metrics, extract_state
from ogbench_methods import OGBenchMethods

OBS_KEY = 'state'
# Joint actions take [-1, 1]
ENV_ACTION_LOW = -1.0
ENV_ACTION_HIGH = 1.0

# Classic SAC keeps a flat transition buffer, not episodes -- there is no
# recurrent model here that needs contiguous time, so sequence structure is
# unnecessary and a 1M-transition ring buffer is the standard choice.
# Actual value comes from train_sac.sac.replay_capacity.
REPLAY_CAPACITY = 1_000_000

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
    # env_only=True returns the env itself, NOT a 3-tuple -- do not unpack it.
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    return env, None, None

def _prefixed(d, default_prefix):
    """ Prefix every key with default_prefix, EXCEPT keys that already
        carry their own namespace (e.g. 'diagnosis/critic_grad_norm')
        -- those pass through unprefixed so they land in their own
        wandb tab. Same helper as train_joint.py so the two runs'
        charts overlay directly. """
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

class TransitionReplay:
    """ Flat (s, a, r, s', not_done) ring buffer.

        Deliberately NOT the episode-based OnlineReplay used by
        train_joint.py: that one preserves contiguous time because the RSSM
        needs 64-step sequences to roll its recurrent state forward. Classic
        SAC bootstraps one step at a time, so it only ever needs independent
        transitions and can sample them uniformly at random. """

    def __init__(self, obs_dim, action_dim, capacity=REPLAY_CAPACITY):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.not_done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        # Success bookkeeping for reporting only -- mirrors the
        # replay/success_frac_* metrics train_joint.py logs, so the baseline
        # and the world-model run can be compared on the same axes.
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
        # not_done gates the bootstrap. Only genuine termination kills it --
        # truncation at the 200-step limit means the episode did not really
        # end, so the value function must still bootstrap past it.
        self.not_done[i] = 0.0 if terminated else 1.0
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
        """ Warm start from the static OGBench dataset. Flattened straight
            into the ring buffer -- no episode structure is needed here. """
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
        self.idx = n % self.capacity
        if n >= self.capacity:
            self.full = True

        # per-episode success counts, split on the dataset's terminal flags
        ep_ok = False
        for t in range(n):
            if rew[t, 0] > -1.0:
                ep_ok = True
            if term[t]:
                self.offline_episodes += 1
                self.offline_success += int(ep_ok)
                ep_ok = False
        return n

    def sample(self, batch_size, device, rng):
        idx = rng.integers(0, len(self), size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.obs), to(self.action), to(self.reward), to(self.next_obs), to(self.not_done)

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

@torch.no_grad()
def td_target(policy, reward, next_obs, not_done, gamma):
    """ Classic SAC 1-step TD target:

            target = r_t + gamma * not_done * [ min(Q1', Q2')(s', a')
                                                - alpha * log pi(a'|s') ]

        Exactly ONE real reward enters -- r_t. Everything beyond s' reaches
        this target only through the critic's own existing estimate, never as
        summed reward values. That is the whole difference from the
        lambda-returns train_joint.py uses over its imagined horizon, and it
        is the point of this baseline. """
    next_mu, next_std = policy.actor(next_obs)
    next_action, next_log_prob = sample_squashed(next_mu, next_std)
    target_Q1, target_Q2 = policy.critic_target(next_obs, next_action)
    target_V = torch.min(target_Q1, target_Q2) - policy.ent_coef.detach() * next_log_prob
    return reward + not_done * gamma * target_V

def eval_sac_in_env(env, policy, num_episodes, device, obs_key, record_video=False):
    """ Same as evaluation.eval_in_env but with the world-model bridge
        removed -- the actor reads raw observations directly. """
    returns, successes = [], []
    frames = []

    def safe_render():
        nonlocal record_video
        if not record_video:
            return
        try:
            frames.append(env.render())
        except Exception as e:
            print(f'Video recording failed, disabling for this eval: {e}')
            record_video = False

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_success = False

        if ep == 0:
            safe_render()

        while not done:
            state = extract_state(obs, obs_key).reshape(-1)
            action = policy.act(state, step=0, eval_mode=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_success = ep_success or bool(info.get('success', False))

            if ep == 0:
                safe_render()

            obs = next_obs

        returns.append(ep_return)
        successes.append(float(ep_success))

    video = None
    if record_video and frames:
        video = np.stack(frames).astype(np.uint8).transpose(0, 3, 1, 2)

    return float(np.mean(returns)), float(np.mean(successes)), video

def train(config):
    general_config, sac_config = config.train_sac.general, config.train_sac.sac
    # train_every=1 and batch_size=256 are SAC's standard settings and live
    # under train_sac.sac; every value there is deliberately kept equal
    # to train_joint.sac so the two arms differ only where they must.
    batch_size = sac_config.batch_size
    train_every = sac_config.train_every

    out_dir = pathlib.Path(general_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(config.seed)
    # Seeds torch (incl. CUDA), numpy's global RNG and Python's random.
    # Must run before SACWorldModelAgent is built so network init is covered.
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device}')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    print(f'env.observation_space = {env.observation_space}')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay = TransitionReplay(obs_dim, action_dim,
                              capacity=sac_config.replay_capacity)
    if train_dataset is not None:
        n = replay.seed_from_offline(train_dataset)
        print(f'Seeded replay buffer with {n} offline transitions '
              f'({replay.offline_episodes} episodes)')

    # repr_dim is the raw observation dim -- there is no world model, so the
    # actor and critic read the environment state directly.
    policy = SACWorldModelAgent(
        repr_dim=obs_dim, action_shape=(action_dim,), device=device,
        lr=sac_config.lr, feature_dim=sac_config.feature_dim, hidden_dim=sac_config.hidden_dim,
        critic_target_tau=sac_config.critic_target_tau,
        init_ent_coef=sac_config.init_ent_coef,
    )

    obs, info = env.reset(seed=config.seed)
    global_step = 0
    n_updates = 0
    print(f'Starting SAC-only training loop (no world model) | '
          f'train_every={train_every} batch_size={batch_size}')

    while global_step < general_config.num_train_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        if global_step < general_config.num_seed_steps:
            action = env.action_space.sample()
        else:
            action = policy.act(state, eval_mode=False)

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        if reward != -1.0:
            print(f'step {global_step:7d} | got reward {reward:.4f} | terminated={terminated}')
        replay.add(state, action, reward, np.asarray(next_obs, dtype=np.float32),
                   terminated, truncated)

        done = bool(terminated or truncated)
        obs = next_obs
        if done:
            obs, info = env.reset()

        global_step += 1
        metrics = {}

        if len(replay) >= batch_size and global_step % train_every == 0:
            b_obs, b_act, b_rew, b_next, b_nd = replay.sample(batch_size, device, rng)
            # Same {-1, 0} -> {0, +1} shift train_joint.py applies to its
            # imagined rewards, applied here to the real ones so the critic
            # in both runs sees the same reward scale.
            b_rew = b_rew + sac_config.reward_shift

            targets = td_target(policy, b_rew, b_next, b_nd, sac_config.gamma)

            # Classic SAC weights every transition equally; the weights
            # argument exists only because the imagined rollouts in
            # train_joint.py decay by cont probability over the horizon.
            weights = torch.ones_like(b_rew)
            metrics.update(_prefixed(policy.update_critic(b_obs, b_act, targets, weights), 'sac'))
            metrics.update(_prefixed(policy.update_actor(b_obs, weights), 'sac'))
            policy.update_target()
            n_updates += 1

            metrics['sac/mean_batch_reward'] = b_rew.mean().item()
            metrics['diagnosis/batch_reward_max'] = b_rew.max().item()
            metrics['diagnosis/batch_reward_min'] = b_rew.min().item()
            metrics['diagnosis/batch_not_done'] = b_nd.mean().item()

        if global_step % general_config.log_every == 0:
            metrics['diagnosis/replay_transitions'] = len(replay)
            # gradient updates so far -- the two arms differ in update cadence,
            # so equal env steps does NOT mean equal compute; report this
            # alongside any comparison rather than leaving it to be inferred
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
            mean_return, success_rate, video = eval_sac_in_env(
                env, policy, general_config.eval_episodes, device,
                OBS_KEY, record_video=True)
            print(f'step {global_step:7d} | eval_return {mean_return:.4f} | eval_success_rate {success_rate:.4f}')
            log_dict = {'eval/mean_return': mean_return, 'eval/success_rate': success_rate}
            if video is not None:
                log_dict['eval/video'] = wandb.Video(video, fps=20, format='mp4')
            wandb.log(log_dict, step=global_step)
            # eval_sac_in_env resets this same env internally, so the
            # training loop's obs is now stale -- the next step would store a
            # transition whose state does not match what the env actually
            # stepped from. Re-sync before resuming collection.
            obs, info = env.reset()

        if global_step % general_config.save_every == 0 and global_step > 0:
            torch.save(policy.state_dict_all(), out_dir / 'sac_latest.pt')
            print(f'Saved checkpoint at step {global_step}')

    torch.save(policy.state_dict_all(), out_dir / 'sac_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac.py --train_sac.general.env_name=cube-single-play-singletask-v0