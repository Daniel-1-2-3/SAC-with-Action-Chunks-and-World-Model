import os
os.environ['MUJOCO_GL'] = 'egl'

import jax
print(jax.devices())

import argparse
import pathlib
import numpy as np
import jax.numpy as jnp
import ninjax as nj
import torch
import torch.nn.functional as F
import elements
import ruamel.yaml as yaml
import wandb

import drqv2.utils as utils
from drqv2.drqv2 import Critic, SACActor
from ogbench_dataset_methods import DatasetMethods
from agent import WorldModelAgent
from embodied.embodied.jax import transform

def load_config(folder, presets=None):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    config = elements.Config(configs['defaults'])
    for name in (presets or []):
        config = config.update(configs[name])
    return config

def build_agent_config(config, batch_size, seq_len, logdir):
    return elements.Config(
        **config.agent,
        logdir=str(logdir),
        seed=config.seed,
        jax=config.jax,
        batch_size=batch_size,
        batch_length=seq_len,
        replay_context=0,
        report_length=seq_len,
        replica=0,
        replicas=1,
    )

def unwrap(v):
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
        return v.item()
    return v

def numeric_metrics(metrics, prefix=''):
    out = {}
    for k, v in metrics.items():
        try:
            out[f'{prefix}{k}'] = float(v)
        except (TypeError, ValueError):
            continue
    return out

def jax_to_torch(x, device):
    x = jnp.asarray(x).astype(jnp.float32)
    return torch.as_tensor(jax.device_get(x).copy(), device=device).float()

def flatten_leading_two_dims_np(tree):
    return {k: jax.device_get(v).reshape((-1,) + v.shape[2:]) for k, v in tree.items()}

def subsample_tree_np(tree, n, rng):
    total = next(iter(tree.values())).shape[0]
    idx = rng.choice(total, size=min(n, total), replace=False)
    return {k: v[idx] for k, v in tree.items()}

def extract_state(obs, obs_key):
    if isinstance(obs, dict):
        return np.asarray(obs[obs_key], dtype=np.float32).reshape(1, -1)
    return np.asarray(obs, dtype=np.float32).reshape(1, -1)

class WorldModelBridge:
    def __init__(self, wm_agent, action_key):
        self.agent = wm_agent
        self.model = wm_agent.model
        self.action_key = action_key

        self.mesh = wm_agent.train_mesh
        self.ts = wm_agent.train_sharded
        tp = wm_agent.train_params_sharding
        tm = wm_agent.train_mirrored
        ar = wm_agent.partition_rules[1]

        self._encode_posterior = transform.apply(
            nj.pure(self.model.encode_posterior), self.mesh,
            (tp, tm, self.ts),
            (self.ts,),
            ar,
            static_argnums=(2,),
            single_output=True,
        )

        self._imagine_step = transform.apply(
            nj.pure(self.model.imagine_step), self.mesh,
            (tp, tm, self.ts, self.ts),
            (self.ts, self.ts, self.ts, self.ts),
            ar,
        )

        self._init_encode = transform.apply(
            nj.pure(self.model.init_encode), self.mesh,
            (tp, tm),
            (self.ts, self.ts),
            ar,
            static_argnums=(2,),
        )

        self._encode_step = transform.apply(
            nj.pure(self.model.encode_step), self.mesh,
            (tp, tm, self.ts, self.ts, self.ts, self.ts, self.ts),
            (self.ts, self.ts, self.ts),
            ar,
        )

        self._seed_counter = 0

    def _next_seed(self):
        self._seed_counter += 1
        return self.agent._seeds(self._seed_counter, self.agent.train_mirrored)

    def seed_pool(self, batch, batch_size):
        pool = self._encode_posterior(
            self.agent.params, self._next_seed(), batch_size, batch)
        return flatten_leading_two_dims_np(pool)

    def place_seed(self, seed_carry_np):
        return jax.device_put(seed_carry_np, self.ts)

    def get_feat(self, dyn_carry):
        return self.model.feat2tensor(dyn_carry)

    def img_step(self, dyn_carry, action_np):
        action = {self.action_key: action_np.astype(np.float32)}
        action = jax.device_put(action, self.ts)
        return self._imagine_step(
            self.agent.params, self._next_seed(), dyn_carry, action)

    def init_encode(self, batch_size):
        return self._init_encode(self.agent.params, self._next_seed(), batch_size)

    def encode_step(self, enc_carry, dyn_carry, state_np, action_np, is_first_np):
        obs = jax.device_put({'state': state_np.astype(np.float32)}, self.ts)
        prevact = jax.device_put({self.action_key: action_np.astype(np.float32)}, self.ts)
        is_first = jax.device_put(is_first_np.astype(bool), self.ts)
        return self._encode_step(
            self.agent.params, self._next_seed(), enc_carry, dyn_carry, obs, prevact, is_first)

class WorldModelSACAgent:
    def __init__(self, repr_dim, action_shape, device, lr, alpha_lr, feature_dim,
                 hidden_dim, critic_target_tau, gamma, init_alpha=0.1, clip_mean=2.0,
                 use_tb=False):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.gamma = gamma
        self.use_tb = use_tb
        self.target_entropy = -0.5 * float(action_shape[0])

        self.actor = SACActor(repr_dim, action_shape, feature_dim, hidden_dim, clip_mean).to(device)
        self.critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_alpha), device=device, requires_grad=True)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.train()
        self.critic_target.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, feat, step, eval_mode):
        with torch.no_grad():
            feat = torch.as_tensor(feat, device=self.device).unsqueeze(0).float()
            mu, log_std = self.actor(feat)
            if eval_mode:
                action = torch.tanh(mu)
            else:
                action, _ = utils.sample_action(mu, log_std)
            return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step, weight):
        metrics = dict()
        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_obs)
            next_action, next_log_prob = utils.sample_action(next_mu, next_log_std)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
            target_Q = reward + discount * target_V

        Q1, Q2 = self.critic(obs, action)
        wsum = weight.sum().clamp_min(1e-6)
        critic_loss = ((weight * (Q1 - target_Q) ** 2).sum() / wsum
                        + (weight * (Q2 - target_Q) ** 2).sum() / wsum)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
        return metrics

    def update_actor(self, obs, step, weight):
        metrics = dict()
        mu, log_std = self.actor(obs)
        action, log_prob = utils.sample_action(mu, log_std)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        wsum = weight.sum().clamp_min(1e-6)
        actor_loss = (weight * (self.alpha.detach() * log_prob - Q)).sum() / wsum

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            h = self.actor.trunk(obs)
            mu_raw, _ = self.actor.policy(h).chunk(2, dim=-1)

        metrics['actor_loss'] = actor_loss.item()
        metrics['alpha'] = self.alpha.item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['actor_pretanh_mean_abs'] = mu_raw.abs().mean().item()
        metrics['actor_pretanh_max_abs'] = mu_raw.abs().max().item()
        if self.use_tb:
            metrics['actor_logprob'] = log_prob.mean().item()
        return metrics

    def state_dict_all(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }

def imagine_rollout(bridge, actor, seed_carry, horizon, device, gamma):
    carry = seed_carry
    feat_t = jax_to_torch(bridge.get_feat(carry), device)

    feats, actions, rewards, conts, next_feats, weights = [], [], [], [], [], []
    cont_by_step = []

    weight = torch.ones(feat_t.shape[0], 1, device=device)

    for _ in range(horizon):
        mu, log_std = actor(feat_t)
        action_t, _ = utils.sample_action(mu, log_std)
        action_np = action_t.detach().cpu().numpy()

        next_carry, next_feat_flat, reward_j, cont_j = bridge.img_step(carry, action_np)
        next_feat_t = jax_to_torch(next_feat_flat, device)
        reward_t = jax_to_torch(reward_j, device).reshape(-1, 1)
        cont_t = jax_to_torch(cont_j, device).reshape(-1, 1)

        feats.append(feat_t)
        actions.append(action_t)
        rewards.append(reward_t)
        conts.append(cont_t)
        next_feats.append(next_feat_t)
        weights.append(weight)
        cont_by_step.append(cont_t.mean().item())

        weight = weight * (gamma * cont_t)
        carry, feat_t = next_carry, next_feat_t

    return (torch.cat(feats), torch.cat(actions), torch.cat(rewards),
            torch.cat(conts), torch.cat(next_feats), torch.cat(weights), cont_by_step)

def eval_in_env(env, bridge, policy, action_dim, num_episodes, device, obs_key, record_video=False):
    returns, successes = [], []
    frames = []
    imag_rewards_ep0 = []

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
        enc_carry, dyn_carry = bridge.init_encode(1)
        prevact = np.zeros((1, action_dim), dtype=np.float32)
        is_first = np.array([True])
        done = False
        ep_return = 0.0
        ep_success = False

        if ep == 0:
            safe_render()

        while not done:
            state = extract_state(obs, obs_key)
            enc_carry, dyn_carry, feat_j = bridge.encode_step(
                enc_carry, dyn_carry, state, prevact, is_first)
            feat_np = np.asarray(jax.device_get(feat_j))[0].copy()
            action = policy.act(feat_np, step=0, eval_mode=True)

            if ep == 0:
                _, _, imag_reward_j, _ = bridge.img_step(dyn_carry, action.reshape(1, -1))
                imag_rewards_ep0.append(float(jax.device_get(imag_reward_j)[0]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_success = ep_success or bool(info.get('success', reward == 0))

            if ep == 0:
                safe_render()

            prevact = action.reshape(1, -1).astype(np.float32)
            is_first = np.array([False])
            obs = next_obs

        returns.append(ep_return)
        successes.append(float(ep_success))

    video = None
    if record_video and frames:
        video = np.stack(frames).astype(np.uint8).transpose(0, 3, 1, 2)

    return float(np.mean(returns)), float(np.mean(successes)), video, imag_rewards_ep0

def train(env_name, obs_key, action_key, presets, seed, wm_ckpt, horizon,
          imagination_batch, seq_len_seed, num_train_steps, log_every,
          save_every, eval_every, eval_episodes, out_dir, lr, alpha_lr,
          feature_dim, hidden_dim, critic_target_tau, gamma, init_alpha, clip_mean,
          wandb_project, wandb_entity, wandb_run_name, wandb_mode):

    folder = pathlib.Path(__file__).parent
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(folder, presets)
    seq_len_seed = seq_len_seed or config.batch_length
    seed_batch_size = config.batch_size

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode=wandb_mode,
        config={
            'env_name': env_name,
            'obs_key': obs_key,
            'action_key': action_key,
            'seed': seed,
            'wm_ckpt': wm_ckpt,
            'horizon': horizon,
            'imagination_batch': imagination_batch,
            'seq_len_seed': seq_len_seed,
            'num_train_steps': num_train_steps,
            'eval_every': eval_every,
            'eval_episodes': eval_episodes,
            'lr': lr,
            'alpha_lr': alpha_lr,
            'feature_dim': feature_dim,
            'hidden_dim': hidden_dim,
            'critic_target_tau': critic_target_tau,
            'gamma': gamma,
            'init_alpha': init_alpha,
            'clip_mean': clip_mean,
            'presets': presets,
        },
    )

    print(f'Loading OGBench dataset: {env_name}')
    env, train_dataset, val_dataset = DatasetMethods.load_ogbench(env_name)
    obs_dim = train_dataset['observations'].shape[-1]
    action_dim = train_dataset['actions'].shape[-1]

    train_episodes = DatasetMethods.make_dreamer_episodes(
        train_dataset, min_length=seq_len_seed, obs_key=obs_key, action_key=action_key)

    obs_space, act_space = DatasetMethods.make_spaces(
        obs_dim, action_dim, obs_key, action_key)

    agent_config = build_agent_config(
        config, seed_batch_size, seq_len_seed, folder / 'world_model_train_out')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)

    print(f'Loading world model checkpoint: {wm_ckpt}')
    raw = np.load(wm_ckpt, allow_pickle=True)
    state = {k: unwrap(raw[k]) for k in raw.files}
    wm_agent.load(state)

    bridge = WorldModelBridge(wm_agent, action_key)

    rng = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    probe_batch = DatasetMethods.sample_jax_dreamer_batch(
        train_episodes, seed_batch_size, seq_len_seed, obs_key, action_key, rng=rng)
    probe_pool = bridge.seed_pool(probe_batch, seed_batch_size)
    probe_carry = bridge.place_seed(subsample_tree_np(probe_pool, 8, rng))
    feat_dim = int(bridge.get_feat(probe_carry).shape[-1])
    print(f'Detected latent feature dim: {feat_dim}')

    policy = WorldModelSACAgent(
        repr_dim=feat_dim,
        action_shape=(action_dim,),
        device=device,
        lr=lr,
        alpha_lr=alpha_lr,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        critic_target_tau=critic_target_tau,
        gamma=gamma,
        init_alpha=init_alpha,
        clip_mean=clip_mean,
    )

    for step in range(num_train_steps):
        batch = DatasetMethods.sample_jax_dreamer_batch(
            train_episodes, seed_batch_size, seq_len_seed, obs_key, action_key, rng=rng)

        seed_pool = bridge.seed_pool(batch, seed_batch_size)
        seed_pool = subsample_tree_np(seed_pool, imagination_batch, rng)
        seed_carry = bridge.place_seed(seed_pool)

        feats, actions, rewards, conts, next_feats, weights, cont_by_step = imagine_rollout(
            bridge, policy.actor, seed_carry, horizon, device, policy.gamma)

        discounts = policy.gamma * conts

        metrics = policy.update_critic(feats, actions, rewards, discounts, next_feats, step, weights)
        metrics.update(policy.update_actor(feats.detach(), step, weights.detach()))
        utils.soft_update_params(policy.critic, policy.critic_target, policy.critic_target_tau)
        metrics['mean_imag_reward'] = rewards.mean().item()
        metrics['mean_imag_cont'] = conts.mean().item()
        metrics['cont_horizon_first'] = cont_by_step[0]
        metrics['cont_horizon_last'] = cont_by_step[-1]
        metrics['mean_horizon_weight'] = weights.mean().item()

        if step % log_every == 0:
            print(f"step {step:6d} | critic_loss {metrics['critic_loss']:.4f} "
                  f"| actor_loss {metrics['actor_loss']:.4f} "
                  f"| alpha {metrics['alpha']:.4f} "
                  f"| mean_imag_reward {metrics['mean_imag_reward']:.4f}")
            wandb.log(numeric_metrics(metrics), step=step)

        if step % eval_every == 0 and step > 0:
            mean_return, success_rate, video, imag_rewards_ep0 = eval_in_env(
                env, bridge, policy, action_dim, eval_episodes, device, obs_key,
                record_video=True)
            print(f"step {step:6d} | eval_return {mean_return:.4f} "
                  f"| eval_success_rate {success_rate:.4f}")
            log_dict = {'eval/mean_return': mean_return,
                        'eval/success_rate': success_rate}
            if video is not None:
                log_dict['eval/video'] = wandb.Video(video, fps=20, format='mp4')
            if imag_rewards_ep0:
                xs = list(range(len(imag_rewards_ep0)))
                log_dict['eval/imagined_reward_per_step'] = wandb.plot.line_series(
                    xs=xs,
                    ys=[imag_rewards_ep0],
                    keys=['world_model_imagined_reward'],
                    title='Eval ep0: world-model-predicted reward per step',
                    xname='env step within episode')
            wandb.log(log_dict, step=step)

        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f'policy_step{step}.pt'
            torch.save(policy.state_dict_all(), ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')
            wandb.summary['last_checkpoint_step'] = step

    torch.save(policy.state_dict_all(), out_dir / 'policy_final.pt')
    print(f"Done. Final policy saved to {out_dir / 'policy_final.pt'}")
    env.close()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--obs_key', type=str, default='state')
    parser.add_argument('--action_key', type=str, default='action')
    parser.add_argument('--presets', type=str, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wm_ckpt', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--imagination_batch', type=int, default=2048)
    parser.add_argument('--seq_len_seed', type=int, default=None)
    parser.add_argument('--num_train_steps', type=int, default=100_000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='policy_train_out')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha_lr', type=float, default=1e-5)
    parser.add_argument('--clip_mean', type=float, default=2.0)
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--critic_target_tau', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--init_alpha', type=float, default=0.1)
    parser.add_argument('--wandb_project', type=str, default='world-model-policy')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online',
                         choices=['online', 'offline', 'disabled'])
    args = parser.parse_args()
    train(**vars(args))

    # python drqv2_world_model.py --env_name cube-single-play-singletask-v0 --wm_ckpt checkpoints_cube_single_play_v0/checkpoint_20000.npz --horizon 15 --num_train_steps 1000000