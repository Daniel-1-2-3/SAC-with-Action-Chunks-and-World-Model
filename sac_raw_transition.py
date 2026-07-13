import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import ogbench
import wandb

import sac.utils as utils
from sac.sac import Critic, SACActor


def numeric_metrics(metrics, prefix=''):
    out = {}
    for k, v in metrics.items():
        try:
            out[f'{prefix}{k}'] = float(v)
        except (TypeError, ValueError):
            continue
    return out


class ReplayBuffer:
    """Just the fixed OGBench offline dataset, held in memory as tensors.
    No online collection, no world model -- sample a random batch of real,
    recorded transitions each step."""

    def __init__(self, dataset, device):
        self.observations = torch.as_tensor(dataset['observations'], dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(dataset['actions'], dtype=torch.float32, device=device)
        self.next_observations = torch.as_tensor(dataset['next_observations'], dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(dataset['rewards'], dtype=torch.float32, device=device).reshape(-1, 1)
        # IMPORTANT: bootstrap using 'masks', not 'terminals'. 'terminals' marks the end of
        # every fixed-length recorded trajectory chunk in the dataset (a data-collection
        # artifact), while 'masks' is 0 only at true task success and 1 everywhere else.
        # Bootstrapping off 'terminals' would falsely zero the value estimate at every
        # chunk boundary, not just real completions, and would depress Q everywhere.
        self.masks = torch.as_tensor(dataset['masks'], dtype=torch.float32, device=device).reshape(-1, 1)
        self.size = self.observations.shape[0]
        self.device = device

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        idx_t = torch.as_tensor(idx, device=self.device)
        return (self.observations[idx_t], self.actions[idx_t], self.rewards[idx_t],
                self.masks[idx_t], self.next_observations[idx_t])


class SACAgent:
    def __init__(self, obs_dim, action_shape, device, lr, alpha_lr, feature_dim,
                 hidden_dim, critic_target_tau, gamma, init_alpha=1.0, clip_mean=2.0,
                 target_entropy_start_scale=-0.5, target_entropy_end_scale=-1.0,
                 target_entropy_anneal_steps=50_000, use_tb=False):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.gamma = gamma
        self.use_tb = use_tb
        self.action_dim = float(action_shape[0])
        # target_entropy is annealed from start_scale*action_dim to end_scale*action_dim
        # over target_entropy_anneal_steps, then held fixed. See prior discussion: a fixed
        # target from step 0 lets alpha settle into whatever stable entropy level the
        # actor/critic loop finds first; annealing keeps some pressure against premature
        # convergence early on.
        self.target_entropy_start_scale = target_entropy_start_scale
        self.target_entropy_end_scale = target_entropy_end_scale
        self.target_entropy_anneal_steps = max(1, target_entropy_anneal_steps)

        self.actor = SACActor(obs_dim, action_shape, feature_dim, hidden_dim, clip_mean).to(device)
        self.critic = Critic(obs_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_shape, feature_dim, hidden_dim).to(device)
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

    def current_target_entropy(self, step):
        frac = min(1.0, step / self.target_entropy_anneal_steps)
        scale = (self.target_entropy_start_scale
                 + (self.target_entropy_end_scale - self.target_entropy_start_scale) * frac)
        return scale * self.action_dim

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
            mu, log_std = self.actor(obs)
            if eval_mode:
                action = torch.tanh(mu)
            else:
                action, _ = utils.sample_action(mu, log_std)
            return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, mask, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_obs)
            next_action, next_log_prob = utils.sample_action(next_mu, next_log_std)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
            target_Q = reward + self.gamma * mask * target_V

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        mu, log_std = self.actor(obs)
        action, log_prob = utils.sample_action(mu, log_std)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        target_entropy = self.current_target_entropy(step)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            h = self.actor.trunk(obs)
            mu_raw, _ = self.actor.policy(h).chunk(2, dim=-1)

        metrics['actor_loss'] = actor_loss.item()
        metrics['alpha'] = self.alpha.item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['target_entropy'] = target_entropy
        metrics['actor_pretanh_mean_abs'] = mu_raw.abs().mean().item()
        metrics['actor_pretanh_max_abs'] = mu_raw.abs().max().item()
        metrics['log_std_mean'] = log_std.mean().item()
        metrics['log_std_min'] = log_std.min().item()
        metrics['log_std_max'] = log_std.max().item()
        metrics['actor_logprob'] = log_prob.mean().item()
        return metrics

    def state_dict_all(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }


def eval_in_env(env, agent, num_episodes):
    returns, successes = [], []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_success = False
        while not done:
            action = agent.act(obs, eval_mode=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_success = ep_success or bool(info.get('success', reward == 0))
            obs = next_obs
        returns.append(ep_return)
        successes.append(float(ep_success))
    return float(np.mean(returns)), float(np.mean(successes))


def train(env_name, seed, num_train_steps, batch_size, log_every, eval_every, eval_episodes,
          save_every, out_dir, lr, alpha_lr, feature_dim, hidden_dim, critic_target_tau, gamma,
          init_alpha, clip_mean, target_entropy_start_scale, target_entropy_end_scale,
          target_entropy_anneal_steps,
          wandb_project, wandb_entity, wandb_run_name, wandb_mode):

    if target_entropy_anneal_steps is None:
        target_entropy_anneal_steps = num_train_steps // 2

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode=wandb_mode,
        config={
            'env_name': env_name,
            'seed': seed,
            'num_train_steps': num_train_steps,
            'batch_size': batch_size,
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
            'target_entropy_start_scale': target_entropy_start_scale,
            'target_entropy_end_scale': target_entropy_end_scale,
            'target_entropy_anneal_steps': target_entropy_anneal_steps,
        },
    )

    print(f'Loading OGBench env + dataset: {env_name}')
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
    obs_dim = train_dataset['observations'].shape[-1]
    action_dim = train_dataset['actions'].shape[-1]
    print(f'obs_dim={obs_dim}  action_dim={action_dim}  '
          f'transitions={train_dataset["observations"].shape[0]}')

    rng = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    buffer = ReplayBuffer(train_dataset, device)

    agent = SACAgent(
        obs_dim=obs_dim,
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
        target_entropy_start_scale=target_entropy_start_scale,
        target_entropy_end_scale=target_entropy_end_scale,
        target_entropy_anneal_steps=target_entropy_anneal_steps,
    )

    for step in range(num_train_steps):
        obs, action, reward, mask, next_obs = buffer.sample(batch_size, rng)

        metrics = agent.update_critic(obs, action, reward, mask, next_obs, step)
        metrics.update(agent.update_actor(obs, step))
        utils.soft_update_params(agent.critic, agent.critic_target, agent.critic_target_tau)
        metrics['batch_reward_mean'] = reward.mean().item()
        metrics['batch_mask_mean'] = mask.mean().item()

        if step % log_every == 0:
            print(f"step {step:6d} | critic_loss {metrics['critic_loss']:.4f} "
                  f"| actor_loss {metrics['actor_loss']:.4f} "
                  f"| alpha {metrics['alpha']:.4f} "
                  f"| log_std {metrics['log_std_mean']:.3f} "
                  f"| target_H {metrics['target_entropy']:.3f} "
                  f"| batch_reward {metrics['batch_reward_mean']:.4f}")
            wandb.log(numeric_metrics(metrics), step=step)

        if step % eval_every == 0 and step > 0:
            mean_return, success_rate = eval_in_env(env, agent, eval_episodes)
            print(f"step {step:6d} | eval_return {mean_return:.4f} "
                  f"| eval_success_rate {success_rate:.4f}")
            wandb.log({'eval/mean_return': mean_return, 'eval/success_rate': success_rate}, step=step)

        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f'policy_step{step}.pt'
            torch.save(agent.state_dict_all(), ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')
            wandb.summary['last_checkpoint_step'] = step

    torch.save(agent.state_dict_all(), out_dir / 'policy_final.pt')
    print(f"Done. Final policy saved to {out_dir / 'policy_final.pt'}")
    env.close()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_train_steps', type=int, default=500_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=25_000)
    parser.add_argument('--eval_every', type=int, default=5_000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='sac_raw_train_out')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha_lr', type=float, default=1e-3)
    parser.add_argument('--clip_mean', type=float, default=2.0)
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--critic_target_tau', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--init_alpha', type=float, default=1.0)
    parser.add_argument('--target_entropy_start_scale', type=float, default=-0.5,
                         help='target_entropy = scale * action_dim; less negative = looser/more entropy required')
    parser.add_argument('--target_entropy_end_scale', type=float, default=-1.0,
                         help='final scale once annealing completes (standard SAC default is -1.0)')
    parser.add_argument('--target_entropy_anneal_steps', type=int, default=None,
                         help='steps to linearly anneal from start_scale to end_scale; defaults to num_train_steps // 2')
    parser.add_argument('--wandb_project', type=str, default='ogbench-raw-sac')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online',
                         choices=['online', 'offline', 'disabled'])
    args = parser.parse_args()
    train(**vars(args))

    # python sac_raw_transitions.py --env_name cube-single-play-singletask-v0 --num_train_steps 500000