""" DrQV2 actor-critic without CNN encoder or random shift
    augmentation, update() removed since update is controlled
    in train_joint.py """

import torch
import torch.nn as nn
from drqv2_wm_utils import TruncatedNormal, schedule, soft_update_params, weight_init

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]))
        self.apply(weight_init)

    def forward(self, feat, std):
        h = self.trunk(feat)
        mu = torch.tanh(self.policy(h))
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.apply(weight_init)

    def forward(self, feat, action):
        h = self.trunk(feat)
        h_action = torch.cat([h, action], dim=-1)
        return self.Q1(h_action), self.Q2(h_action)

class DrQV2WorldModelAgent:
    """ DrQV2 policy trained entirely from imagined rollouts through a
        jointly-trained Dreamer world model """

    def __init__(self, repr_dim, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 stddev_schedule, stddev_clip, use_tb=False):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb

        self.actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def act(self, feat, step, eval_mode):
        """ feat: (repr_dim,) numpy array from WorldModelBridge.get_feat """
        feat_t = torch.as_tensor(feat, device=self.device).float().unsqueeze(0)
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(feat_t, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, feat, action, reward, discount, next_feat, step, weight):
        metrics = {}
        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            next_dist = self.actor(next_feat, stddev)
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_feat, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + discount * target_V

        Q1, Q2 = self.critic(feat, action)
        wsum = weight.sum().clamp_min(1e-6)
        critic_loss = ((weight * (Q1 - target_Q) ** 2).sum() / wsum +
                        (weight * (Q2 - target_Q) ** 2).sum() / wsum)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
        return metrics

    def update_actor(self, feat, step, weight):
        metrics = {}
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(feat, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(feat, action)
        Q = torch.min(Q1, Q2)
        wsum = weight.sum().clamp_min(1e-6)
        actor_loss = (weight * -Q).sum() / wsum

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        if self.use_tb:
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def update_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def state_dict_all(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }

    def load_state_dict_all(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
