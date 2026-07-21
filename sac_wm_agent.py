import torch
import torch.nn as nn
from sac_wm_utils import soft_update_params, weight_init

# SB3's exact bounds (stable_baselines3/sac/policies.py)
LOG_STD_MIN = -20
LOG_STD_MAX = 2
# SB3's exact epsilon for the squash correction (common/distributions.py)
EPSILON = 1e-6

def sample_squashed(mu, std):
    gaussian = torch.distributions.Normal(mu, std)
    x = gaussian.rsample()
    action = torch.tanh(x)
    log_prob = gaussian.log_prob(x).sum(-1, keepdim=True)
    log_prob -= torch.log(1 - action.pow(2) + EPSILON).sum(-1, keepdim=True)
    return action, log_prob

class RunningScale:
    # tracks a running (EMA) estimate of the critic target's std, so the
    # critic can regress onto target / scale (bounded) instead of the
    # raw target (which can drift/compound with nothing anchoring it)
    def __init__(self, rate=0.01, min_scale=1.0):
        self.rate = rate
        self.min_scale = min_scale
        self.mean = 0.0
        self.mean_sq = 0.0
        self.initialized = False

    def update(self, x):
        with torch.no_grad():
            batch_mean = x.mean().item()
            batch_mean_sq = (x ** 2).mean().item()
        if not self.initialized:
            self.mean, self.mean_sq = batch_mean, batch_mean_sq
            self.initialized = True
        else:
            self.mean = (1 - self.rate) * self.mean + self.rate * batch_mean
            self.mean_sq = (1 - self.rate) * self.mean_sq + self.rate * batch_mean_sq

    @property
    def scale(self):
        var = max(self.mean_sq - self.mean ** 2, 0.0)
        return max(var ** 0.5, self.min_scale)

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.mu = nn.Linear(hidden_dim, action_shape[0])
        self.log_std = nn.Linear(hidden_dim, action_shape[0])
        self.apply(weight_init)

    def forward(self, feat):
        h = self.net(self.trunk(feat))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std.exp()

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

class SACWorldModelAgent:
    """ SAC policy trained entirely from imagined rollouts through a
        jointly-trained Dreamer world model. """

    def __init__(self, repr_dim, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, init_ent_coef=1.0):
        self.device = device
        self.critic_target_tau = critic_target_tau
        # SB3's 'auto' target_entropy default
        self.target_entropy = -float(action_shape[0])

        self.actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.log_ent_coef = torch.log(torch.ones(1, device=device) * init_ent_coef).requires_grad_(True)
        self.ent_coef_opt = torch.optim.Adam([self.log_ent_coef], lr=lr)

        self.target_scale = RunningScale()  # norm fix: tracks critic target scale

        self.train()
        self.critic_target.train()

    @property
    def ent_coef(self):
        return self.log_ent_coef.exp()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def act(self, feat, eval_mode, step=None):
        """ feat: (repr_dim,) numpy array from WorldModelBridge.get_feat """
        feat_t = torch.as_tensor(feat, device=self.device).float().unsqueeze(0)
        mu, std = self.actor(feat_t)
        if eval_mode:
            action = torch.tanh(mu)
        else:
            action, _ = sample_squashed(mu, std)
        return action.cpu().numpy()[0]

    def update_critic(self, feat, action, reward, discount, next_feat, weight):
        metrics = {}
        scale = self.target_scale.scale  # norm fix: current scale, used consistently below
        with torch.no_grad():
            next_mu, next_std = self.actor(next_feat)
            next_action, next_log_prob = sample_squashed(next_mu, next_std)
            target_Q1, target_Q2 = self.critic_target(next_feat, next_action)
            target_V = torch.min(target_Q1, target_Q2) * scale - self.ent_coef.detach() * next_log_prob  # norm fix: denormalize critic_target's output
            target_Q = reward + discount * target_V
            self.target_scale.update(target_Q)  # norm fix: refresh scale for next call

        target_Q_normed = target_Q / scale  # norm fix: critic regresses onto the normalized target

        Q1, Q2 = self.critic(feat, action)
        wsum = weight.sum().clamp_min(1e-6)
        critic_loss = ((weight * (Q1 - target_Q_normed) ** 2).sum() / wsum +
                        (weight * (Q2 - target_Q_normed) ** 2).sum() / wsum)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = (Q1 * scale).mean().item()  # norm fix: denormalized for readability
        metrics['critic_q2'] = (Q2 * scale).mean().item()
        metrics['diagnosis/critic_grad_norm'] = critic_grad_norm.item()
        metrics['diagnosis/critic_q1_std'] = (Q1 * scale).std().item()
        metrics['diagnosis/critic_q2_std'] = (Q2 * scale).std().item()
        metrics['diagnosis/critic_target_q_min'] = target_Q.min().item()
        metrics['diagnosis/critic_target_q_max'] = target_Q.max().item()
        metrics['diagnosis/target_scale'] = scale  # norm fix
        return metrics

    def update_actor(self, feat, weight):
        metrics = {}
        mu, std = self.actor(feat)
        action, log_prob = sample_squashed(mu, std)
        Q1, Q2 = self.critic(feat, action)
        Q = torch.min(Q1, Q2) * self.target_scale.scale  # norm fix: denormalize before mixing with the (real-unit) entropy term
        wsum = weight.sum().clamp_min(1e-6)
        actor_loss = (weight * (self.ent_coef.detach() * log_prob - Q)).sum() / wsum

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        self.actor_opt.step()

        # Updates entropy coefficient
        ent_coef_loss = -(self.log_ent_coef * (weight * (log_prob + self.target_entropy)).detach()).sum() / wsum
        self.ent_coef_opt.zero_grad(set_to_none=True)
        ent_coef_loss.backward()
        self.ent_coef_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = -log_prob.mean().item()
        metrics['diagnosis/actor_grad_norm'] = actor_grad_norm.item()
        metrics['diagnosis/ent_coef'] = self.ent_coef.item()
        metrics['diagnosis/ent_coef_loss'] = ent_coef_loss.item()
        # the actual tanh-saturation early-warning signal: watch this
        # for growth over time, and log_std_mean for collapse toward
        # LOG_STD_MIN (over-confident, under-exploring).
        metrics['diagnosis/actor_mu_abs_mean'] = mu.abs().mean().item() # mu is mean of the action dist
        metrics['diagnosis/actor_mu_abs_max'] = mu.abs().max().item()
        metrics['diagnosis/actor_std_mean'] = std.mean().item()
        return metrics

    def update_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def state_dict_all(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_ent_coef': self.log_ent_coef.detach().cpu(),
            'target_scale_mean': self.target_scale.mean,
            'target_scale_mean_sq': self.target_scale.mean_sq,
            'target_scale_initialized': self.target_scale.initialized,
        }

    def load_state_dict_all(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        if 'log_ent_coef' in state:
            with torch.no_grad():
                self.log_ent_coef.copy_(state['log_ent_coef'].to(self.device))
        if 'target_scale_mean' in state:
            self.target_scale.mean = state['target_scale_mean']
            self.target_scale.mean_sq = state['target_scale_mean_sq']
            self.target_scale.initialized = state['target_scale_initialized']
