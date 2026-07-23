import torch
import torch.nn as nn
from sac_wm_utils import soft_update_params, weight_init

# SB3's exact bounds (stable_baselines3/sac/policies.py)
LOG_STD_MIN = -20
LOG_STD_MAX = 2
# SB3's exact epsilon for the squash correction (common/distributions.py)
EPSILON = 1e-6
# Bug 1 fix: mu was unclamped, letting pre-tanh means grow unbounded and
# saturate tanh. Matches this project's earlier fix (clip_mean=2.0).
MU_CLIP = 2.0
# Bug 2 fix: replaces the previous float('inf') no-op clip.
GRAD_CLIP_NORM = 10.0

def sample_squashed(mu, std):
    gaussian = torch.distributions.Normal(mu, std)
    x = gaussian.rsample()
    action = torch.tanh(x)
    log_prob = gaussian.log_prob(x).sum(-1, keepdim=True)
    log_prob -= torch.log(1 - action.pow(2) + EPSILON).sum(-1, keepdim=True)
    return action, log_prob

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
        raw_mu = self.mu(h)
        # DIAGNOSTIC (temporary): stored so update_actor can report pre-clamp
        # statistics -- this is the only way to tell "clamp is masking an
        # unbounded raw output" apart from "raw output is naturally bounded
        # and the clamp rarely engages". No behavior change: mu returned
        # below is still clamped exactly as before.
        self.raw_mu = raw_mu
        mu = torch.clamp(raw_mu, -MU_CLIP, MU_CLIP)
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
        with torch.no_grad():
            next_mu, next_std = self.actor(next_feat)
            next_action, next_log_prob = sample_squashed(next_mu, next_std)
            target_Q1, target_Q2 = self.critic_target(next_feat, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.ent_coef.detach() * next_log_prob
            target_Q = reward + discount * target_V

        Q1, Q2 = self.critic(feat, action)
        wsum = weight.sum().clamp_min(1e-6)
        critic_loss = ((weight * (Q1 - target_Q) ** 2).sum() / wsum +
                        (weight * (Q2 - target_Q) ** 2).sum() / wsum)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        # Bug 2 fix: was float('inf'), which made this a no-op (norm always
        # computed, never actually clipped). Now a real threshold.
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_NORM)
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['diagnosis/critic_grad_norm'] = critic_grad_norm.item()
        metrics['diagnosis/critic_q1_std'] = Q1.std().item()
        metrics['diagnosis/critic_q2_std'] = Q2.std().item()
        metrics['diagnosis/critic_target_q_min'] = target_Q.min().item()
        metrics['diagnosis/critic_target_q_max'] = target_Q.max().item()
        # DIAGNOSTIC (temporary): single number for the min/max-convergence
        # collapse signature -- expect this to shrink toward 0 at the same
        # point mu explodes, per the earlier target_q_min/max analysis.
        metrics['diagnose_actor_mu_explosion/critic_target_q_range'] = (target_Q.max() - target_Q.min()).item()
        return metrics

    def update_actor(self, feat, weight):
        metrics = {}
        mu, std = self.actor(feat)
        raw_mu = self.actor.raw_mu
        action, log_prob = sample_squashed(mu, std)
        Q1, Q2 = self.critic(feat, action)
        Q = torch.min(Q1, Q2)
        wsum = weight.sum().clamp_min(1e-6)
        # DIAGNOSTIC (temporary): split into its two components so we can see
        # exactly when the Q-term starts dominating the entropy term, rather
        # than only seeing their already-summed total.
        ent_term = (weight * self.ent_coef.detach() * log_prob).sum() / wsum
        q_term = (weight * (-Q)).sum() / wsum
        actor_loss = ent_term + q_term

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        # Bug 2 fix: same as critic above — real clip threshold instead of inf.
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
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
        # DIAGNOSTIC (temporary): everything below is new, all under the tab
        # requested for this investigation.
        metrics['diagnose_actor_mu_explosion/actor_mu_raw_abs_mean'] = raw_mu.detach().abs().mean().item()
        metrics['diagnose_actor_mu_explosion/actor_mu_raw_abs_max'] = raw_mu.detach().abs().max().item()
        # fraction of the batch where the clamp actually engaged -- ties
        # directly to "did mu abs_mean also increase drastically" from
        # earlier: a rising fraction here is the population-wide version,
        # not just a few outlier samples.
        metrics['diagnose_actor_mu_explosion/frac_actions_saturated'] = (raw_mu.detach().abs() > MU_CLIP).float().mean().item()
        metrics['diagnose_actor_mu_explosion/actor_std_min'] = std.min().item()
        # positive gap = measured entropy still above target = ent_coef still
        # being pushed down. Watch for this crossing zero right as mu climbs.
        metrics['diagnose_actor_mu_explosion/entropy_gap'] = (-log_prob.mean() - self.target_entropy).item()
        metrics['diagnose_actor_mu_explosion/actor_loss_ent_term'] = ent_term.item()
        metrics['diagnose_actor_mu_explosion/actor_loss_q_term'] = q_term.item()
        return metrics

    def update_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def state_dict_all(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_ent_coef': self.log_ent_coef.detach().cpu(),
        }

    def load_state_dict_all(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        if 'log_ent_coef' in state:
            with torch.no_grad():
                self.log_ent_coef.copy_(state['log_ent_coef'].to(self.device))