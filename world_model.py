import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Compresses raw proprioceptive observations into a flat embedding vector.
    This is the e_t in our discussion — the encoded observation that gets
    passed into the RSSM to compute the posterior stochastic state z_t.

    For SawyerPickPlaceEnvV3 the obs is a 39-element flat vector (hand pos,
    gripper state, object pos/quat, previous obs, goal pos), so we use an MLP.
    """

    def __init__(self, obs_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, embed_dim),
            nn.ELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, obs):
        # obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
        return self.net(obs)


# ---------------------------------------------------------------------------
# RSSM (Recurrent State Space Model)
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """
    The core of the world model.

    Maintains a two-part latent state at every timestep:
      - h_t: deterministic state — the GRU hidden state, carries memory
      - z_t: stochastic state — sampled from a 32x32 categorical grid,
             captures uncertainty about the current world state

    The full state is s_t = concat(h_t, z_t_flat).

    During training (observe): uses real observations to compute the
    posterior z_t ~ q(z_t | h_t, e_t) for accuracy.

    During imagination (imagine): uses only h_t to compute the prior
    z_t ~ p(z_t | h_t), since no real observations are available.

    The KL loss between posterior and prior trains the prior to be good
    enough to stand in for the posterior during imagination.
    """

    def __init__(
        self,
        embed_dim,      # size of encoder output (e_t)
        action_dim,     # size of action vector
        deter_dim=512,  # size of GRU hidden state h_t (deterministic)
        stoch_dim=32,   # number of categorical groups
        classes=32,     # number of values per group
        hidden_dim=512, # size of intermediate MLP layers
    ):
        super().__init__()

        self.deter_dim  = deter_dim
        self.stoch_dim  = stoch_dim
        self.classes    = classes
        self.stoch_flat = stoch_dim * classes  # flattened z_t size

        # --- GRU input projection ---
        # Before the GRU, we project [h_{t-1}, z_{t-1}_flat, a_{t-1}]
        # into a single input vector. This is cleaner than feeding them
        # raw into the GRU and lets each modality have its own linear layer.
        self.gru_input_proj = nn.Linear(
            deter_dim + self.stoch_flat + action_dim, deter_dim
        )

        # --- GRU cell ---
        # This IS the deterministic state update.
        # h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
        # Takes the projected input and previous hidden state,
        # produces the new deterministic memory vector h_t.
        self.gru_cell = nn.GRUCell(deter_dim, deter_dim)

        # --- Posterior MLP ---
        # Computes z_t using h_t AND the real encoded observation e_t.
        # Used during world model training where real obs are available.
        # z_t ~ q(z_t | h_t, e_t)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * classes),  # outputs 32x32 logits
        )

        # --- Prior MLP ---
        # Computes z_t using ONLY h_t, no real observation.
        # Used during imagination when no real obs is available.
        # z_t ~ p(z_t | h_t)
        # Trained via KL loss to match the posterior.
        self.prior_mlp = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * classes),  # outputs 32x32 logits
        )

    @property
    def state_dim(self):
        """Dimension of the full latent state s_t = concat(h_t, z_t_flat)."""
        return self.deter_dim + self.stoch_flat

    def initial_state(self, batch_size, device):
        """
        At the very first timestep, h_0 and z_0 are initialized to zeros.
        The model learns to handle this through training since it sees
        episode starts regularly (marked by is_first flags).
        """
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_flat, device=device)
        return h, z

    def _gru_step(self, h, z, action):
        """
        One GRU update step.
        h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})

        Concatenates the three inputs, projects them, then passes through
        the GRU cell to produce the new deterministic hidden state.
        """
        # concatenate previous hidden state, previous stochastic state, action
        x = torch.cat([h, z, action], dim=-1)       # (B, deter + stoch_flat + action)
        x = F.elu(self.gru_input_proj(x))            # project to deter_dim
        h_new = self.gru_cell(x, h)                  # GRU update → new h_t
        return h_new

    def _sample_stoch(self, logits):
        """
        Given logits of shape (B, stoch_dim * classes), reshape to (B, stoch_dim, classes),
        apply softmax over each row (each group of 32 values), then sample one
        integer per group using the straight-through estimator so gradients
        can flow through the discrete sample.

        Returns:
          z_flat:  (B, stoch_flat) — the sampled one-hot flattened vector
          logits:  (B, stoch_dim, classes) — raw logits (used for KL loss)
          probs:   (B, stoch_dim, classes) — softmax probabilities
        """
        logits = logits.view(-1, self.stoch_dim, self.classes)
        probs  = F.softmax(logits, dim=-1)            # softmax over each row of 32

        # sample one index per group: shape (B, stoch_dim)
        indices = torch.distributions.Categorical(probs=probs).sample()

        # convert to one-hot so we can concat with h_t as a continuous vector
        z_onehot = F.one_hot(indices, num_classes=self.classes).float()

        # straight-through estimator: forward uses the sample,
        # backward pretends we used the probabilities (allows gradient flow)
        z_st = z_onehot + probs - probs.detach()

        z_flat = z_st.view(-1, self.stoch_flat)       # flatten 32x32 → 1024
        return z_flat, logits, probs

    def observe(self, embeds, actions, is_first, h=None, z=None):
        """
        Unroll the RSSM over a full sequence using REAL observations.
        This is used during world model training.

        For each timestep t:
          1. Reset h and z to zero if is_first[t] is True (episode boundary)
          2. h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
          3. Posterior: z_t ~ q(z_t | h_t, e_t)   ← uses real obs
          4. Prior:    z̃_t ~ p(z̃_t | h_t)         ← for KL loss only
          5. s_t = concat(h_t, z_t)

        Args:
          embeds:   (B, T, embed_dim) — encoded observations from encoder
          actions:  (B, T, action_dim) — actions taken
          is_first: (B, T) — True at episode starts, triggers state reset

        Returns:
          states:        list of T tensors (B, state_dim) — full latent states s_t
          post_logits:   list of T tensors (B, stoch_dim, classes) — posterior logits
          prior_logits:  list of T tensors (B, stoch_dim, classes) — prior logits
        """
        B, T, _ = embeds.shape
        device   = embeds.device

        # initialize to zeros if no carry provided
        if h is None or z is None:
            h, z = self.initial_state(B, device)

        states       = []
        post_logits  = []
        prior_logits = []

        for t in range(T):
            # reset state at episode boundaries
            # is_first[:,t] is True when a new episode starts at this step
            reset_mask = is_first[:, t].unsqueeze(-1).float()  # (B, 1)
            h = h * (1.0 - reset_mask)
            z = z * (1.0 - reset_mask)

            # --- Step 1: Update deterministic state ---
            # use action from previous step (at t=0 this is zeros due to reset)
            prev_action = actions[:, t]
            h = self._gru_step(h, z, prev_action)

            # --- Step 2: Compute prior (no real obs) ---
            # p(z̃_t | h_t) — what the model would guess without seeing obs
            prior_logit_raw = self.prior_mlp(h)                        # (B, stoch_flat)
            _, p_logits, _  = self._sample_stoch(prior_logit_raw)
            prior_logits.append(p_logits)

            # --- Step 3: Compute posterior (with real obs) ---
            # q(z_t | h_t, e_t) — accurate estimate using real observation
            e_t            = embeds[:, t]                              # encoded obs at t
            post_input     = torch.cat([h, e_t], dim=-1)
            post_logit_raw = self.posterior_mlp(post_input)
            z, q_logits, _ = self._sample_stoch(post_logit_raw)
            post_logits.append(q_logits)

            # --- Step 4: Form full state s_t = concat(h_t, z_t) ---
            s_t = torch.cat([h, z], dim=-1)
            states.append(s_t)

        # stack along time dimension: (B, T, dim)
        states      = torch.stack(states,      dim=1)
        post_logits = torch.stack(post_logits, dim=1)
        prior_logits= torch.stack(prior_logits,dim=1)

        return states, post_logits, prior_logits, h, z

    def imagine(self, h, z, actor, horizon):
        """
        Roll out the RSSM forward in IMAGINATION — no real environment needed.
        Uses only the PRIOR since there are no real observations.

        At each step:
          1. h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
          2. z_t ~ p(z_t | h_t)         ← prior only, no real obs
          3. s_t = concat(h_t, z_t)
          4. actor picks next action from s_t

        Args:
          h, z:    starting latent state from a real sequence
          actor:   DrQV2 actor — maps s_t to action distribution
          horizon: number of imagination steps (e.g. 15)

        Returns:
          imag_states:   (horizon, B, state_dim) — imagined latent states
          imag_actions:  (horizon, B, action_dim) — actions taken in imagination
          imag_rewards:  placeholder; filled in by reward head after this call
        """
        imag_states  = []
        imag_actions = []

        for _ in range(horizon):
            # actor picks action from current latent state (no grad through actor here)
            s_t    = torch.cat([h, z], dim=-1)
            action = actor(s_t)                    # (B, action_dim)

            # update deterministic state
            h = self._gru_step(h, z, action)

            # compute stochastic state from prior (no real obs available)
            prior_logit_raw = self.prior_mlp(h)
            z, _, _         = self._sample_stoch(prior_logit_raw)

            # form full state
            s_t = torch.cat([h, z], dim=-1)
            imag_states.append(s_t)
            imag_actions.append(action)

        imag_states  = torch.stack(imag_states,  dim=0)  # (H, B, state_dim)
        imag_actions = torch.stack(imag_actions, dim=0)  # (H, B, action_dim)

        return imag_states, imag_actions


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Reconstructs the observation from the full latent state s_t.
    Trained via reconstruction loss: predicted obs vs real obs.
    This forces the latent state to actually encode meaningful information
    about the world — if s_t didn't capture the obs well, the decoder
    loss would be high.
    """

    def __init__(self, state_dim, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, state):
        # state: (B, T, state_dim) or (B, state_dim)
        return self.net(state)


# ---------------------------------------------------------------------------
# Reward Head
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """
    Predicts the reward at each timestep from the latent state s_t.
    Trained alongside the world model so that when we later do imagination
    rollouts for policy training, the model can predict rewards internally
    without querying the real environment.

    As we discussed: reward is a function of the state, and predicting it
    from s_t (rather than reconstructed obs) is more accurate because the
    latent state is a cleaner signal than a lossy reconstruction.
    """

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        # state: (B, T, state_dim) or (B, state_dim)
        return self.net(state)


# ---------------------------------------------------------------------------
# World Model (puts it all together)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Full world model: Encoder + RSSM + Decoder + RewardHead.

    Training process (per batch):
      1. Encode sequence of obs → embeddings e_t
      2. Unroll RSSM with observe() → latent states s_t, posterior, prior
      3. Decode s_t → reconstructed obs
      4. Predict reward from s_t
      5. Compute three losses:
           - reconstruction loss: predicted obs vs real obs
           - reward loss:         predicted reward vs real reward
           - KL loss:             push prior toward posterior so imagination works
      6. Backprop through everything jointly
    """

    def __init__(self, obs_dim, action_dim, embed_dim=256,
                 deter_dim=512, stoch_dim=32, classes=32,
                 hidden_dim=512, free_nats=1.0):
        super().__init__()

        self.encoder     = Encoder(obs_dim, embed_dim)
        self.rssm        = RSSM(embed_dim, action_dim, deter_dim,
                                stoch_dim, classes, hidden_dim)
        self.decoder     = Decoder(self.rssm.state_dim, obs_dim, hidden_dim)
        self.reward_head = RewardHead(self.rssm.state_dim, hidden_dim)

        # free nats: don't penalize KL below this threshold early in training.
        # prevents the model from collapsing the stochastic state to zero
        # before it's had a chance to learn anything useful.
        self.free_nats = free_nats

    @property
    def state_dim(self):
        return self.rssm.state_dim

    def forward(self, obs_seq, action_seq, reward_seq, is_first):
        """
        Full world model forward pass and loss computation.

        Args:
          obs_seq:    (B, T, obs_dim)    — raw observations
          action_seq: (B, T, action_dim) — actions taken
          reward_seq: (B, T, 1)          — rewards received
          is_first:   (B, T)             — episode start flags

        Returns:
          losses: dict with reconstruction, reward, kl losses
          states: (B, T, state_dim) — latent states for downstream use
        """
        B, T, _ = obs_seq.shape

        # --- Step 1: Encode all observations ---
        # flatten to (B*T, obs_dim), encode, reshape back to (B, T, embed_dim)
        embeds = self.encoder(obs_seq.view(B * T, -1))
        embeds = embeds.view(B, T, -1)

        # --- Step 2: Unroll RSSM ---
        # gets latent states using posterior (real obs) for training accuracy
        states, post_logits, prior_logits, _, _ = self.rssm.observe(
            embeds, action_seq, is_first
        )

        # --- Step 3: Decode reconstructed observations ---
        # states: (B, T, state_dim) → decoder → (B, T, obs_dim)
        recon_obs = self.decoder(states)

        # --- Step 4: Predict rewards ---
        pred_reward = self.reward_head(states)  # (B, T, 1)

        # --- Step 5: Compute losses ---

        # Reconstruction loss: how well did we reconstruct the observation?
        # Using the obs at each step as the target (not next_obs — we're
        # reconstructing the current state, not predicting the future)
        recon_loss = F.mse_loss(recon_obs, obs_seq)

        # Reward loss: how well did we predict the reward?
        reward_loss = F.mse_loss(pred_reward, reward_seq)

        # KL loss: push prior toward posterior so imagination is possible later.
        # post_logits: (B, T, stoch_dim, classes) — what we know with real obs
        # prior_logits:(B, T, stoch_dim, classes) — what we'd guess without obs
        # We want the prior to be a good predictor of the posterior.
        post_dist  = torch.distributions.Categorical(logits=post_logits)
        prior_dist = torch.distributions.Categorical(logits=prior_logits)

        # KL(posterior || prior) — push prior to match posterior
        kl = torch.distributions.kl_divergence(post_dist, prior_dist)
        kl = kl.sum(dim=-1)  # sum over stoch_dim groups

        # free nats: clamp KL below threshold — don't train early on tiny KL
        kl = torch.clamp(kl, min=self.free_nats).mean()

        losses = {
            'reconstruction': recon_loss,
            'reward':         reward_loss,
            'kl':             kl,
        }

        return losses, states.detach()  # detach states for downstream DrQV2 use