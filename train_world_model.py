"""
train_world_model.py — Train ONLY the DreamerV3 world model on OGBench data.

This mirrors the WORLD MODEL portion of DreamerV3's training loop
(agent.py loss(), world-model section). It does NOT train a policy —
no imagination, no actor, no critic. Just Encoder + RSSM + Decoder +
reward head + continue head, trained on offline OGBench sequences.

PIPELINE
--------
  1. DatasetMethods.ensure_datasets()  — download OGBench dataset
  2. ogbench.make_env_and_datasets()    — load env + relabelled dataset
  3. DatasetMethods.extract_episodes()  — split into per-episode dicts
  4. Build a flat replay of episodes, sample (B, T) sequences
  5. Train world model: encode → RSSM observe → decode + heads → losses
  6. Backprop dyn + rep + rew + con + recon jointly

LOSS SCALES (DreamerV3 defaults, config.loss_scales):
  recon (per obs key) = 1.0
  reward              = 1.0
  cont                = 1.0
  dyn                 = 0.5
  rep                 = 0.1

WHAT TO WATCH
-------------
  recon  — should drop steadily (RSSM learning to represent obs)
  rew    — should drop (reward head learning sparse -1/0 signal)
  dyn    — prior chasing posterior; stabilizes near free_nats (1.0)
  rep    — posterior regularized to prior; stabilizes near free_nats
  con    — continue head; drops as it learns episode terminations

TO RUN
------
    pip install ogbench requests tqdm torch
    python train_world_model.py
"""

import os
import numpy as np
import torch
import ogbench

from ogbench_dataset_methods import DatasetMethods
from rssm import WorldModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_NAME = 'cube-single-play-singletask-task1-v0'

# --- world model architecture (DreamerV3 defaults) ---
DETER      = 4096
HIDDEN     = 2048
STOCH      = 32
CLASSES    = 32
BLOCKS     = 8
UNIMIX     = 0.01
FREE_NATS  = 1.0
UNITS      = 1024

# --- training ---
LR           = 1e-4
BATCH_SIZE   = 16        # sequences per batch (DreamerV3 batch_size)
SEQ_LEN      = 64        # timesteps per sequence (DreamerV3 batch_length)
GRAD_CLIP    = 1000.0    # DreamerV3 uses agc; we use norm clip
TRAIN_STEPS  = 500_000
LOG_EVERY    = 200
SAVE_EVERY   = 50_000
SAVE_DIR     = 'checkpoints'

# --- loss scales (DreamerV3 config.loss_scales) ---
SCALE_RECON  = 1.0
SCALE_REW    = 1.0
SCALE_CON    = 1.0
SCALE_DYN    = 0.5
SCALE_REP    = 0.1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Sequence sampler over extracted episodes
# ---------------------------------------------------------------------------

class EpisodeSequenceSampler:
    """
    Holds the OGBench dataset as flat arrays plus episode boundaries,
    and samples (B, T) sequences that do not cross episode boundaries.

    We derive is_first from the episode structure: the first timestep of
    every sampled window that aligns with an episode start is marked.
    To keep it simple and correct, we sample windows fully WITHIN a single
    episode, and mark is_first=True only at t=0 of the window if that
    coincides with the true episode start (otherwise the RSSM just carries
    a zero state which is fine because we always start fresh per window).
    """

    def __init__(self, dataset, seq_len, device):
        self.seq_len = seq_len
        self.device  = device

        self.obs        = dataset['observations'].astype(np.float32)
        self.actions    = dataset['actions'].astype(np.float32)
        self.rewards    = dataset['rewards'].astype(np.float32)
        self.terminals  = dataset['terminals'].astype(np.float32)

        # episode start indices via DatasetMethods.extract_episodes logic:
        # episodes end where terminals == 1.0
        ends = np.where(self.terminals == 1.0)[0]
        starts = np.concatenate([[0], ends[:-1] + 1])
        self.episode_ranges = [
            (s, e) for s, e in zip(starts, ends) if (e - s + 1) >= seq_len
        ]
        assert len(self.episode_ranges) > 0, \
            "No episodes long enough for the requested SEQ_LEN"

        self.obs_dim    = self.obs.shape[-1]
        self.action_dim = self.actions.shape[-1]

    def sample(self, batch_size):
        """Sample (B, T) sequences fully inside single episodes."""
        obs_b, act_b, rew_b, term_b, first_b = [], [], [], [], []

        for _ in range(batch_size):
            s, e   = self.episode_ranges[np.random.randint(len(self.episode_ranges))]
            start  = np.random.randint(s, e - self.seq_len + 2)
            sl     = slice(start, start + self.seq_len)

            obs_b.append(self.obs[sl])
            act_b.append(self.actions[sl])
            rew_b.append(self.rewards[sl])
            term_b.append(self.terminals[sl])

            # is_first: True only at window position 0 (fresh RSSM state)
            first = np.zeros(self.seq_len, dtype=np.float32)
            first[0] = 1.0
            first_b.append(first)

        to_t = lambda arr: torch.from_numpy(np.stack(arr)).to(self.device)
        return (
            to_t(obs_b),    # (B, T, obs_dim)
            to_t(act_b),    # (B, T, action_dim)
            to_t(rew_b),    # (B, T)
            to_t(term_b),   # (B, T)
            to_t(first_b),  # (B, T)
        )


# ---------------------------------------------------------------------------
# Training step (mirrors world-model section of agent.loss)
# ---------------------------------------------------------------------------

def train_step(world_model, optimizer, sampler):
    obs, actions, rewards, terminals, is_first = sampler.sample(BATCH_SIZE)

    losses, _ = world_model.loss(
        obs, actions, rewards, is_first, terminals
    )

    total = (
        SCALE_RECON * losses['recon'] +
        SCALE_REW   * losses['rew']   +
        SCALE_CON   * losses['con']   +
        SCALE_DYN   * losses['dyn']   +
        SCALE_REP   * losses['rep']
    )

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {
        'total': total.item(),
        'recon': losses['recon'].item(),
        'rew':   losses['rew'].item(),
        'con':   losses['con'].item(),
        'dyn':   losses['dyn'].item(),
        'rep':   losses['rep'].item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # 1. download dataset via DatasetMethods
    train_path, val_path = DatasetMethods.ensure_datasets(ENV_NAME)

    # 2. load env + relabelled dataset (pass dataset_path to skip OGBench download)
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        ENV_NAME,
        dataset_path=train_path,
    )
    print("Dataset keys:", list(train_dataset.keys()))

    # 3. build sequence sampler (uses extract_episodes logic internally)
    sampler = EpisodeSequenceSampler(train_dataset, SEQ_LEN, DEVICE)
    obs_dim    = sampler.obs_dim
    action_dim = sampler.action_dim

    print(f"obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"usable episodes (>= {SEQ_LEN} steps): {len(sampler.episode_ranges)}")
    print(f"latent feat dim: {DETER + STOCH * CLASSES}")

    # 4. build world model
    world_model = WorldModel(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        deter      = DETER,
        hidden     = HIDDEN,
        stoch      = STOCH,
        classes    = CLASSES,
        blocks     = BLOCKS,
        unimix     = UNIMIX,
        free_nats  = FREE_NATS,
        units      = UNITS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(world_model.parameters(), lr=LR)

    # 5. training loop
    print(f"\nTraining world model only — {TRAIN_STEPS} steps")
    print(f"batch={BATCH_SIZE} seqs x {SEQ_LEN} steps\n")

    for step in range(1, TRAIN_STEPS + 1):
        m = train_step(world_model, optimizer, sampler)

        if step % LOG_EVERY == 0:
            print(f"step {step:6d} | total {m['total']:.3f} | "
                  f"recon {m['recon']:.3f} | rew {m['rew']:.3f} | "
                  f"con {m['con']:.3f} | dyn {m['dyn']:.3f} | rep {m['rep']:.3f}")

        if step % SAVE_EVERY == 0:
            path = os.path.join(SAVE_DIR, f'wm_step{step}.pt')
            torch.save({
                'step':        step,
                'env_name':    ENV_NAME,
                'obs_dim':     obs_dim,
                'action_dim':  action_dim,
                'world_model': world_model.state_dict(),
                'optimizer':   optimizer.state_dict(),
            }, path)
            print(f"  saved {path}")

    print("\nWorld model training complete.")


if __name__ == '__main__':
    main()