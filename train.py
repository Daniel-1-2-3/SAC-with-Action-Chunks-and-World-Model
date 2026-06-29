"""
train.py — World model training on OGBench offline dataset.

PIPELINE OVERVIEW
-----------------
This follows the Q-chunking approach exactly:
  1. Load a pre-collected OGBench offline dataset (no env interaction needed)
  2. Load all transitions into the replay buffer
  3. Train the world model by sampling sequences from the buffer

OGBench provides datasets collected by scripted/play policies that actually
solve the task — so the world model sees successful manipulation trajectories
from the start, unlike random action collection.

SETUP
-----
Install OGBench:
    pip install ogbench

Available cube manipulation tasks (matching Q-chunking paper):
    cube-single-play-singletask-task1-v0
    cube-double-play-singletask-task1-v0
    cube-triple-play-singletask-task1-v0   ← hardest, used in Q-chunking paper
    cube-quadruple-play-singletask-task1-v0

TO RUN
------
    python train.py

WHAT TO WATCH
-------------
  recon  — should drop steadily (encoder + decoder learning)
  reward — should drop (reward head learning)
  kl     — should stabilize around FREE_NATS value
           if it hits 0: stochastic state is being ignored
           if it explodes: prior and posterior are diverging
"""

import os
import numpy as np
import torch
import ogbench

from world_model import WorldModel
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# OGBench task — start with single cube, easiest to learn from
# Change to cube-double or cube-triple for harder tasks matching the paper
ENV_NAME = 'cube-single-play-singletask-task1-v0'

# world model architecture
EMBED_DIM  = 256    # encoder MLP output size (e_t)
DETER_DIM  = 512    # GRU hidden state size (h_t)
STOCH_DIM  = 32     # number of categorical groups in z_t
CLASSES    = 32     # values per group (z_t is 32x32 = 1024 flat)
HIDDEN_DIM = 512    # MLP hidden size in RSSM posterior/prior/decoder
FREE_NATS  = 1.0    # KL loss floor — prevents stochastic state collapse

# world model training
WM_LR          = 1e-4
WM_BATCH_SIZE  = 16     # sequences per batch
WM_SEQ_LEN     = 50     # timesteps per sequence
WM_GRAD_CLIP   = 100.0
WM_RECON_COEF  = 1.0
WM_REWARD_COEF = 1.0
WM_KL_COEF     = 0.1

# training
TRAIN_STEPS  = 500_000   # gradient steps on the offline dataset
LOG_EVERY    = 500
SAVE_EVERY   = 50_000
SAVE_DIR     = 'checkpoints'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ogbench(env_name):
    """
    Load OGBench environment and offline dataset.

    ogbench.make_env_and_datasets() returns:
      env        — the gymnasium environment for online interaction later
      eval_env   — a separate env instance for evaluation
      train_data — dict with keys:
                     observations      (N, obs_dim)
                     actions           (N, action_dim)
                     rewards           (N,)
                     next_observations (N, obs_dim)
                     terminals         (N,)
                     masks             (N,)  1=episode continues, 0=episode ended
      val_data   — same format, smaller validation split

    The dataset is collected by play/scripted policies that actually solve
    the task, so it contains successful cube manipulation trajectories.
    This is exactly how Q-chunking loads their data.
    """
    env, eval_env, train_data, val_data = ogbench.make_env_and_datasets(env_name)
    return env, eval_env, train_data, val_data


# ---------------------------------------------------------------------------
# World model training step
# ---------------------------------------------------------------------------

def train_world_model(world_model, optimizer, replay_buffer):
    """
    One world model gradient update.

    Samples a batch of sequences from the offline replay buffer,
    runs the full forward pass, computes three losses, backprops.
    """
    if len(replay_buffer) < WM_SEQ_LEN * WM_BATCH_SIZE:
        return {}

    # sample consecutive sequences — RSSM needs temporal order
    obs_seq, action_seq, reward_seq, _, _, is_first_seq = \
        replay_buffer.sample_sequences(WM_BATCH_SIZE, WM_SEQ_LEN)

    optimizer.zero_grad()

    # forward pass: encode → RSSM → decoder + reward head → losses
    losses, _ = world_model(obs_seq, action_seq, reward_seq, is_first_seq)

    total_loss = (
        WM_RECON_COEF  * losses['reconstruction'] +
        WM_REWARD_COEF * losses['reward']         +
        WM_KL_COEF     * losses['kl']
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), WM_GRAD_CLIP)
    optimizer.step()

    return {
        'total':  total_loss.item(),
        'recon':  losses['reconstruction'].item(),
        'reward': losses['reward'].item(),
        'kl':     losses['kl'].item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # --- Load OGBench dataset ---
    print(f"Loading OGBench dataset: {ENV_NAME}")
    env, eval_env, train_data, val_data = load_ogbench(ENV_NAME)

    # get obs and action dims from the dataset itself
    # this way we don't hardcode dims and it works for any OGBench task
    obs_dim    = train_data['observations'].shape[-1]
    action_dim = train_data['actions'].shape[-1]
    N          = len(train_data['observations'])

    print(f"Dataset size:     {N} transitions")
    print(f"Obs dim:          {obs_dim}")
    print(f"Action dim:       {action_dim}")
    print(f"Latent state dim: {DETER_DIM + STOCH_DIM * CLASSES}")

    # --- Replay buffer ---
    # capacity = dataset size + some headroom for future online transitions
    replay_buffer = ReplayBuffer(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        capacity   = N + 10_000,
        device     = DEVICE,
    )

    # load the full offline dataset into the buffer
    # this handles is_first derivation from masks automatically
    replay_buffer.load_from_dataset(train_data)

    # --- World model ---
    world_model = WorldModel(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        embed_dim  = EMBED_DIM,
        deter_dim  = DETER_DIM,
        stoch_dim  = STOCH_DIM,
        classes    = CLASSES,
        hidden_dim = HIDDEN_DIM,
        free_nats  = FREE_NATS,
    ).to(DEVICE)

    # single Adam optimizer for all world model parameters jointly
    optimizer = torch.optim.Adam(world_model.parameters(), lr=WM_LR)

    # --- Training loop ---
    # pure offline: sample from dataset, train world model, no env interaction
    print(f"\nStarting world model training for {TRAIN_STEPS} steps...")
    print(f"Batch size: {WM_BATCH_SIZE} sequences x {WM_SEQ_LEN} timesteps\n")

    for step in range(1, TRAIN_STEPS + 1):

        metrics = train_world_model(world_model, optimizer, replay_buffer)

        if step % LOG_EVERY == 0 and metrics:
            print(f"Step {step:6d} | "
                  f"total={metrics['total']:.4f} | "
                  f"recon={metrics['recon']:.4f} | "
                  f"reward={metrics['reward']:.4f} | "
                  f"kl={metrics['kl']:.4f}")

        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f'world_model_step{step}.pt')
            torch.save({
                'step':        step,
                'env_name':    ENV_NAME,
                'obs_dim':     obs_dim,
                'action_dim':  action_dim,
                'world_model': world_model.state_dict(),
                'optimizer':   optimizer.state_dict(),
            }, ckpt_path)
            print(f"  [Saved: {ckpt_path}]")

    print("\nWorld model training complete.")


if __name__ == '__main__':
    main()