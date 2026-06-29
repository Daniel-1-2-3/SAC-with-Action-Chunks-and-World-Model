import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.

    Supports two modes:
      1. load_from_dataset() — load a pre-collected offline dataset (OGBench style)
      2. add() — add transitions one at a time from live environment interaction

    For world model training we use sample_sequences() which returns
    consecutive timestep sequences so the RSSM can unroll properly over time.

    is_first tracks episode boundaries so the RSSM knows when to reset
    h and z to zero. In the OGBench offline dataset, episode boundaries
    are marked by the 'masks' field — mask=0 means the episode ended
    at that transition (i.e. the NEXT step is is_first=True).
    """

    def __init__(self, obs_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0
        self.size     = 0

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)
        self.is_first = np.zeros((capacity,),            dtype=np.float32)

    def load_from_dataset(self, dataset):
        """
        Load a pre-collected offline dataset into the buffer.

        OGBench datasets (and Q-chunking's Dataset class) have these keys:
          observations     : (N, obs_dim)
          actions          : (N, action_dim)
          rewards          : (N,)
          next_observations: (N, obs_dim)
          terminals        : (N,)  — True if truly terminal (rare in OGBench)
          masks            : (N,)  — 1.0 = episode continues, 0.0 = episode ended

        We derive is_first from masks: wherever masks[t] == 0, the next
        timestep t+1 is the start of a new episode, so is_first[t+1] = 1.
        The very first transition is always is_first=True.
        """
        obs      = np.array(dataset['observations'],      dtype=np.float32)
        actions  = np.array(dataset['actions'],           dtype=np.float32)
        rewards  = np.array(dataset['rewards'],           dtype=np.float32)
        next_obs = np.array(dataset['next_observations'], dtype=np.float32)
        masks    = np.array(dataset['masks'],             dtype=np.float32)
        terminals= np.array(dataset['terminals'],         dtype=np.float32)

        N = len(obs)
        assert N <= self.capacity, \
            f"Dataset size {N} exceeds buffer capacity {self.capacity}"

        # derive is_first from masks
        # mask=0 at step t means episode ended, so step t+1 is is_first
        is_first        = np.zeros(N, dtype=np.float32)
        is_first[0]     = 1.0                        # first transition always
        is_first[1:]    = 1.0 - masks[:-1]           # wherever previous mask=0

        self.obs     [:N] = obs
        self.next_obs[:N] = next_obs
        self.actions [:N] = actions
        self.rewards [:N] = rewards.reshape(N, 1)
        self.dones   [:N] = terminals.reshape(N, 1)
        self.is_first[:N] = is_first

        self.size = N
        self.ptr  = N % self.capacity

        print(f"Loaded {N} transitions from dataset.")
        print(f"Episode boundaries (is_first=True): {int(is_first.sum())}")

    def add(self, obs, action, reward, next_obs, done, is_first=False):
        """Add a single transition from live environment interaction."""
        self.obs     [self.ptr] = obs
        self.actions [self.ptr] = action
        self.rewards [self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones   [self.ptr] = done
        self.is_first[self.ptr] = float(is_first)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch of individual transitions for policy updates."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs     [idx]).to(self.device),
            torch.FloatTensor(self.actions [idx]).to(self.device),
            torch.FloatTensor(self.rewards [idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.dones   [idx]).to(self.device),
        )

    def sample_sequences(self, batch_size, seq_len):
        """
        Sample sequences of consecutive transitions for world model training.
        The RSSM needs temporal order to unroll correctly — individual
        transitions are not enough.

        Returns (B, T, dim) tensors including is_first so the RSSM can
        reset h and z at real episode boundaries within the sequence.
        """
        max_start = self.size - seq_len
        assert max_start > 0, "Not enough data for sequence sampling"

        starts = np.random.randint(0, max_start, size=batch_size)

        obs_seq      = np.stack([self.obs     [s:s+seq_len] for s in starts])
        action_seq   = np.stack([self.actions [s:s+seq_len] for s in starts])
        reward_seq   = np.stack([self.rewards [s:s+seq_len] for s in starts])
        next_obs_seq = np.stack([self.next_obs[s:s+seq_len] for s in starts])
        done_seq     = np.stack([self.dones   [s:s+seq_len] for s in starts])
        is_first_seq = np.stack([self.is_first[s:s+seq_len] for s in starts])

        return (
            torch.FloatTensor(obs_seq).to(self.device),
            torch.FloatTensor(action_seq).to(self.device),
            torch.FloatTensor(reward_seq).to(self.device),
            torch.FloatTensor(next_obs_seq).to(self.device),
            torch.FloatTensor(done_seq).to(self.device),
            torch.FloatTensor(is_first_seq).to(self.device),
        )

    def __len__(self):
        return self.size