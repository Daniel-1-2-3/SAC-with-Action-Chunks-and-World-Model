import os
import requests
import numpy as np
import ogbench
import jax.numpy as jnp
import elements

class DatasetMethods:
    """
    Converts OGBench transition datasets into Dreamer-style flat replay batches.

    OGBench format, with flat arrays:
        observations[t]
        actions[t]
        rewards[t]
        next_observations[t]
        terminals[t]
        masks[t] 

    Dreamer Agent.train() expects the following flat data dict:
        data["state"]       (B, T, obs_dim)
        data["action"]      (B, T, action_dim)
        data["reward"]      (B, T)
        data["is_first"]    (B, T)
        data["is_last"]     (B, T)
        data["is_terminal"] (B, T)
        data["stepid"]      (B, T, 20)
        data["consec"]      (B, T)
    """

    @staticmethod
    def get_dataset_file_name(env_name: str) -> str:
        splits = env_name.split("-")

        if "singletask" in splits:
            pos = splits.index("singletask")
            dataset_name = "-".join(splits[:pos] + splits[-1:])
        else:
            dataset_name = env_name

        return dataset_name

    @staticmethod
    def download_file(url: str, dest_path: str) -> None:
        r = requests.get(url, stream=True)
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = downloaded / total * 100
                    mb = downloaded // 1024 // 1024
                    tot = total // 1024 // 1024
                    print(f"\r  {pct:.1f}% ({mb}MB/{tot}MB)", end="", flush=True)
        print()

    @staticmethod
    def ensure_datasets(env_name: str):
        cache_dir = os.path.join(os.path.expanduser("~"), ".ogbench", "data")
        os.makedirs(cache_dir, exist_ok=True)

        dataset_name = DatasetMethods.get_dataset_file_name(env_name)
        train_path = os.path.join(cache_dir, f"{dataset_name}.npz")
        val_path = os.path.join(cache_dir, f"{dataset_name}-val.npz")

        base_dir = "https://rail.eecs.berkeley.edu/datasets/ogbench"

        if not os.path.exists(train_path):
            print(f"Downloading train dataset: {dataset_name}.npz")
            DatasetMethods.download_file(f"{base_dir}/{dataset_name}.npz", train_path)

        if not os.path.exists(val_path):
            print(f"Downloading val dataset: {dataset_name}-val.npz")
            DatasetMethods.download_file(f"{base_dir}/{dataset_name}-val.npz", val_path)

        print(f"Train path: {os.path.basename(train_path)}")
        print(f"Val path: {os.path.basename(val_path)}")

        return train_path, val_path

    @staticmethod
    def load_ogbench(env_name: str, render_mode: str = "rgb_array"):
        train_path, val_path = DatasetMethods.ensure_datasets(env_name)

        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            env_name,
            dataset_path=train_path,
            render_mode=render_mode,
        )

        return env, train_dataset, val_dataset

    @staticmethod
    def flat_dataset_to_episodes(dataset: dict):
        required_keys = ["observations", "actions", "rewards", "next_observations", "terminals"]

        for key in required_keys:
            if key not in dataset:
                raise KeyError(f"Missing required dataset key: {key}")

        terminals = np.asarray(dataset["terminals"]).astype(bool)
        episode_ends = np.where(terminals)[0]
        episodes = []
        start = 0

        for end in episode_ends:
            ep = {k: np.asarray(v[start:end + 1]) for k, v in dataset.items()}
            episodes.append(ep)
            start = end + 1

        if start < len(terminals):
            ep = {k: np.asarray(v[start:]) for k, v in dataset.items()}
            episodes.append(ep)

        return episodes

    @staticmethod
    def ogbench_to_dreamer_episode(
        ep: dict,
        obs_key: str = "state",
        action_key: str = "action",
    ):
        """
        Converts one raw OGBench transition episode into a Dreamer replay episode. 
        For N OGBench transitions, this creates N + 1 Dreamer time steps. This 
        conversion is done as follows:

            Dreamer state[0]        = OGBench observations[0]
            Dreamer state[t + 1]    = OGBench next_observations[t]
            
            Dreamer action[t]       = OGBench actions[t]
            Dreamer action[-1]      = zero padding (Dreamer dataset gets more 1 more
                                      timestep and there is action to assign)

            Dreamer reward[0]       = 0 (Dreamer uses "reward for arriving at state[t + 1]"
                                      so there isn't a reward at the state[0])
            Dreamer reward[t + 1]   = OGBench rewards[t]
        """
        
        obs = np.asarray(ep["observations"], dtype=np.float32)
        next_obs = np.asarray(ep["next_observations"], dtype=np.float32)
        actions = np.asarray(ep["actions"], dtype=np.float32)
        rewards = np.asarray(ep["rewards"], dtype=np.float32)
        terminals = np.asarray(ep["terminals"]).astype(bool)

        if len(obs) == 0:
            raise ValueError("Found empty episode.")

        num_transitions = len(obs)
        obs_dim = obs.shape[-1]
        action_dim = actions.shape[-1]

        # N transitions become N + 1 Dreamer observations
        state_seq = np.concatenate([obs[:1], next_obs], axis=0).astype(np.float32)
        
        zero_action = np.zeros((1, action_dim), dtype=np.float32) # Process action
        action_seq = np.concatenate([actions, zero_action], axis=0).astype(np.float32)
        reward_seq = np.concatenate([np.zeros((1,), dtype=np.float32), rewards], axis=0).astype(np.float32) # Process reward

        # Mark start and end of episodes
        is_first = np.zeros((num_transitions + 1,), dtype=bool)
        is_first[0] = True
        is_last = np.concatenate([np.zeros((1,), dtype=bool), terminals], axis=0)

        # Boundaries where action is no longer consecutive is_terminal
        if "masks" in ep:
            masks = np.asarray(ep["masks"], dtype=np.float32)
            is_terminal = np.concatenate([np.zeros((1,), dtype=bool), masks < 0.5], axis=0)
            discount = np.concatenate([np.ones((1,), dtype=np.float32), masks], axis=0)
        else:
            is_terminal = is_last.copy()
            discount = 1.0 - is_terminal.astype(np.float32)

        stepid = np.zeros((num_transitions + 1, 20), dtype=np.uint8) # (T, 20), dtype uint8
        consec = np.arange(num_transitions + 1, dtype=np.int32)

        return {
            obs_key: state_seq,
            action_key: action_seq,
            "reward": reward_seq,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "discount": discount.astype(np.float32),
            "stepid": stepid,
            "consec": consec,
        }

    @staticmethod
    def make_dreamer_episodes(
        dataset: dict,
        min_length: int,
        obs_key: str = "state",
        action_key: str = "action",
    ):
        transition_episodes = DatasetMethods.flat_dataset_to_episodes(dataset)
        dreamer_episodes = []
        for ep in transition_episodes:
            dreamer_ep = DatasetMethods.ogbench_to_dreamer_episode(ep, obs_key=obs_key, action_key=action_key)
            if len(dreamer_ep[obs_key]) >= min_length:
                dreamer_episodes.append(dreamer_ep)

        return dreamer_episodes

    @staticmethod
    def sample_dreamer_batch(
        episodes,
        batch_size: int,
        seq_len: int,
        obs_key: str = "state",
        action_key: str = "action",
        rng: np.random.Generator | None = None,
        force_reset_at_chunk_start: bool = True,
    ):
        """
        Refer to class description for the shape of the 
        dictionary returned for a dreamer batch. 
        """
        
        if rng is None:
            rng = np.random.default_rng()

        batch = {obs_key: [],
            action_key: [],
            "reward": [],
            "is_first": [],
            "is_last": [],
            "is_terminal": [],
            "discount": [],
            "stepid": [],
            "consec": [],
        }

        for _ in range(batch_size):
            ep = episodes[rng.integers(0, len(episodes))]
            ep_len = len(ep[obs_key])

            if ep_len < seq_len:
                raise ValueError(f"Episode length {ep_len} is shorter than seq_len {seq_len}.")

            start = rng.integers(0, ep_len - seq_len + 1)
            end = start + seq_len
            chunk = {k: np.asarray(v[start:end]).copy() for k, v in ep.items()}

            if force_reset_at_chunk_start:
                chunk["is_first"][0] = True
                chunk["reward"][0] = 0.0
                chunk["is_terminal"][0] = False
                chunk["is_last"][0] = False
                chunk["discount"][0] = 1.0
                chunk["consec"] = np.arange(seq_len, dtype=np.int32)

            for k in batch:
                batch[k].append(chunk[k])

        out = {}

        out[obs_key] = np.stack(batch[obs_key], axis=0).astype(np.float32)
        out[action_key] = np.stack(batch[action_key], axis=0).astype(np.float32)

        out["reward"] = np.stack(batch["reward"], axis=0).astype(np.float32)
        out["is_first"] = np.stack(batch["is_first"], axis=0).astype(bool)
        out["is_last"] = np.stack(batch["is_last"], axis=0).astype(bool)
        out["is_terminal"] = np.stack(batch["is_terminal"], axis=0).astype(bool)
        out["discount"] = np.stack(batch["discount"], axis=0).astype(np.float32)

        out["stepid"] = np.stack(batch["stepid"], axis=0).astype(np.uint8)
        out["consec"] = np.stack(batch["consec"], axis=0).astype(np.int32)

        return out

    @staticmethod
    def sample_jax_dreamer_batch(
        episodes,
        batch_size: int,
        seq_len: int,
        obs_key: str = "state",
        action_key: str = "action",
        rng: np.random.Generator | None = None,
        force_reset_at_chunk_start: bool = True,
    ):
        batch = DatasetMethods.sample_dreamer_batch(
            episodes=episodes,
            batch_size=batch_size,
            seq_len=seq_len,
            obs_key=obs_key,
            action_key=action_key,
            rng=rng,
            force_reset_at_chunk_start=force_reset_at_chunk_start,
        )

        return DatasetMethods.to_jax(batch)

    @staticmethod
    def make_spaces(obs_dim: int, action_dim: int, obs_key: str = "state", action_key: str = "action"):
        """
        Creates obs_space and act_space matching the flat batch, 
        used for initializing Dreamer agent or the RSSM
        """

        obs_space = {
            obs_key: elements.Space(np.float32, (obs_dim,)),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, (), 0, 2),
            "is_last": elements.Space(bool, (), 0, 2),
            "is_terminal": elements.Space(bool, (), 0, 2),
        }

        act_space = {
            action_key: elements.Space(np.float32, (action_dim,)),
        }

        return obs_space, act_space

    @staticmethod
    def print_summary(dataset: dict, episodes=None, obs_key: str = "state"):
        obs_dim = dataset["observations"].shape[-1]
        action_dim = dataset["actions"].shape[-1]
        total_transitions = len(dataset["observations"])

        print("Dataset keys:", list(dataset.keys()))
        print(f"obs_dim={obs_dim}, action_dim={action_dim}")
        print(f"total_transitions={total_transitions}")

        if episodes is not None:
            lengths = np.array([len(ep[obs_key]) for ep in episodes])
            print(f"usable Dreamer episodes: {len(episodes)}")
            print(f"min length:  {lengths.min()}")
            print(f"mean length: {lengths.mean():.1f}")
            print(f"max length:  {lengths.max()}")
    
    @staticmethod
    def to_jax(batch: dict): # Converts numpy batch to jax arrays
        out = {}
        for k, v in batch.items():
            if v.dtype == np.bool_:
                out[k] = jnp.asarray(v, dtype=bool)
            elif v.dtype == np.uint8:
                out[k] = jnp.asarray(v, dtype=jnp.uint8)
            elif v.dtype == np.int32:
                out[k] = jnp.asarray(v, dtype=jnp.int32)
            else:
                out[k] = jnp.asarray(v, dtype=jnp.float32)

        return out