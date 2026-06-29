import os
import numpy as np
import requests
import ogbench
from tqdm import tqdm

class DatasetMethods:

    @staticmethod
    def get_dataset_file_name(env_name):
        splits = env_name.split('-')
        if 'singletask' in splits:
            pos = splits.index('singletask')
            dataset_name = '-'.join(splits[:pos] + splits[-1:])
        else:
            dataset_name = env_name
        return dataset_name

    @staticmethod
    def download_file(url, dest_path):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded // 1024 // 1024
                    tot = total // 1024 // 1024
                    print(f"\r  {pct:.1f}% ({mb}MB/{tot}MB)", end='', flush=True)

    @staticmethod
    def ensure_datasets(env_name):
        """
        Download train and val datasets if not 
        already cached. Returns (train_path, val_path).
        """
        cache_dir = os.path.join(os.path.expanduser("~"), ".ogbench", "data")
        os.makedirs(cache_dir, exist_ok=True)

        dataset_name = DatasetMethods.get_dataset_file_name(env_name)
        train_path = os.path.join(cache_dir, f"{dataset_name}.npz")
        val_path = os.path.join(cache_dir, f"{dataset_name}-val.npz")

        base_dir = "https://rail.eecs.berkeley.edu/datasets/ogbench"
        if not os.path.exists(train_path):
            DatasetMethods.download_file(f"{base_dir}/{dataset_name}.npz", train_path)
        if not os.path.exists(val_path):
            DatasetMethods.download_file(f"{base_dir}/{dataset_name}-val.npz", val_path)
            
        print(f"Train path: {os.path.basename(train_path)}") 
        print(f"Val path: {os.path.basename(val_path)}")
        
        return train_path, val_path

    @staticmethod
    def extract_episodes(dataset):
        """
        OGBench regular dataset (compact_dataset=False) has:
        observations: (N, obs_dim)
        actions: (N, action_dim)
        next_observations: (N, obs_dim)
        terminals: (N,) where 1.0 at episode end, 0.0 otherwise
        rewards: (N,) added by relabel_dataset automatically for all envs labeled singletask 
        
        Split flat dataset into episodes. Episode boundaries are where terminals == 1.0
        """
        terminals = dataset['terminals']
        episode_ends = np.where(terminals == 1.0)[0]

        episodes = []
        start = 0
        for end in episode_ends:
            episodes.append({k:v[start:end+1] for k,v in dataset.items()})
            start = end + 1

        if start < len(terminals):
            episodes.append({k: v[start:] for k, v in dataset.items()})

        return episodes

    @staticmethod
    def print_step(step, obs, reward, action):
        print(f"\nStep {step}")
        print(f"\tObs (ex):\t{np.round(obs[:6], 3)}")
        print(f"\tAction:\t\t{np.round(action, 3)}")
        print(f"\tReward:\t\t{reward:.4f}")

if __name__ == "__main__":
    env_name = 'cube-single-play-singletask-task1-v0'
    print_every = 50
    
    train_path, val_path = DatasetMethods.ensure_datasets(env_name)
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        env_name,
        dataset_path=train_path,
        render_mode='rgb_array',
    )
    
    # {observations, actions, terminals, next_observations, rewards, masks}
    print(train_dataset.keys())
    obs_dim = train_dataset['observations'].shape[-1]
    action_dim = train_dataset['actions'].shape[-1]
    tot_transitions = len(train_dataset['observations'])
    print(f"obs_dim {obs_dim}, action_dim {action_dim}, total_transitions {tot_transitions}")
    
    episodes = DatasetMethods.extract_episodes(train_dataset)
    print(f"Episodes count: {len(episodes)}")
    
    total_rewards = []
    for ep_idx in tqdm(range(len(episodes)), desc="Episodes"):
        episode = episodes[ep_idx]
        steps = len(episode['observations'])
        total_reward = 0
        
        for t in range(steps):
            obs = episode['observations'][t]
            action = episode['actions'][t]
            reward = episode['rewards'][t]
            total_reward += reward
            term = episode['terminals'][t]

            # DatasetMethods.print_step(t, obs, reward, action)
            if term == 1.0:
                break
        
        total_rewards.append(total_reward)
    print(f"Mean episodic reward: {np.mean(total_rewards)}")
