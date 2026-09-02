import os
import requests
import numpy as np
import ogbench

class OGBenchMethods:
    """ Dataset download and OGBench env construction. """

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

        dataset_name = OGBenchMethods.get_dataset_file_name(env_name)
        train_path = os.path.join(cache_dir, f"{dataset_name}.npz")
        val_path = os.path.join(cache_dir, f"{dataset_name}-val.npz")

        base_dir = "https://rail.eecs.berkeley.edu/datasets/ogbench"

        if not os.path.exists(train_path):
            print(f"Downloading train dataset: {dataset_name}.npz")
            OGBenchMethods.download_file(f"{base_dir}/{dataset_name}.npz", train_path)

        if not os.path.exists(val_path):
            print(f"Downloading val dataset: {dataset_name}-val.npz")
            OGBenchMethods.download_file(f"{base_dir}/{dataset_name}-val.npz", val_path)

        print(f"Train path: {os.path.basename(train_path)}")
        print(f"Val path: {os.path.basename(val_path)}")

        return train_path, val_path

    @staticmethod
    def load_ogbench(env_name: str, render_mode: str = "rgb_array"):
        train_path, val_path = OGBenchMethods.ensure_datasets(env_name)

        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            env_name,
            dataset_path=train_path,
            render_mode=render_mode,
        )

        return env, train_dataset, val_dataset
