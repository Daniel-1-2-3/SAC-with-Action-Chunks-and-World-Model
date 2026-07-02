import pathlib
import argparse
import numpy as np
import elements
import ruamel.yaml as yaml
from ogbench_dataset_methods import DatasetMethods
from agent import WorldModelAgent

def load_config(folder, presets=None):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    config = elements.Config(configs['defaults'])
    for name in (presets or []):
        config = config.update(configs[name])
    return config

def build_agent_config(config, batch_size, seq_len, logdir):
    return elements.Config(
        **config.agent,
        logdir=str(logdir),
        seed=config.seed,
        jax=config.jax,
        batch_size=batch_size,
        batch_length=seq_len,
        replay_context=0,
        report_length=seq_len,
        replica=0,
        replicas=1,
    )

def unwrap(v):
    """
    np.asarray() on a dict/list produces a 0-d object array wrapping the
    original Python object instead of flattening it, undo that action
    """
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
        return v.item()
    return v

def validate_heads(env_name, obs_key, action_key, presets, seed, ckpt, n_batches):
    folder = pathlib.Path(__file__).parent
    logdir = folder / 'world_model_train_out'

    config = load_config(folder, presets)
    batch_size = config.batch_size
    seq_len = config.batch_length

    print(f'Loading OGBench dataset: {env_name}')
    env, train_dataset, val_dataset = DatasetMethods.load_ogbench(env_name)
    obs_dim = train_dataset['observations'].shape[-1]
    action_dim = train_dataset['actions'].shape[-1]
    env.close()

    val_episodes = DatasetMethods.make_dreamer_episodes(
        val_dataset, min_length=seq_len, obs_key=obs_key, action_key=action_key)

    obs_space, act_space = DatasetMethods.make_spaces(
        obs_dim, action_dim, obs_key, action_key)

    agent_config = build_agent_config(config, batch_size, seq_len, logdir)
    agent = WorldModelAgent(obs_space, act_space, agent_config)

    if ckpt:
        print(f'Loading checkpoint: {ckpt}')
        raw = np.load(ckpt, allow_pickle=True)
        state = {k: unwrap(raw[k]) for k in raw.files}
        agent.load(state)
    else:
        print('No checkpoint given')
        return

    rng = np.random.default_rng(seed)
    carry = agent.init_report(batch_size)

    all_true_rew, all_pred_rew = [], []
    all_true_con, all_pred_con = [], []

    for i in range(n_batches):
        batch = DatasetMethods.sample_jax_dreamer_batch(
            val_episodes, batch_size, seq_len, obs_key, action_key, rng=rng)
        batch.pop('discount')
        batch['seed'] = agent._seeds(i, agent.train_mirrored)

        carry, mets = agent.report(carry, batch)

        if 'pred/rew' not in mets:
            raise KeyError("Rewards cannot be accessed, metrics['pred/rew'] does not exist")

        all_true_rew.append(np.asarray(batch['reward']))
        all_pred_rew.append(np.asarray(mets['pred/rew']))

        if 'pred/con' in mets:
            true_con = ~np.asarray(batch['is_terminal'])
            all_true_con.append(true_con.astype(np.float32))
            all_pred_con.append(np.asarray(mets['pred/con']))

    true_rew = np.concatenate([t.reshape(-1) for t in all_true_rew])
    pred_rew = np.concatenate([p.reshape(-1) for p in all_pred_rew])
    success_mask = np.isclose(true_rew, 0.0)
    common_mask = np.isclose(true_rew, -1.0)

    report_path = folder / 'validate_heads_report.txt'
    with open(report_path, 'w') as f:

        f.write(f"Total timesteps checked: {len(true_rew)}\n")
        f.write(f"'At target' (reward=0) timesteps: {success_mask.sum()} ({100*success_mask.mean():.2f}%)\n")
        f.write(f"'Not at target:' (reward=-1) timesteps: {common_mask.sum()} ({100*common_mask.mean():.2f}%)\n")
        f.write("\n")
        f.write("The following logged metrics looks at transitions where the reward\n")
        f.write("is -1.0 (not at target) and 0.0 (at target). If the mean predicted reward for 0.0\n")
        f.write("at target situations (which is rare) is near 0.0, it means that the model is\n")
        f.write("actually predicting 0.0 when it is supposed to, and not blindly outputing -1.0\n")
        f.write("\n")

        f.write("reward = -1.0, not at target:\n")
        if common_mask.sum() > 0:
            f.write(f"Predicted reward mean: {pred_rew[common_mask].mean():.5f} (true: -1.0)\n")
            f.write(f"Predicted reward std: {pred_rew[common_mask].std():.5f}\n")
        f.write("\n")

        f.write("reward = 0.0, at target:\n")
        if success_mask.sum() > 0:
            f.write(f"Count: {success_mask.sum()}\n")
            f.write(f"Predicted reward mean: {pred_rew[success_mask].mean():.5f} (true: 0.0)\n")
            f.write(f"Predicted reward std: {pred_rew[success_mask].std():.5f}\n")
            mae_success = np.abs(true_rew[success_mask] - pred_rew[success_mask]).mean()
            baseline_mae_success = np.abs(true_rew[success_mask] - (-1.0)).mean()
            f.write(f"MAE (model): {mae_success:.5f}\n")
            f.write(f"MAE (always predict -1.0): {baseline_mae_success:.5f} (majority-class baseline)\n")
        f.write("\n")

        if all_pred_con:
            true_con = np.concatenate([t.reshape(-1) for t in all_true_con])
            pred_con = np.concatenate([p.reshape(-1) for p in all_pred_con])
            terminal_mask = true_con < 0.5
            if terminal_mask.sum() > 0:
                f.write(f"Terminal timesteps found: {terminal_mask.sum()}\n")
                f.write(f"Predicted continue-prob at true terminals (should be low): {pred_con[terminal_mask].mean():.5f}\n")

    print(f"Written to {report_path.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--obs_key', type=str, default='state')
    parser.add_argument('--action_key', type=str, default='action')
    parser.add_argument('--presets', type=str, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--n_batches', type=int, default=20)
    args = parser.parse_args()
    validate_heads(**vars(args))