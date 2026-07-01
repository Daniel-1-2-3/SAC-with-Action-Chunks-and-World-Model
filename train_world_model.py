import pathlib
import argparse
import elements
import numpy as np
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

def train(
    env_name: str,
    obs_key: str = 'state',
    action_key: str = 'action',
    seq_len: int = None,
    batch_size: int = None,
    train_steps: int = 50_000,
    log_every: int = 100,
    eval_every: int = 1_000,
    ckpt_every: int = 5_000,
    presets: list = None,
    seed: int = 0,
):
    folder = pathlib.Path(__file__).parent
    logdir = folder / 'world_model_train_out'
    logdir.mkdir(parents=True, exist_ok=True)

    config = load_config(folder, presets)
    batch_size = config.batch_size if batch_size is None else batch_size
    seq_len = config.batch_length if seq_len is None else seq_len

    print(f'Loading OGBench dataset: {env_name}')
    env, train_dataset, val_dataset = DatasetMethods.load_ogbench(env_name)
    obs_dim = train_dataset['observations'].shape[-1]
    action_dim = train_dataset['actions'].shape[-1]
    env.close()

    train_episodes = DatasetMethods.make_dreamer_episodes(
        train_dataset, min_length=seq_len, obs_key=obs_key, action_key=action_key)
    val_episodes = DatasetMethods.make_dreamer_episodes(
        val_dataset, min_length=seq_len, obs_key=obs_key, action_key=action_key)
    DatasetMethods.print_summary(train_dataset, train_episodes, obs_key=obs_key)

    obs_space, act_space = DatasetMethods.make_spaces(
        obs_dim, action_dim, obs_key, action_key)

    agent_config = build_agent_config(config, batch_size, seq_len, logdir)
    agent = WorldModelAgent(obs_space, act_space, agent_config)

    train_carry = agent.init_train(batch_size)
    eval_carry = agent.init_report(batch_size)
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1)

    def fmt(metrics, prefix='loss/'):
        items = {k[len(prefix):]: float(v) for k, v in metrics.items()
                if k.startswith(prefix)}
        return '  '.join(f'{k}={v:.4f}' for k, v in sorted(items.items()))

    print(f'Starting world model training for {train_steps} steps.')
    for step in range(1, train_steps + 1):
        batch = DatasetMethods.sample_jax_dreamer_batch(
            train_episodes, batch_size, seq_len, obs_key, action_key, rng=rng)
        batch.pop('discount')
        batch['seed'] = agent._seeds(step, agent.train_mirrored)
        train_carry, outs, metrics = agent.train(train_carry, batch)

        if step % log_every == 0:
            print(f'step {step:>7}  {fmt(metrics)}')

        if step % eval_every == 0:
            val_batch = DatasetMethods.sample_jax_dreamer_batch(
                val_episodes, batch_size, seq_len, obs_key, action_key, rng=eval_rng)
            val_batch.pop('discount')
            val_batch['seed'] = agent._seeds(step, agent.train_mirrored)
            eval_carry, eval_metrics = agent.report(eval_carry, val_batch)
            print(f'  [eval] step {step:>7}  {fmt(eval_metrics)}')

        if step % ckpt_every == 0:
            ckpt_path = logdir / f'checkpoint_{step}.npz'
            state = agent.save()
            np.savez(ckpt_path, **{k: np.asarray(v) for k, v in state.items()})
            print(f'  saved checkpoint: {ckpt_path}')

    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cube-single-singletask-v0')
    parser.add_argument('--obs_key', type=str, default='state')
    parser.add_argument('--action_key', type=str, default='action')
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--train_steps', type=int, default=50_000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1_000)
    parser.add_argument('--ckpt_every', type=int, default=5_000)
    parser.add_argument('--presets', type=str, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train(**vars(args))
    
    # python train_world_model.py --env_name cube-single-play-singletask-v0 --presets debug