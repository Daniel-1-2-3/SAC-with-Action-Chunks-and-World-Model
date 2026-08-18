import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false') # Don't let JAX hog the GPU before torch inits
os.environ.setdefault('MUJOCO_GL', 'egl') # Headless for video rendering

import pathlib
import time
import elements
import jax
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import ogbench

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from sac_chunked.sac_chunk_agent import ChunkAgent
from sac_chunked.flow_bc import FlowBC
from helpers.sac_wm_utils import set_seed_everywhere
from sac_chunked.chunk_utils import chunk_lambda_targets, chunk_pair_indices
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from wm.imagination_chunk import imagine_chunk_rollout
from helpers.interop import jax_to_torch, numeric_metrics, subsample_tree_np, unwrap
from helpers.ogbench_methods import OGBenchMethods
from helpers.online_replay import OnlineReplay

OBS_KEY = 'state'
ACTION_KEY = 'action'
ENV_ACTION_LOW = -1.0
ENV_ACTION_HIGH = 1.0

def load_config(folder, argv=None):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
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

def build_real_env(env_name, load_offline_dataset):
    if load_offline_dataset:
        return OGBenchMethods.load_ogbench(env_name)
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    return env, None, None

def _param_norm(params):
    leaves = jax.tree_util.tree_leaves(params)
    squares = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves]
    total = jax.numpy.sum(jax.numpy.stack(squares))
    return float(jax.device_get(total)) ** 0.5

def _prefixed(d, default_prefix):
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

def train(config):
    general_config = config.train_sac_chunked_wm.general
    dreamer_config = config.train_sac_chunked_wm.dreamer
    chunk_config = config.train_sac_chunked_wm.chunk

    out_dir = pathlib.Path(general_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config.batch_size
    seq_len = config.batch_length
    chunk_len = chunk_config.chunk_len
    num_chunks = chunk_config.num_chunks
    gamma = chunk_config.gamma
    gamma_h = gamma ** chunk_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)
    print(f'PyTorch device: {device} | JAX devices: {jax.devices()}')
    print(f'chunk_len={chunk_len} num_chunks={num_chunks} '
          f'-> {chunk_len * num_chunks} imagined env steps per rollout')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)

    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)
    env.action_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(obs_dim, action_dim, OBS_KEY, ACTION_KEY)

    agent_config = build_agent_config(config, batch_size, seq_len, out_dir / 'wm_ckpts')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)
    if general_config.wm_ckpt:
        print(f'Loading world model checkpoint: {general_config.wm_ckpt}')
        raw = np.load(general_config.wm_ckpt, allow_pickle=True)
        wm_agent.load({k: unwrap(raw[k]) for k in raw.files})
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)
    rssm_cfg = agent_config.dyn.rssm
    feat_dim = int(rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes)
    print(f'World model feature dim: {feat_dim}')

    replay = OnlineReplay(obs_key=OBS_KEY, action_key=ACTION_KEY, max_episodes=dreamer_config.max_episodes)
    if train_dataset is not None:
        offline_episodes = OGBenchMethods.make_dreamer_episodes(
            train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
        replay.seed_from_offline(offline_episodes, rng=rng)
        print(f'Seeded replay buffer with {len(replay.offline_episodes)} offline episodes')

    policy = ChunkAgent(
        repr_dim=feat_dim, action_dim=action_dim, chunk_len=chunk_len, device=device,
        lr=chunk_config.lr, feature_dim=chunk_config.feature_dim,
        hidden_dim=chunk_config.hidden_dim, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, bc_alpha=chunk_config.bc_alpha,
        normalize_q=chunk_config.normalize_q,
    )
    flow_bc = FlowBC(
        repr_dim=feat_dim, chunk_dim=action_dim * chunk_len, device=device,
        lr=chunk_config.lr, feature_dim=chunk_config.feature_dim,
        hidden_dim=chunk_config.hidden_dim, flow_steps=chunk_config.flow_steps,
    )

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm='world_model',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    obs, info = env.reset(seed=config.seed)
    enc_carry, dyn_carry = bridge.init_encode(1)
    prevact = np.zeros((1, action_dim), dtype=np.float32)
    is_first = np.array([True])
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    n_updates = 0
    print('Starting world-model + chunked-agent training loop')

    while global_step < general_config.num_train_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        enc_carry, dyn_carry, feat_jax = bridge.encode_step(enc_carry, dyn_carry, state, prevact, is_first)

        if global_step < general_config.num_seed_steps:
            action = env.action_space.sample()
        else:
            # The actor is queried once per chunk; the remaining chunk_len - 1
            # steps replay the buffered actions open loop. encode_step above
            # still runs every step, so the posterior is current when the next
            # chunk decision is made.
            if chunk_pos >= chunk_len:
                feat_np = np.asarray(jax.device_get(feat_jax))[0].copy()
                chunk_buffer = policy.act(feat_np, eval_mode=False)
                chunk_pos = 0
            action = chunk_buffer[chunk_pos]
            chunk_pos += 1

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        replay.add_step(state[0], action, reward, np.asarray(next_obs, dtype=np.float32), terminated, truncated)

        done = bool(terminated or truncated)
        prevact = action.reshape(1, -1).astype(np.float32)
        is_first = np.array([False])
        obs = next_obs
        if done:
            obs, info = env.reset()
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])
            chunk_pos = chunk_len

        global_step += 1
        metrics = {}
        ready = replay.ready(seq_len)

        if ready and global_step % dreamer_config.train_every == 0:
            batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
            batch = OGBenchMethods.to_jax(batch_np)
            batch.pop('discount', None)
            batch['seed'] = wm_agent._seeds(global_step, wm_agent.train_mirrored)
            wm_carry = wm_agent.init_train(batch_size)
            wm_carry, outs, wm_mets = wm_agent.train(wm_carry, batch)
            metrics.update(_prefixed(wm_mets, 'wm'))

            if global_step % general_config.log_every == 0:
                metrics['diagnosis/wm_param_norm'] = _param_norm(wm_agent.params)
                metrics['diagnosis/replay_transitions'] = len(replay)
                metrics['diagnosis/replay_episodes'] = len(replay.dreamer_episodes)
                _succ = replay.success_stats
                metrics['replay/success_frac_total'] = _succ['total_frac']
                metrics['replay/success_frac_online'] = _succ['online_frac']
                metrics['replay/success_frac_offline'] = _succ['offline_frac']
                metrics['replay/success_episodes_online'] = _succ['online_success']
                metrics['replay/success_episodes_total'] = _succ['offline_success'] + _succ['online_success']

        if ready and global_step % chunk_config.train_every == 0:
            seed_batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
            seed_batch = OGBenchMethods.to_jax(seed_batch_np)
            pool = bridge.seed_pool(seed_batch, batch_size)

            # Flow behavior model: real (latent, chunk) pairs from the same
            # batch. Indices are filtered so the chunk stays inside the sampled
            # sequence and inside one episode.
            bc_idx, bc_chunks_np = chunk_pair_indices(seed_batch_np, chunk_len, action_key=ACTION_KEY)
            if len(bc_idx) > 0:
                take = min(chunk_config.bc_batch, len(bc_idx))
                sel = rng.choice(len(bc_idx), size=take, replace=False)
                bc_pool = {k: v[bc_idx[sel]] for k, v in pool.items()}
                bc_feat = jax_to_torch(bridge.get_feat(bridge.place_seed(bc_pool)), device)
                bc_chunks = torch.as_tensor(bc_chunks_np[sel], device=device).float()
                metrics.update(_prefixed(flow_bc.update(bc_feat, bc_chunks), 'sac'))

            # Imagination seeds are sampled without the chunk-window filter, so
            # the seed distribution is not biased away from late timesteps.
            seed_carry = bridge.place_seed(
                subsample_tree_np(pool, chunk_config.imagination_batch, rng))

            feats, chunks, chunk_rewards, chunk_conts, next_feats, weights, step_rewards = \
                imagine_chunk_rollout(
                    bridge, policy, seed_carry, num_chunks, chunk_len,
                    device, gamma, reward_shift=chunk_config.reward_shift)

            next_values = policy.chunk_target_values(next_feats)
            targets = chunk_lambda_targets(
                chunk_rewards, chunk_conts, next_values, gamma_h, chunk_config.lam, num_chunks)

            metrics.update(_prefixed(policy.update_critic(feats, chunks, targets, weights), 'sac'))
            metrics.update(_prefixed(policy.update_actor(feats.detach(), weights.detach(), flow_bc), 'sac'))
            policy.update_target()
            n_updates += 1

            metrics['sac/mean_chunk_reward'] = chunk_rewards.mean().item()
            metrics['sac/mean_chunk_cont'] = chunk_conts.mean().item()
            metrics['sac/chunk_diversity'] = policy.chunk_diversity(feats.detach())

            cr = chunk_rewards.reshape(num_chunks, -1)
            cc = chunk_conts.reshape(num_chunks, -1)
            cw = weights.reshape(num_chunks, -1)
            metrics['diagnosis/rollout_cont_first_chunk'] = cc[0].mean().item()
            metrics['diagnosis/rollout_cont_last_chunk'] = cc[-1].mean().item()
            metrics['diagnosis/rollout_reward_first_chunk'] = cr[0].mean().item()
            metrics['diagnosis/rollout_reward_last_chunk'] = cr[-1].mean().item()
            metrics['diagnosis/rollout_reward_max'] = chunk_rewards.max().item()
            metrics['diagnosis/rollout_weight_last_chunk'] = cw[-1].mean().item()
            # Reward at each position INSIDE a chunk. If the masking in
            # pool_chunk is wrong, reward keeps appearing at late positions
            # after cont has already collapsed -- compare these two numbers
            # against rollout_cont_last_chunk.
            metrics['diagnosis/intra_chunk_reward_first'] = step_rewards[0].mean().item()
            metrics['diagnosis/intra_chunk_reward_last'] = step_rewards[-1].mean().item()

        if metrics and global_step % general_config.log_every == 0:
            wandb.log(numeric_metrics(metrics), step=global_step)

        if global_step % general_config.eval_every == 0 and global_step > 0:
            results = eval_chunk_in_env(
                env, bridge, policy, action_dim, general_config.eval_episodes,
                device, OBS_KEY, chunk_len, eef_slice=eef_slice, record_video=True)
            print(f'step {global_step:7d} | return {results["mean_return"]:.2f} | '
                  f'success {results["success_rate"]:.2f} | coherence {results["coherence"]:.4f}')
            eval_csv.append(global_step, n_updates, time.time() - start_time, results)
            log_dict = {
                'eval/mean_return': results['mean_return'],
                'eval/success_rate': results['success_rate'],
                'eval/coherence': results['coherence'],
                'eval/mean_episode_len': results['mean_episode_len'],
            }
            if results['video'] is not None:
                log_dict['eval/video'] = wandb.Video(results['video'], fps=20, format='mp4')
            wandb.log(log_dict, step=global_step)
            # eval resets this same env, so the loop's obs and the RSSM carry
            # are both stale -- resync before resuming collection.
            obs, info = env.reset()
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])
            chunk_pos = chunk_len

        if global_step % general_config.save_every == 0 and global_step > 0:
            torch.save(policy.state_dict_all(), out_dir / 'chunk_latest.pt')
            torch.save(flow_bc.state_dict_all(), out_dir / 'flow_latest.pt')
            wm_cp = elements.Checkpoint(out_dir / 'wm_latest.pkl')
            wm_cp.agent = wm_agent
            wm_cp.save()

    torch.save(policy.state_dict_all(), out_dir / 'chunk_final.pt')
    torch.save(flow_bc.state_dict_all(), out_dir / 'flow_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac_chunked_wm.py --train_sac_chunked_wm.general.env_name=cube-double-play-singletask-v0
