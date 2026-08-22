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
from helpers.sac_wm_utils import set_seed_everywhere
from sac_chunked.chunk_utils import chunk_pair_indices, real_chunk_transitions
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

def _wm_update(wm_agent, replay, batch_size, seq_len, rng, global_step):
    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
    batch = OGBenchMethods.to_jax(batch_np)
    batch.pop('discount', None)
    batch['seed'] = wm_agent._seeds(global_step, wm_agent.train_mirrored)
    wm_carry = wm_agent.init_train(batch_size)
    wm_carry, outs, wm_mets = wm_agent.train(wm_carry, batch)
    return wm_mets

def _real_batch(bridge, pool, seed_batch_np, chunk_config, chunk_len, gamma, device, rng):
    """ Real chunk transitions from the SAME sampled sequence that seeded
        imagination, encoded through the SAME posterior pool -- no extra
        world-model calls beyond the one seed_pool already computed.

        This is exactly what plain QC-FQL's critic/actor already train on
        every step; nothing here is new relative to the reference, only
        pulled out as its own function so it can be concatenated with an
        imagined batch below. """
    flat_idx, next_flat_idx, chunks_np, chunk_reward_np, chunk_mask_np, valid_np = \
        real_chunk_transitions(seed_batch_np, chunk_len, gamma, action_key=ACTION_KEY)
    if len(flat_idx) == 0:
        return None
    take = min(chunk_config.real_batch, len(flat_idx))
    sel = rng.choice(len(flat_idx), size=take, replace=False)

    both_idx = np.concatenate([flat_idx[sel], next_flat_idx[sel]])
    both_pool = {k: v[both_idx] for k, v in pool.items()}
    both_feat = jax_to_torch(bridge.get_feat(bridge.place_seed(both_pool)), device)
    feat, next_feat = both_feat[:take], both_feat[take:]

    to = lambda x: torch.as_tensor(x, device=device).float()
    return (feat, next_feat, to(chunks_np[sel]), to(chunk_reward_np[sel]),
            to(chunk_mask_np[sel]), to(valid_np[sel]))

def _agent_update(bridge, policy, replay, config_batch, seq_len, chunk_config,
                  chunk_len, num_chunks, device, rng, gamma, gamma_h):
    """ One QC-FQL update where the critic/actor Q-loss batch is REAL chunk
        transitions AUGMENTED with imagined ones -- not replaced (MBPO-style
        model-data augmentation, not Dreamer-style full substitution).

        The real component is sized and sourced identically to what plain
        QC-FQL already trains on every step (chunk_config.real_batch chunks
        from replay, same bc_flow_loss). The imagined component is added on
        top as EXTRA rows in the same critic and actor calls. If the model's
        predictions are uninformative this degrades toward plain QC-FQL,
        rather than the full-replacement version's risk of training a
        near-empty-reward-signal critic when the current policy rarely
        reaches success states in imagination. """
    seed_batch_np = replay.sample_batch(config_batch, seq_len, rng=rng)
    seed_batch = OGBenchMethods.to_jax(seed_batch_np)
    pool = bridge.seed_pool(seed_batch, config_batch)

    real = _real_batch(bridge, pool, seed_batch_np, chunk_config, chunk_len, gamma, device, rng)
    if real is None:
        return None
    real_feat, real_next_feat, real_chunk, real_reward, real_mask, real_valid = real

    # Separate sample for bc_flow_loss: PER-POSITION valid masking, which the
    # real-critic sample above does not carry (it only has a whole-window
    # valid flag). Windows may overlap with the critic sample above; that is
    # fine, they serve different loss terms.
    bc_idx, bc_chunks_np, bc_valid_np = chunk_pair_indices(seed_batch_np, chunk_len, action_key=ACTION_KEY)
    if len(bc_idx) == 0:
        return None
    take_bc = min(chunk_config.bc_batch, len(bc_idx))
    sel_bc = rng.choice(len(bc_idx), size=take_bc, replace=False)
    bc_pool = {k: v[bc_idx[sel_bc]] for k, v in pool.items()}
    bc_feat = jax_to_torch(bridge.get_feat(bridge.place_seed(bc_pool)), device)
    bc_chunks = torch.as_tensor(bc_chunks_np[sel_bc], device=device).float()
    bc_valid = torch.as_tensor(bc_valid_np[sel_bc], device=device).float()

    seed_carry = bridge.place_seed(
        subsample_tree_np(pool, chunk_config.imagination_batch, rng))
    (img_feat, img_chunk, img_reward, img_cont, img_next_feat, img_weight,
     step_rewards) = imagine_chunk_rollout(
        bridge, policy, seed_carry, num_chunks, chunk_len,
        device, gamma, reward_shift=chunk_config.reward_shift)

    # QC's own TD target form, applied to each source separately (their next
    # latents come from different places -- real replay vs. imagination --
    # so they must be bootstrapped separately before concatenating).
    with torch.no_grad():
        real_next_values = policy.chunk_target_values(real_next_feat)
        real_targets = real_reward + gamma_h * real_mask * real_next_values
        img_next_values = policy.chunk_target_values(img_next_feat)
        img_targets = img_reward + gamma_h * img_cont * img_next_values

    feat = torch.cat([real_feat, img_feat], dim=0)
    chunk = torch.cat([real_chunk, img_chunk], dim=0)
    target = torch.cat([real_targets, img_targets], dim=0)
    weight = torch.cat([real_valid, img_weight], dim=0)

    metrics = {}
    metrics.update(_prefixed(policy.update_critic(feat, chunk, target, weight), 'sac'))
    metrics.update(_prefixed(policy.update_actor(
        feat.detach(), weight.detach(), bc_feat=bc_feat, bc_chunk=bc_chunks, bc_valid=bc_valid), 'sac'))
    policy.update_target()

    metrics['sac/mean_chunk_reward'] = target.mean().item()
    metrics['sac/chunk_diversity'] = policy.chunk_diversity(feat.detach())
    metrics['diagnosis/real_transitions'] = float(real_feat.shape[0])
    metrics['diagnosis/imagined_transitions'] = float(img_feat.shape[0])
    metrics['diagnosis/real_reward_nonzero_frac'] = (real_reward.abs() > 1e-6).float().mean().item()
    metrics['diagnosis/real_reward_mean'] = real_reward.mean().item()
    metrics['diagnosis/imagined_reward_nonzero_frac'] = (img_reward.abs() > 1e-6).float().mean().item()
    metrics['diagnosis/imagined_reward_mean'] = img_reward.mean().item()
    metrics['diagnosis/real_valid_frac'] = real_valid.mean().item()
    metrics['diagnosis/imagined_weight_last_chunk'] = img_weight.reshape(num_chunks, -1)[-1].mean().item()
    return metrics

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

    real_n = chunk_config.real_batch
    img_n = chunk_config.imagination_batch * num_chunks
    print(f'PyTorch device: {device} | JAX devices: {jax.devices()}')
    print(f'chunk_len={chunk_len} num_chunks={num_chunks} alpha={chunk_config.alpha}')
    print(f'critic batch per update: {real_n} real + {img_n} imagined = {real_n + img_n} '
          f'(plain QC-FQL: {real_n} real only)')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)
    wandb.log({'diagnosis/real_batch': real_n, 'diagnosis/imagined_batch': img_n}, step=0)

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
        lr=chunk_config.lr, hidden_dim=chunk_config.hidden_dim,
        num_layers=chunk_config.num_layers, critic_target_tau=chunk_config.critic_target_tau,
        ensemble=chunk_config.ensemble, alpha=chunk_config.alpha,
        flow_steps=chunk_config.flow_steps, q_agg=chunk_config.q_agg,
    )

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm='world_model',
                       env_name=general_config.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk_config.eef_slice)
    start_time = time.time()

    def run_eval(step, n_updates):
        results = eval_chunk_in_env(
            env, bridge, policy, action_dim, general_config.eval_episodes,
            device, OBS_KEY, chunk_len, eef_slice=eef_slice, record_video=True)
        print(f'step {step:7d} | return {results["mean_return"]:.2f} | '
              f'success {results["success_rate"]:.2f} | coherence {results["coherence"]:.4f}')
        eval_csv.append(step, n_updates, time.time() - start_time, results)
        log_dict = {
            'eval/mean_return': results['mean_return'],
            'eval/success_rate': results['success_rate'],
            'eval/coherence': results['coherence'],
            'eval/mean_episode_len': results['mean_episode_len'],
        }
        if results['video'] is not None:
            log_dict['eval/video'] = wandb.Video(results['video'], fps=20, format='mp4')
        wandb.log(log_dict, step=step)

    n_updates = 0
    offline_steps = general_config.num_offline_steps
    for i in range(1, offline_steps + 1):
        metrics = {}
        if not replay.ready(seq_len):
            continue
        if i % dreamer_config.train_every == 0:
            metrics.update(_prefixed(_wm_update(wm_agent, replay, batch_size, seq_len, rng, i), 'wm'))
        if i % chunk_config.train_every == 0:
            m = _agent_update(bridge, policy, replay, batch_size, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng, gamma, gamma_h)
            if m is not None:
                metrics.update(m)
                n_updates += 1
        if metrics and i % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = _param_norm(wm_agent.params)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 0
            wandb.log(numeric_metrics(metrics), step=i)
        if i % general_config.eval_every == 0:
            run_eval(i, n_updates)
            env.reset()
    if offline_steps > 0:
        torch.save(policy.state_dict_all(), out_dir / 'chunk_offline.pt')
        print(f'Offline phase done: {n_updates} policy updates')

    obs, info = env.reset(seed=config.seed)
    enc_carry, dyn_carry = bridge.init_encode(1)
    prevact = np.zeros((1, action_dim), dtype=np.float32)
    is_first = np.array([True])
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    print('Starting online phase (QC-FQL, real + imagined augmented critic)')

    while global_step < general_config.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        enc_carry, dyn_carry, feat_jax = bridge.encode_step(enc_carry, dyn_carry, state, prevact, is_first)

        if global_step < general_config.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
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
        log_step = offline_steps + global_step
        metrics = {}
        ready = replay.ready(seq_len)

        if ready and global_step % dreamer_config.train_every == 0:
            metrics.update(_prefixed(
                _wm_update(wm_agent, replay, batch_size, seq_len, rng, log_step), 'wm'))

        if (ready and global_step % chunk_config.train_every == 0
                and global_step >= general_config.start_training):
            m = _agent_update(bridge, policy, replay, batch_size, seq_len,
                              chunk_config, chunk_len, num_chunks, device, rng, gamma, gamma_h)
            if m is not None:
                metrics.update(m)
                n_updates += 1

        if metrics and global_step % general_config.log_every == 0:
            metrics['diagnosis/wm_param_norm'] = _param_norm(wm_agent.params)
            metrics['diagnosis/replay_transitions'] = len(replay)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 1
            _succ = replay.success_stats
            metrics['replay/success_frac_total'] = _succ['total_frac']
            metrics['replay/success_frac_online'] = _succ['online_frac']
            wandb.log(numeric_metrics(metrics), step=log_step)

        if global_step % general_config.eval_every == 0:
            run_eval(log_step, n_updates)
            obs, info = env.reset()
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])
            chunk_pos = chunk_len

        if global_step % general_config.save_every == 0:
            torch.save(policy.state_dict_all(), out_dir / 'chunk_latest.pt')
            wm_cp = elements.Checkpoint(out_dir / 'wm_latest.pkl')
            wm_cp.agent = wm_agent
            wm_cp.save()

    torch.save(policy.state_dict_all(), out_dir / 'chunk_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_sac_chunked_wm.py --train_sac_chunked_wm.general.env_name=cube-double-play-singletask-v0