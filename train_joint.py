import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false') # Don't let JAX hog the GPU before torch inits
os.environ.setdefault('MUJOCO_GL', 'egl') # Headless for video rendering

import pathlib
import elements
import jax
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import ogbench

from dreamer.wm_agent import WorldModelAgent # JAX dreamer agent
from dreamer.wm_bridge import WorldModelBridge # handles JAX and numpy conversions
from sac_wm_agent import SACWorldModelAgent # SAC + world model (file kept as drqv2_wm_agent.py for continuity)
from evaluation import eval_in_env
from imagination import imagine_rollout
from interop import numeric_metrics, subsample_tree_np, unwrap # JAX to torch/dict helpers
from ogbench_methods import OGBenchMethods # for formatting OGBench batches to be used by world model
from online_replay import OnlineReplay

OBS_KEY = 'state'
ACTION_KEY = 'action'
# Joint actions take [-1, 1]
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

def build_agent_config(config, batch_size, seq_len, logdir): # Config for wm agent
    """ replay_context used when training wm agent, running rssm on a 
        few timesteps to have a warmed up deterministic state, then start 
        computing loss. Turn this off since replay buffer don't suppor it currently """
    return elements.Config(
        **config.agent,
        logdir=str(logdir),
        seed=config.seed,
        jax=config.jax,
        batch_size=batch_size, # sequences per training batch
        batch_length=seq_len, # timesteps per sequence
        replay_context=0,
        report_length=seq_len,
        replica=0,
        replicas=1,
    )

def build_real_env(env_name, load_offline_dataset):
    if load_offline_dataset:
        return OGBenchMethods.load_ogbench(env_name) # Download OGBench, later prefill some of replay buffer
    env, _, _ = ogbench.make_env_and_datasets(env_name, env_only=True)
    return env, None, None

def _param_norm(params):
    """ L2 norm across every leaf of a JAX param pytree -- used to
        watch the world model's weight scale over the whole run.
        Sums on-device, then does ONE explicit device_get at the end --
        float(jax_array) is an implicit transfer, which GPU transfer
        guards reject (CPU transfers are always allowed regardless of
        guard level, which is why this only broke once running on
        cuda:0, not during local CPU testing). jax.device_get() is an
        explicit transfer, which guards permit even in "disallow" mode.
        Also avoids Python's sum() builtin here: sum(iterable) starts
        its accumulator at the plain int 0, so the first addition mixes
        a host-side Python int with a JAX array -- an implicit
        host-to-device transfer, same guard, opposite direction.
        jax.numpy.stack keeps every value on-device until the single
        explicit device_get at the end. """
    leaves = jax.tree_util.tree_leaves(params)
    squares = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves]
    total = jax.numpy.sum(jax.numpy.stack(squares))
    return float(jax.device_get(total)) ** 0.5

def _prefixed(d, default_prefix):
    """ Prefix every key with default_prefix, EXCEPT keys that already
        carry their own namespace (e.g. 'diagnosis/critic_grad_norm')
        -- those pass through unprefixed so they land in their own
        wandb tab instead of sac/diagnosis/... """
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

def train(config):
    general_config, dreamer_config, sac_config = config.joint.general, config.joint.dreamer, config.joint.sac

    out_dir = pathlib.Path(general_config.out_dir) # Checkpoints and outs dir
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config.batch_size # Num sequences in a batch for world model update
    seq_len = config.batch_length # length of a sequence

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(config.seed)
    print(f'PyTorch device: {device} | JAX devices: {jax.devices()}')
    wandb.init(project=general_config.wandb_project, mode=general_config.wandb_mode, config=config.flat)
    env, train_dataset, _ = build_real_env(general_config.env_name, general_config.seed_from_offline)

    print(f'env.observation_space = {env.observation_space}')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(
        obs_dim, action_dim, OBS_KEY, ACTION_KEY)

    # Construct JAX wm agent
    agent_config = build_agent_config(config, batch_size, seq_len, out_dir / 'wm_ckpts')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)

    if general_config.wm_ckpt:
        print(f'Loading world model checkpoint: {general_config.wm_ckpt}')
        raw = np.load(general_config.wm_ckpt, allow_pickle=True)
        state = {k: unwrap(raw[k]) for k in raw.files}
        wm_agent.load(state)
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)
    rssm_cfg = agent_config.dyn.rssm
    feat_dim = int(rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes)
    print(f'World model feature dim: {feat_dim}')

    # Create replay buffer
    replay = OnlineReplay(obs_key=OBS_KEY, action_key=ACTION_KEY, max_episodes=dreamer_config.max_episodes)
    if train_dataset is not None:
        offline_episodes = OGBenchMethods.make_dreamer_episodes(
            train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
        replay.seed_from_offline(offline_episodes, rng=rng) # Put offline episodes into buffer for warm start
        print(f'Seeded replay buffer with {len(replay.dreamer_episodes)} offline episodes')

    # Create SAC policy
    policy = SACWorldModelAgent( # repr_dim same as world-model feature dim
        repr_dim=feat_dim, action_shape=(action_dim,), device=device,
        lr=sac_config.lr, feature_dim=sac_config.feature_dim, hidden_dim=sac_config.hidden_dim,
        critic_target_tau=sac_config.critic_target_tau,
        init_ent_coef=sac_config.init_ent_coef,
    )

    # Reset
    obs, info = env.reset()
    enc_carry, dyn_carry = bridge.init_encode(1)
    prevact = np.zeros((1, action_dim), dtype=np.float32) # No actions taken yet
    is_first = np.array([True])
    global_step = 0
    print('Starting world-model + SAC training loop')

    while global_step < general_config.num_train_steps:
        # Encode the obs 
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        enc_carry, dyn_carry, feat_jax = bridge.encode_step(enc_carry, dyn_carry, state, prevact, is_first)

        if global_step < general_config.num_seed_steps:
            action = env.action_space.sample()
        else:
            # Using encoded obs as input, calculate action to take with SAC policy
            feat_np = np.asarray(jax.device_get(feat_jax))[0].copy()
            action = policy.act(feat_np, eval_mode=False)

        # Take step and store the observations in replay
        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        # Print whenever the real env gives a non-baseline reward (-1 is the
        # sparse "no progress" default for this task, so anything else means
        # the agent actually made progress/succeeded).
        if reward != -1.0:
            print(f'step {global_step:7d} | got reward {reward:.4f} | terminated={terminated}')
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

        global_step += 1
        metrics = {}
        ready = replay.ready(seq_len)

        # Update world model
        if ready and global_step % dreamer_config.train_every == 0:
            batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
            batch = OGBenchMethods.to_jax(batch_np)
            batch.pop('discount', None) # Not part of WorldModelAgent's spaces
            batch['seed'] = wm_agent._seeds(global_step, wm_agent.train_mirrored)
            wm_carry = wm_agent.init_train(batch_size)

            wm_carry, outs, wm_mets = wm_agent.train(wm_carry, batch)
            metrics.update({f'wm/{k}': v for k, v in wm_mets.items()})

            # diagnosis/: gated behind log_every since the WM trains every
            # step but a param-norm tree traversal isn't free -- no point
            # paying that cost on steps that never reach wandb anyway.
            if global_step % general_config.log_every == 0:
                metrics['diagnosis/wm_param_norm'] = _param_norm(wm_agent.params)
                metrics['diagnosis/replay_transitions'] = len(replay)
                metrics['diagnosis/replay_episodes'] = len(replay.dreamer_episodes)

        # Update SAC policy
        if ready and global_step % sac_config.train_every == 0:
            # Get a pool of seed states
            seed_batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
            seed_batch = OGBenchMethods.to_jax(seed_batch_np)
            seed_pool = bridge.seed_pool(seed_batch, batch_size)
            seed_pool = subsample_tree_np(seed_pool, sac_config.imagination_batch, rng)
            seed_carry = bridge.place_seed(seed_pool)

            # Using seed states imagine horizon steps forward
            feats, actions, rewards, conts, next_feats, weights = imagine_rollout(
                bridge, policy, seed_carry, sac_config.horizon, 
                device, sac_config.gamma, global_step
            )
            discounts = sac_config.gamma * conts

            # Update policy using the imagined states
            metrics.update(_prefixed(policy.update_critic(feats, actions, rewards, discounts, next_feats, weights), 'sac'))
            metrics.update(_prefixed(policy.update_actor(feats.detach(), weights.detach()), 'sac'))
            policy.update_target()
            metrics['sac/mean_imag_reward'] = rewards.mean().item()
            metrics['sac/mean_imag_cont'] = conts.mean().item()

            # diagnosis/: rewards/conts/weights come back flattened as
            # (horizon * batch, 1), step-major -- reshape to (horizon, batch)
            # to see WHERE in the rollout things go wrong, not just the
            # average across the whole thing. This is what would have
            # shown the cont-probability drift as a first-step-vs-last-step
            # split instead of one blended number.
            H = sac_config.horizon
            rewards_by_step = rewards.reshape(H, -1)
            conts_by_step = conts.reshape(H, -1)
            weights_by_step = weights.reshape(H, -1)
            metrics['diagnosis/rollout_cont_first_step'] = conts_by_step[0].mean().item()
            metrics['diagnosis/rollout_cont_last_step'] = conts_by_step[-1].mean().item()
            metrics['diagnosis/rollout_cont_std'] = conts.std().item()
            metrics['diagnosis/rollout_reward_first_step'] = rewards_by_step[0].mean().item()
            metrics['diagnosis/rollout_reward_last_step'] = rewards_by_step[-1].mean().item()
            metrics['diagnosis/rollout_reward_std'] = rewards.std().item()
            metrics['diagnosis/rollout_reward_min'] = rewards.min().item()
            metrics['diagnosis/rollout_reward_max'] = rewards.max().item()
            # how much of the horizon's weight survives to the last step --
            # collapses toward 0 if cont keeps predicting early termination
            metrics['diagnosis/rollout_weight_last_step'] = weights_by_step[-1].mean().item()
            # imagined-action saturation -- pinned actions were the DrQV2-era
            # bug this whole SAC switch is meant to fix; worth watching to
            # confirm it actually stays gone
            metrics['diagnosis/imag_action_sat_frac'] = (actions.abs() > 0.95).float().mean().item()
            metrics['diagnosis/imag_action_mean_abs'] = actions.abs().mean().item()

        # Log metrics
        if metrics and global_step % general_config.log_every == 0:
            wandb.log(numeric_metrics(metrics), step=global_step)

        # Run eval steps in real environment
        if global_step % general_config.eval_every == 0 and global_step > 0:
            mean_return, success_rate, video = eval_in_env(
                env, bridge, policy, action_dim, 
                general_config.eval_episodes, device,
                OBS_KEY, record_video=True
            )
            print(f'step {global_step:7d} | eval_return {mean_return:.4f} | eval_success_rate {success_rate:.4f}')
            log_dict = {'eval/mean_return': mean_return, 'eval/success_rate': success_rate}
            if video is not None:
                log_dict['eval/video'] = wandb.Video(video, fps=20, format='mp4')
            wandb.log(log_dict, step=global_step)

        if global_step % general_config.save_every == 0 and global_step > 0:
            torch.save(policy.state_dict_all(), out_dir / 'sac_latest.pt')
            wm_cp = elements.Checkpoint(out_dir / 'wm_latest.pkl')
            wm_cp.agent = wm_agent
            wm_cp.save()
            print(f'Saved checkpoints at step {global_step}')

    torch.save(policy.state_dict_all(), out_dir / 'sac_final.pt')
    env.close()
    wandb.finish()
    print('Finish training')

if __name__ == '__main__':
    _folder = pathlib.Path(__file__).parent
    _config = load_config(_folder)
    train(_config)

# python train_joint.py --joint.general.env_name=cube-single-play-singletask-v0