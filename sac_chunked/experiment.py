""" The one training loop every arm runs.

    Offline phase: gradient updates on the static dataset, no interaction.
    Online phase: act in chunks, store transitions, update. Evaluate every
    eval_every steps. This is QC's main.py protocol, and every number in the
    QC paper's tables uses it.

    An arm plugs into this loop through the `Arm` interface below. The loop
    itself is arm-agnostic: it does not know whether a latent model exists.
    That is what keeps the comparison controlled -- the only code that
    differs between two arms is inside their Arm classes. """

import pathlib
import time

import elements
import numpy as np
import ruamel.yaml as yaml
import torch

from sac_chunked.sac_chunk_agent import ChunkAgent
from sac_chunked.replay import ChunkTransitionReplay
from sac_chunked.evaluation_chunk import eval_chunk_in_env, EvalCSV
from helpers.sac_wm_utils import set_seed_everywhere
from helpers.interop import numeric_metrics

OBS_KEY = 'state'
ENV_ACTION_LOW = -1.0
ENV_ACTION_HIGH = 1.0


# ------------------------------------------------------------------ config

def load_config(folder, argv=None):
    """ configs.yaml `defaults`, plus any presets named with --configs (none
        are defined today), plus dotted CLI overrides (--chunk.select_n=16). """
    configs_txt = elements.Path(pathlib.Path(folder) / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        if name != 'defaults':
            config = config.update(configs[name])
    return elements.Flags(config).parse(other)


def prefixed(d, default_prefix):
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}


# --------------------------------------------------------------------- env

def build_env(general, seed):
    """ (env, offline_dataset), OGBench. """
    from helpers.ogbench_methods import OGBenchMethods
    import ogbench
    if general.seed_from_offline:
        env, train_dataset, _ = OGBenchMethods.load_ogbench(general.env_name)
        return env, train_dataset
    return ogbench.make_env_and_datasets(general.env_name, env_only=True), None


# --------------------------------------------------------------------- arm

class Arm:
    """ What an arm must provide. The base class IS the control: QC-FQL with
        the critic picking the best of select_n candidate chunks, no model
        anywhere.

        Subclasses override exactly the hook their idea changes:
          build_selector   what picks the candidate chunk at act time
          critic_target    what the QC critic regresses onto
          model_update     how the latent model (if any) trains
          report           model diagnostics at eval, zero env steps """

    name = 'qcfql_bon'

    def __init__(self, config, obs_dim, action_dim, device, rng):
        self.config = config
        self.chunk = config.chunk
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.rng = rng
        self.gamma = self.chunk.gamma
        self.chunk_len = self.chunk.chunk_len
        self.gamma_h = self.gamma ** self.chunk_len
        self.model = None
        self.policy = ChunkAgent(
            repr_dim=obs_dim, action_dim=action_dim, chunk_len=self.chunk_len,
            device=device, lr=self.chunk.lr, hidden_dim=self.chunk.hidden_dim,
            num_layers=self.chunk.num_layers,
            critic_target_tau=self.chunk.critic_target_tau,
            ensemble=self.chunk.ensemble, alpha=self.chunk.alpha,
            flow_steps=self.chunk.flow_steps, q_agg=self.chunk.q_agg,
            compile_nets=self.chunk.compile_nets)
        self.build_model()
        self.selector = self.build_selector()

    def build_model(self):
        return None

    def build_selector(self):
        from wm.chunk_selector import ChunkSelector
        return ChunkSelector(None, self.policy, self.action_dim, self.chunk_len,
                             self.chunk.select_n, self.gamma, self.device)

    def describe(self):
        n = self.chunk.select_n
        return (f'{self.name}: critic best-of-{n}' if n > 1
                else f'{self.name}: plain QC-FQL (select_n=1)')

    def critic_target(self, next_obs, reward, mask, metrics_on=False):
        """ QC eq. 15: R_real + gamma^h * mask * Q_target(s', mu(s', z)).
            metrics_on lets an override skip its own bookkeeping syncs on
            steps whose metrics are discarded. """
        with torch.no_grad():
            return reward + self.gamma_h * mask * self.policy.chunk_target_values(next_obs)

    def model_update(self, replay, metrics_on):
        return {}

    def report(self, replay):
        return {}

    def log_extra(self):
        out = self.selector.pop_stats()
        if self.model is not None:
            out['diagnosis/wm_param_norm'] = self.model.param_norm()
        return out

    def save(self, out_dir, tag):
        torch.save(self.policy.state_dict_all(), out_dir / f'chunk_{tag}.pt')
        if self.model is not None:
            torch.save(self.model.state_dict_all(), out_dir / f'model_{tag}.pt')


# ------------------------------------------------------------ QC-FQL update

def agent_update(arm, replay, metrics_on=True):
    """ One QC-FQL critic + actor update on real replay chunks. Identical in
        every arm except for what critic_target returns. """
    chunk = arm.chunk
    batch = replay.sample_chunks(chunk.batch_size, arm.device, arm.rng, arm.gamma)
    if batch is None:
        return None
    b_obs, b_chunk, b_rew, b_mask, b_valid, b_step_valid, b_next = batch

    targets = arm.critic_target(b_next, b_rew, b_mask, metrics_on=metrics_on)

    metrics = {}
    metrics.update(prefixed(arm.policy.update_critic(
        b_obs, b_chunk, targets, b_valid, metrics_on=metrics_on), 'sac'))
    # Reference passes one batch to both terms; weight is ones because the
    # actor loss in acfql.actor_loss is unweighted.
    metrics.update(prefixed(arm.policy.update_actor(
        b_obs, torch.ones_like(b_valid), bc_feat=b_obs, bc_chunk=b_chunk,
        bc_valid=b_step_valid, metrics_on=metrics_on), 'sac'))
    arm.policy.update_target()

    if not metrics_on:
        return metrics
    metrics['sac/mean_chunk_reward'] = b_rew.mean().item()
    metrics['sac/mean_chunk_mask'] = b_mask.mean().item()
    metrics['sac/valid_frac'] = b_valid.mean().item()
    metrics['sac/chunk_diversity'] = arm.policy.chunk_diversity(b_obs)
    metrics['diagnosis/batch_reward_max'] = b_rew.max().item()
    return metrics


# --------------------------------------------------------------------- run

def run(config, arm_cls):
    general = config.general
    chunk = config.chunk
    out_dir = pathlib.Path(general.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_len = chunk.chunk_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(config.seed)
    set_seed_everywhere(config.seed)

    use_wandb = general.wandb_mode != 'disabled'
    if use_wandb:
        import wandb
        wandb.init(project=general.wandb_project, mode=general.wandb_mode,
                   name=(general.run_name or None), config=config.flat)
    log = (lambda d, step: wandb.log(d, step=step)) if use_wandb else (lambda d, step: None)

    env, train_dataset = build_env(general, config.seed)
    env.action_space.seed(config.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # main.py sizes the buffer as max(buffer_size, dataset.size + 1), so
    # offline data is never evicted once online transitions arrive.
    dataset_size = len(train_dataset['observations']) if train_dataset is not None else 0
    capacity = max(chunk.replay_capacity, dataset_size + general.num_online_steps + 1)
    replay = ChunkTransitionReplay(obs_dim, action_dim, chunk_len, capacity=capacity,
                                   online_frac=chunk.online_frac)
    if train_dataset is not None:
        n = replay.seed_from_offline(train_dataset)
        print(f'Seeded replay with {n} offline transitions '
              f'({replay.offline_episodes} episodes, success frac '
              f'{replay.success_stats["offline_frac"]:.2f}) | capacity {capacity}')

    arm = arm_cls(config, obs_dim, action_dim, device, rng)
    print(f'PyTorch device: {device} | chunk_len={chunk_len} alpha={chunk.alpha}')
    print(f'ARM  {arm.describe()}')
    if general.model_ckpt and arm.model is not None:
        print(f'Loading latent model checkpoint: {general.model_ckpt}')
        arm.model.load_state_dict_all(torch.load(general.model_ckpt, map_location=device))

    eval_csv = EvalCSV(out_dir / 'eval_log.csv', arm=arm.name,
                       env_name=general.env_name, seed=config.seed, chunk_len=chunk_len)
    eef_slice = tuple(chunk.eef_slice)
    start_time = time.time()
    history = []
    last_extra = {}

    def run_eval(step, n_updates):
        results = eval_chunk_in_env(
            env, arm.policy, general.eval_episodes, OBS_KEY, chunk_len,
            eef_slice=eef_slice, record_video=use_wandb, selector=arm.selector)
        print(f'step {step:7d} | return {results["mean_return"]:.2f} | '
              f'success {results["success_rate"]:.2f} | '
              f'coherence {results["coherence"]:.4f} | '
              f'{time.time() - start_time:.0f}s')
        eval_csv.append(step, n_updates, time.time() - start_time, results)
        history.append((step, results['mean_return'], results['success_rate']))
        log_dict = {
            'eval/mean_return': results['mean_return'],
            'eval/success_rate': results['success_rate'],
            'eval/coherence': results['coherence'],
            'eval/mean_episode_len': results['mean_episode_len'],
        }
        rep = arm.report(replay)
        if rep:
            log_dict.update(numeric_metrics(rep))
        # Selection attribution since the last log step, so a run without
        # wandb still shows whether the model is doing anything.
        attrib = {k: v for k, v in last_extra.items()
                  if k.startswith('select/')}
        if attrib:
            print('  attribution: ' + '  '.join(
                f'{k.split("/", 1)[1]} {v:.3f}' for k, v in sorted(attrib.items())))
        if results['video'] is not None:
            import wandb
            log_dict['eval/video'] = wandb.Video(results['video'], fps=20, format='mp4')
        log(log_dict, step)

    # ---------------------------------------------------------- offline
    n_updates = 0
    offline_steps = general.num_offline_steps
    for i in range(1, offline_steps + 1):
        on = (i % general.log_every == 0)
        metrics = {}
        if i % config.tdmpc.train_every == 0:
            metrics.update(prefixed(arm.model_update(replay, metrics_on=on), 'wm'))
        m = agent_update(arm, replay, metrics_on=on)
        if m is not None:
            metrics.update(m)
            n_updates += 1
        if on:
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 0
            last_extra = arm.log_extra()
            metrics.update(last_extra)
            log(numeric_metrics(metrics), i)
        if i % general.eval_every == 0:
            run_eval(i, n_updates)
            env.reset()
    if offline_steps > 0:
        arm.save(out_dir, 'offline')
        print(f'Offline phase done: {n_updates} policy updates')

    # ----------------------------------------------------------- online
    obs, info = env.reset(seed=config.seed)
    chunk_buffer = None
    chunk_pos = chunk_len
    global_step = 0
    ep_return = 0.0
    print(f'Starting online phase ({arm.describe()})')

    while global_step < general.num_online_steps:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)

        if global_step < general.num_seed_steps and offline_steps == 0:
            action = env.action_space.sample()
        else:
            # The chunk is executed fully before the next decision, matching
            # main.py's action_queue. The selector chooses WHICH chunk runs;
            # the open-loop commitment length is unchanged.
            if chunk_pos >= chunk_len:
                chunk_buffer = arm.selector.select(state)
                chunk_pos = 0
            action = chunk_buffer[chunk_pos]
            chunk_pos += 1

        env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        replay.add(state, action, reward, np.asarray(next_obs, dtype=np.float32),
                   terminated, truncated)

        obs = next_obs
        ep_return += float(reward)
        if terminated or truncated:
            # Real online episode finished: feed its return to the selector's
            # learning-progress gate (explore arm; absent elsewhere). Eval
            # episodes never come through here.
            if hasattr(arm.selector, 'report_episode_return'):
                arm.selector.report_episode_return(ep_return)
            ep_return = 0.0
            obs, info = env.reset()
            chunk_pos = chunk_len

        global_step += 1
        log_step = offline_steps + global_step
        on = (global_step % general.log_every == 0)
        metrics = {}

        training = global_step >= general.start_training
        # main.py: `if i >= FLAGS.start_training`. No gradient updates for
        # the first start_training online steps -- just interaction, so the
        # buffer accumulates fresh online transitions before training resumes
        # on them.
        if training and global_step % config.tdmpc.train_every == 0:
            metrics.update(prefixed(arm.model_update(replay, metrics_on=on), 'wm'))
        for _ in range(chunk.utd_ratio if training else 0):
            m = agent_update(arm, replay, metrics_on=on)
            if m is not None:
                metrics.update(m)
                n_updates += 1

        if on:
            metrics['diagnosis/replay_transitions'] = len(replay)
            metrics['diagnosis/gradient_updates'] = n_updates
            metrics['diagnosis/phase'] = 1
            last_extra = arm.log_extra()
            metrics.update(last_extra)
            s = replay.success_stats
            metrics['replay/success_frac_total'] = s['total_frac']
            metrics['replay/success_frac_online'] = s['online_frac']
            metrics['replay/success_episodes_online'] = s['online_success']
            metrics['replay/online_batch_frac'] = replay.last_online_frac
            log(numeric_metrics(metrics), log_step)

        if global_step % general.eval_every == 0:
            run_eval(log_step, n_updates)
            obs, info = env.reset()
            chunk_pos = chunk_len
            ep_return = 0.0   # the interrupted episode is not a finished one

        if global_step % general.save_every == 0:
            arm.save(out_dir, 'latest')

    arm.save(out_dir, 'final')
    env.close()
    if use_wandb:
        import wandb
        wandb.finish()
    print('Finish training')
    return history


def main(arm_cls, argv=None):
    folder = pathlib.Path(__file__).resolve().parent.parent
    return run(load_config(folder, argv), arm_cls)
