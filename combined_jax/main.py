import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets


# COMBINED PATCH: robomimic is not vendored (the combined arm runs OGBench
# only), so the name check from envs/robomimic_utils.py is inlined here.
# Actually requesting a robomimic env still fails in envs/env_utils.py.
def is_robomimic_env(env_name):
    return any(env_name.startswith(s)
               for s in ('lift', 'can', 'square', 'transport', 'tool_hang'))

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

# WM PATCH: the wm_explore arm -- the combined arm plus a bandit-gated
# dynamics-disagreement bonus on act-time selection during online
# collection only. Training, targets and eval are the combined arm's.
flags.DEFINE_bool('use_wm', False, 'Train a TD-MPC2 latent model alongside and add the novelty bonus at act time (wm_explore arm). Requires agent.actor_num_candidates > 1.')
config_flags.DEFINE_config_file('wm', 'wm/tdmpc2.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # WM PATCH: latent model + novelty selector (wm_explore arm)
    wm_model, selector, wm_info, wm_rng = None, None, {}, None
    if FLAGS.use_wm:
        assert config['actor_num_candidates'] > 1, \
            'wm_explore selects among candidates: set --agent.actor_num_candidates'
        from wm.tdmpc2 import WorldModel
        from wm.selector import NoveltySelector
        wm_model = WorldModel.create(
            FLAGS.seed, example_batch['observations'].shape[-1],
            example_batch['actions'].shape[-1], FLAGS.horizon_length,
            discount, FLAGS.wm)
        selector = NoveltySelector(FLAGS.wm)
        wm_rng = jax.random.PRNGKey(FLAGS.seed + 1)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        # WM PATCH: model update every train_every steps, on real windows
        # from the same data (offline: the online region is empty, so the
        # balanced sampler falls back to uniform, as in the PyTorch loop)
        if FLAGS.use_wm and i % FLAGS.wm.train_every == 0:
            wbatch = train_dataset.sample_windows(
                FLAGS.wm.batch_size, FLAGS.wm.horizon,
                online_frac=FLAGS.wm.online_frac, offline_size=train_dataset.size)
            wm_model, wm_info = wm_model.update(wbatch)

        if i % FLAGS.log_interval == 0:
            if wm_info:
                offline_info = dict(offline_info)
                offline_info.update({f'wm/{k}': v for k, v in wm_info.items()})
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()

    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # WM PATCH: controller state for the first online episode
    wm_ep_return = 0.0
    if FLAGS.use_wm:
        selector.begin_episode()

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            if FLAGS.use_wm:
                # WM PATCH: the selector picks WHICH candidate chunk runs;
                # the open-loop commitment length is unchanged
                wm_rng, skey = jax.random.split(wm_rng)
                action = selector.select(agent, wm_model, np.asarray(ob), skey)
            else:
                action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # WM PATCH: per-episode return for the controller, final step included
        if FLAGS.use_wm:
            wm_ep_return += float(int_reward)

        # done
        if done:
            # WM PATCH: real online episode finished -> feed its return to
            # the controller, then pick the next episode's arm
            if FLAGS.use_wm:
                selector.report_episode_return(wm_ep_return)
                wm_ep_return = 0.0
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
            if FLAGS.use_wm:
                selector.begin_episode()
        else:
            ob = next_ob

        # WM PATCH: model update on balanced real windows (online_frac of
        # the starts from the online region), same start_training gate as
        # the agent's updates
        wm_log = None
        if FLAGS.use_wm and i >= FLAGS.start_training and i % FLAGS.wm.train_every == 0:
            wbatch = replay_buffer.sample_windows(
                FLAGS.wm.batch_size, FLAGS.wm.horizon,
                online_frac=FLAGS.wm.online_frac, offline_size=train_dataset.size)
            wm_model, wm_info = wm_model.update(wbatch)
            wm_log = {f'wm/{k}': v for k, v in wm_info.items()}
            wm_log.update(selector.last)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)

        # WM PATCH: fold the model/selector metrics into the online log
        if wm_log is not None:
            update_info['online_agent'] = {
                **update_info.get('online_agent', {}), **wm_log}

        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
