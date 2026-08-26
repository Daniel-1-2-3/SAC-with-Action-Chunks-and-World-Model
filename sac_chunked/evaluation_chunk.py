import csv
import pathlib
import jax
import numpy as np
from sac_chunked.chunk_utils import temporal_coherence
from helpers.interop import extract_state

EVAL_CSV_FIELDS = [
    'arm', 'env', 'seed', 'chunk_len', 'env_step', 'gradient_updates',
    'mean_return', 'success_rate', 'coherence', 'mean_episode_len', 'wall_time_s',
]

def eval_chunk_in_env(env, bridge, policy, action_dim, num_episodes, device,
                      obs_key, chunk_len, eef_slice=(0, 3), record_video=False,
                      selector=None):
    """ Chunked evaluation.

        The actor is queried once every chunk_len steps and the chunk is
        executed open loop, matching how the policy is used during training.

        bridge=None, selector=None: raw observations straight into the actor
        (the MVE and Dyna trainers, and the no-world-model arm).

        selector: a ChunkSelector (selection trainer). Evaluation then
        measures the DEPLOYED behavior -- model-scored chunk selection -- not
        the bare policy. The selector's own observe/record_action calls keep
        its posterior latent filtered on every step; with select_n<=1 they are
        no-ops and this is identical to the bare-policy path.

        bridge non-None (legacy latent-policy path): the observation is
        encoded on EVERY step so the RSSM posterior stays correct for the
        next chunk decision -- committing to actions is not a reason to stop
        looking. """
    returns, successes, lengths, coherences = [], [], [], []
    frames = []

    def safe_render():
        nonlocal record_video
        if not record_video:
            return
        try:
            frames.append(env.render())
        except Exception as e:
            print(f'Video recording failed, disabling for this eval: {e}')
            record_video = False

    for ep in range(num_episodes):
        obs, info = env.reset()
        if selector is not None:
            selector.reset()
        if bridge is not None:
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])

        done = False
        ep_return = 0.0
        ep_success = False
        ep_len = 0
        eef_track = []
        chunk = None
        chunk_pos = chunk_len

        if ep == 0:
            safe_render()

        while not done:
            state = extract_state(obs, obs_key)
            eef_track.append(state[0][eef_slice[0]:eef_slice[1]])

            if bridge is not None:
                enc_carry, dyn_carry, feat_j = bridge.encode_step(
                    enc_carry, dyn_carry, state, prevact, is_first)
                feat_np = np.asarray(jax.device_get(feat_j))[0].copy()
            else:
                feat_np = state[0]

            if selector is not None:
                selector.observe(state[0])

            if chunk_pos >= chunk_len:
                if selector is not None:
                    chunk = selector.select(state[0])
                else:
                    chunk = policy.act(feat_np, eval_mode=True)
                chunk_pos = 0
            action = chunk[chunk_pos]
            chunk_pos += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_len += 1
            ep_success = ep_success or bool(info.get('success', False))

            if ep == 0:
                safe_render()

            if selector is not None:
                selector.record_action(action)
            if bridge is not None:
                prevact = action.reshape(1, -1).astype(np.float32)
                is_first = np.array([False])
            obs = next_obs

        returns.append(ep_return)
        successes.append(float(ep_success))
        lengths.append(ep_len)
        coherences.append(temporal_coherence(np.stack(eef_track), stride=5))

    video = None
    if record_video and frames:
        video = np.stack(frames).astype(np.uint8).transpose(0, 3, 1, 2)

    return {
        'mean_return': float(np.mean(returns)),
        'success_rate': float(np.mean(successes)),
        'coherence': float(np.mean(coherences)),
        'mean_episode_len': float(np.mean(lengths)),
        'video': video,
    }

class EvalCSV:
    """ Appends one row per evaluation to a CSV, so runs can be compared
        offline without going through wandb. compare_to_paper.py reads these. """

    def __init__(self, path, arm, env_name, seed, chunk_len):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fixed = {'arm': arm, 'env': env_name, 'seed': seed, 'chunk_len': chunk_len}
        if not self.path.exists():
            with open(self.path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS).writeheader()

    def append(self, env_step, gradient_updates, wall_time_s, results):
        row = dict(self.fixed)
        row.update({
            'env_step': env_step,
            'gradient_updates': gradient_updates,
            'wall_time_s': round(wall_time_s, 1),
            'mean_return': round(results['mean_return'], 4),
            'success_rate': round(results['success_rate'], 4),
            'coherence': round(results['coherence'], 6),
            'mean_episode_len': round(results['mean_episode_len'], 1),
        })
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS).writerow(row)