import jax
import numpy as np
from interop import extract_state

def eval_in_env(env, bridge, policy, action_dim, num_episodes, device, obs_key, record_video=False):
    returns, successes = [], []
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
        enc_carry, dyn_carry = bridge.init_encode(1)
        prevact = np.zeros((1, action_dim), dtype=np.float32)
        is_first = np.array([True])
        done = False
        ep_return = 0.0
        ep_success = False

        if ep == 0:
            safe_render()

        while not done:
            state = extract_state(obs, obs_key)
            enc_carry, dyn_carry, feat_j = bridge.encode_step(
                enc_carry, dyn_carry, state, prevact, is_first)
            feat_np = np.asarray(jax.device_get(feat_j))[0].copy()
            action = policy.act(feat_np, step=0, eval_mode=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_success = ep_success or bool(info.get('success', False))

            if ep == 0:
                safe_render()

            prevact = action.reshape(1, -1).astype(np.float32)
            is_first = np.array([False])
            obs = next_obs

        returns.append(ep_return)
        successes.append(float(ep_success))

    video = None
    if record_video and frames:
        video = np.stack(frames).astype(np.uint8).transpose(0, 3, 1, 2)

    return float(np.mean(returns)), float(np.mean(successes)), video
