import torch
from drqv2_wm_utils import schedule
from interop import jax_to_torch

def imagine_rollout(bridge, drqv2_agent, seed_carry, horizon, device, gamma, global_step):
    carry = seed_carry # Seed states
    feat_t = jax_to_torch(bridge.get_feat(carry), device)
    # Determine amount of exploration noise drqv2 actor uses when picking action
    stddev = schedule(drqv2_agent.stddev_schedule, global_step)

    feats, actions, rewards, conts, next_feats, weights = [], [], [], [], [], []
    weight = torch.ones(feat_t.shape[0], 1, device=device)

    for _ in range(horizon): # Imagines horizon steps forward
        # Using current policy pick out an action based on the imagined state
        with torch.no_grad():
            dist = drqv2_agent.actor(feat_t, stddev)
            action_t = dist.sample(clip=drqv2_agent.stddev_clip)
        action_np = action_t.detach().cpu().numpy()

        # Step the world model to predict next state based on the action just taken
        next_carry, next_feat_flat, reward_j, cont_j = bridge.img_step(carry, action_np)
        next_feat_t = jax_to_torch(next_feat_flat, device)
        reward_t = jax_to_torch(reward_j, device).reshape(-1, 1)
        cont_t = jax_to_torch(cont_j, device).reshape(-1, 1)

        # Store imagined state to return all together
        feats.append(feat_t)
        actions.append(action_t)
        rewards.append(reward_t)
        conts.append(cont_t)
        next_feats.append(next_feat_t)
        weights.append(weight)

        weight = weight * (gamma * cont_t)
        carry, feat_t = next_carry, next_feat_t

    return (
        torch.cat(feats), torch.cat(actions), torch.cat(rewards),
        torch.cat(conts), torch.cat(next_feats), torch.cat(weights)
    )