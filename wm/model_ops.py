import numpy as np
import torch

def decode_obs(bridge, carry, obs_key, device):
    """ Latent -> predicted observation, as a torch tensor.

        The actor and critic live in observation space, so every point where
        an imagined latent has to talk to them goes through here -- all three
        method modules (imagination_chunk, chunk_selector, dyna) and the
        wm_report diagnostics share this one function, so a decoder fix lands
        everywhere at once. The result is a PREDICTION, not a real state: it
        carries decoder error on top of whatever dynamics error accumulated to
        reach this latent. verify_decoder.py checks this pathway directly. """
    decoded = bridge.decode_state(carry)[obs_key]
    return torch.as_tensor(np.asarray(decoded, dtype=np.float32), device=device)