""" Bridges the JAX-side Dreamer world model to numpy/PyTorch-friendly calls """

import jax
import numpy as np
import ninjax as nj
from interop import flatten_leading_two_dims_np
from embodied.embodied.jax import transform

class WorldModelBridge:
    def __init__(self, wm_agent, action_key, obs_key='state'):
        self.agent = wm_agent
        self.model = wm_agent.model
        self.action_key = action_key
        self.obs_key = obs_key

        self.mesh = wm_agent.train_mesh
        self.ts = wm_agent.train_sharded
        tp = wm_agent.train_params_sharding
        tm = wm_agent.train_mirrored
        ar = wm_agent.partition_rules[1]

        self._encode_posterior = transform.apply(
            nj.pure(self.model.encode_posterior), self.mesh,
            (tp, tm, self.ts),
            (self.ts,),
            ar,
            static_argnums=(2,),
            single_output=True,
        )

        self._imagine_step = transform.apply(
            nj.pure(self.model.imagine_step), self.mesh,
            (tp, tm, self.ts, self.ts),
            (self.ts, self.ts, self.ts, self.ts),
            ar,
        )

        self._init_encode = transform.apply(
            nj.pure(self.model.init_encode), self.mesh,
            (tp, tm),
            (self.ts, self.ts),
            ar,
            static_argnums=(2,),
        )

        self._encode_step = transform.apply(
            nj.pure(self.model.encode_step), self.mesh,
            (tp, tm, self.ts, self.ts, self.ts, self.ts, self.ts),
            (self.ts, self.ts, self.ts),
            ar,
        )

        self._seed_counter = 0

    def _next_seed(self):
        self._seed_counter += 1
        return self.agent._seeds(self._seed_counter, self.agent.train_mirrored)

    def seed_pool(self, batch, batch_size):
        """ Encodes a batch of REAL sequences into posterior RSSM states
            used as launch points for imagination. `batch` must already be a
            JAX-converted Dreamer-format batch (see OGBenchMethods.to_jax) """
        pool = self._encode_posterior(
            self.agent.params, self._next_seed(), batch_size, batch)
        return flatten_leading_two_dims_np(pool)

    def place_seed(self, seed_carry_np):
        return jax.device_put(seed_carry_np, self.ts)

    def get_feat(self, dyn_carry):
        return self.model.feat2tensor(dyn_carry)

    def img_step(self, dyn_carry, action_np):
        action = {self.action_key: action_np.astype(np.float32)}
        action = jax.device_put(action, self.ts)
        return self._imagine_step(
            self.agent.params, self._next_seed(), dyn_carry, action)

    def init_encode(self, batch_size):
        return self._init_encode(self.agent.params, self._next_seed(), batch_size)

    def encode_step(self, enc_carry, dyn_carry, state_np, action_np, is_first_np):
        """ Feedsone real environment observation through the encoder to
            update the posterior RSSM state used both for online acting
            (choosing the next real action) and for evaluation """
        obs = jax.device_put({self.obs_key: state_np.astype(np.float32)}, self.ts)
        prevact = jax.device_put({self.action_key: action_np.astype(np.float32)}, self.ts)
        is_first = jax.device_put(is_first_np.astype(bool), self.ts)
        return self._encode_step(
            self.agent.params, self._next_seed(), enc_carry, dyn_carry, obs, prevact, is_first)