# Action-chunked agent, with and without a world model

Two training arms that differ in exactly one thing: whether a world model exists.
Everything about the policy — architecture, chunk length, behavior constraint,
critic ensemble, discount, reward shift — is shared, so the comparison isolates
the world model.

## Files

| File | What it is |
|---|---|
| `train_sac_wm.py` | SAC + world model, single step (was `train_joint.py`) |
| `train_sac.py` | SAC baseline, single step |
| `train_sac_chunked_wm.py` | Chunked + world model |
| `train_sac_chunked.py` | Chunked, no world model |
| `configs.yaml` | All four arms |
| `dreamer/` | JAX Dreamer agent, RSSM, JAX/torch bridge |
| `wm/imagination.py` | Single-step rollout through the RSSM |
| `wm/imagination_chunk.py` | Chunked rollout through the RSSM |
| `wm/check_rollout_animation.py` | Model-accuracy check against real trajectories |
| `sac/sac_wm_agent.py` | Single-step SAC actor/critic |
| `sac/evaluation.py` | Single-step eval |
| `sac_chunked/sac_chunk_agent.py` | Noise-conditioned chunk actor + chunk critic ensemble |
| `sac_chunked/flow_bc.py` | Flow-matching behavior model over chunks |
| `sac_chunked/chunk_utils.py` | Chunk pooling, chunked λ-returns, chunk-pair indexing, coherence metric |
| `sac_chunked/evaluation_chunk.py` | Chunked eval, coherence measurement, CSV logging |
| `helpers/` | `interop`, `ogbench_methods`, `sac_wm_utils`, `online_replay`, `comparison`, `compare_to_paper`, `test_physics` |

Everything runs from the repo root:

```bash
python train_sac_chunked_wm.py --seed=0
python helpers/compare_to_paper.py train_sac_chunked_wm_out/eval_log.csv ...
```

Scripts that live in subfolders put the repo root on `sys.path` themselves, so
`python wm/check_rollout_animation.py` works without `-m`.

---

## 1. The policy

Three networks, all shared between the two arms.

**Chunk actor.** `actor(feat, z) -> 25 numbers` (5 actions × 5 dims). Randomness
is an *input*. A per-dimension Gaussian head would give each of the 5 actions its
own independent draw, so executing the chunk open loop would just be jitter —
strictly worse than single-step SAC, since you lose feedback and gain nothing.
Here the network sees the whole noise draw and can spend it on one consistent
motion. Different `z` means a different motion, not a different amount of shaking.

**Flow behavior model.** Learns the distribution of real 5-step chunks in the
replay data. It knows nothing about the task, only what plausible motion looks
like. This is what supplies coherence: the actor is kept near it.

**Chunk critic.** `Q(feat, chunk)`, ensemble of 10, averaged rather than
min-reduced (matching the QC-FQL critic loss; a min over 10 members would be
severely pessimistic).

### Why the same `z` goes into both networks

The actor loss is:

```
-Q(feat, chunk) / |Q|  +  alpha * || chunk - flow(feat, z) ||^2
```

with the same `z` in `actor(feat, z)` and `flow(feat, z)`.

If the second term just said "match the data," and two motions were valid from
this state — reach left, reach right — the actor would learn their average and
reach straight forward into nothing. That's the classic behavior-cloning failure.
Sharing `z` changes the question to: *for this particular draw, what would the
behavior model have done?* Nothing gets averaged, and the actor reproduces the
behavior model's full range rather than its mean.

`|Q|` normalization keeps `alpha` meaningful as Q grows during training,
otherwise a value tuned early silently becomes weak later.

### What went away

No `log_std`, no `log_prob`, no `log_ent_coef`, no `target_entropy`, no
`MU_CLIP`, no `MU_REG`. The mu-explosion failure mode you debugged cannot arise:
there is no unbounded pre-tanh mean chasing a Q gradient with nothing opposing
it, because the distillation term is a direct opposing force on the output.

What replaces entropy collapse as the thing to watch is **chunk diversity** —
see §6.

---

## 2. World-model arm, step by step

### Acting

Every environment step:

1. Encode the observation into the RSSM posterior (`encode_step`) — **every
   step, unconditionally**. Committing to actions is not a reason to stop
   looking; the posterior must be current when the next chunk decision is made.
2. If the chunk buffer is exhausted, query the actor once and refill it with 5
   actions. Otherwise pop the next buffered action.
3. Step the environment, store the transition.

The buffer is reset on episode end and after every eval (eval resets the same
env object).

### World model update

Unchanged from `train_sac_wm.py`.

### Flow behavior model update

Uses the same batch as the imagination seeds, so no extra sampling:

1. `bridge.seed_pool(batch)` encodes the sequences into posterior latents,
   flattened over `(batch, seq_len)`.
2. `chunk_pair_indices` finds flat indices `i` where the 5 actions starting at
   that timestep stay inside the sampled sequence and inside one episode. Windows
   crossing an episode boundary are dropped — the last action of an episode is
   zero padding, not a real action.
3. Those latents are placed and featurized; the matching chunks come from
   `batch['action'][seq, t:t+5]`.

Dreamer's convention is that `action[t]` is the action taken **from** `state[t]`
(see `ogbench_to_dreamer_episode`), so the chunk launched at latent `t` is
`action[t:t+5]`. Getting this off by one would train the behavior model on
chunks that don't correspond to their latent, which would fail silently.

Imagination seeds are subsampled from the *unfiltered* pool, so the seed
distribution isn't biased away from late timesteps.

### Imagined rollout

For each of 3 chunks:

1. Query the actor once → 5 actions.
2. **Step the RSSM five times**, one action per step. The model is never asked
   to jump. Every predicted latent is computed from the realized previous
   latent, so causality holds exactly — and per-step reward resolution survives,
   which matters when the reward is a single transition.
3. Pool the 5 steps into one chunk-level transition (§3).
4. The end latent becomes the next chunk's start latent.

Output: 3 chunk transitions per starting latent.

### Critic target

```
gamma_h = gamma ** 5
ret = chunk_reward[t] + gamma_h * chunk_cont[t] * ((1-lam)*v[t] + lam*ret)
```

Identical recursion to your existing `lambda_targets`, run over chunks. The only
change is `gamma -> gamma**5`, because 5 real steps elapse between entries.

`v[t]` is the target critic's score for the chunk the current actor would take
at that boundary — computed fresh, which handles the final boundary correctly.

The entropy term is **not** in the target. With 25 dimensions its magnitude
would be ~5× larger and it would inject a large offset into every target. Since
the noise-conditioned actor has no `log_prob` anyway, it's simply gone.

---

## 3. Reward and termination — the part most likely to break

Five single-step `(reward, cont)` pairs become one chunk-level pair:

```
chunk_reward = sum_k  gamma^k * (prod_{j<k} cont_j) * r_k
chunk_cont   = prod_k cont_k
```

The `prod_{j<k} cont_j` factor is load-bearing. Success at step 3 of 5 means
steps 4 and 5 never happen, so their rewards must not be credited. Without the
mask, reward leaks in from steps that don't exist — and on a task with exactly
one reward event per episode, a single leaked reward per chunk is a large
fraction of the total signal.

This is also the concrete reason the RSSM stays single-step. A model that jumps
5 steps has no way to express "the episode ended partway through this jump."

The unpooled per-step rewards are returned from the rollout and logged as
`diagnosis/intra_chunk_reward_first` / `_last`. If the masking is wrong, reward
keeps appearing at late positions after `rollout_cont_last_chunk` has collapsed.
Check those three numbers together on the first run.

The `{-1, 0} -> {0, +1}` reward shift is applied per step, before pooling, so it
composes with the discount the same way it did in the single-step version.

---

## 4. No-world-model arm

Same three networks, same losses. The difference is where chunk transitions come
from.

`ChunkTransitionReplay` is your flat ring buffer plus an episode id per slot. A
window starting at `i` is valid only if all 5 slots share an episode id and the
window doesn't straddle the write head — otherwise the "chunk" would splice
together unrelated timesteps. Rewards are pooled with the identical masked
discounted sum (`pool_chunk_np`), so both arms build chunk rewards the same way.

The target is a single 5-step backup:

```
target = chunk_reward + gamma^5 * chunk_cont * V(next_obs)
```

No λ-returns, because there's no imagined horizon to run them over. This is
precisely the unbiased n-step backup the QC paper is built around: the critic
scores the whole chunk that produced these rewards, so there's no off-policy
mismatch between the action being valued and the rewards being summed.

**What this arm structurally cannot do:** the only chunks available are chunks
that were actually executed. There is no way to ask what a *different* chunk
would have done from the same state. That absence is the world model's main
advantage, and it's why the comparison is worth running.

---

## 5. What changed from your single-step version, and why

| Change | Reason |
|---|---|
| `num_chunks: 3`, not 15 | 3 × 5 = 15 imagined env steps, the depth you verified. 15 chunks would be 75 steps of hallucination. |
| `imagination_batch: 4096` | 3 chunks per rollout gives 3 critic examples instead of 15. More starting latents partly offsets it. Drop this first if the rollout is too slow. |
| `ensemble: 10` | QC reports 10 helps substantially over 2 and it's nearly free at this network size. Applied to both arms so it isn't a confound. |
| Entropy removed from the target | 25-dim `log_prob` would be ~5× larger and offset every target. |
| Eval resyncs env + RSSM carry | `eval_chunk_in_env` resets the same env object; the training loop's `obs`, carries and chunk buffer are all stale afterward. `train_sac_wm.py` has this bug; the chunked version fixes it. |
| λ over 3 chunks | `lam: 0.95` across 3 entries is nearly a pure 3-chunk return. Worth sweeping — it was tuned for 15 steps. |

---

## 6. What to watch

| Metric | Meaning |
|---|---|
| `sac/chunk_diversity` | **The Option-A failure mode.** Spread of chunks across 8 different `z` at fixed state. Trending to zero means the actor has learned to ignore `z` and exploration is dead → raise `bc_alpha`. This is the replacement for `actor_std_mean`. |
| `diagnosis/actor_intra_chunk_jerk` | Mean absolute difference between consecutive actions inside a chunk. Should *fall* as the behavior constraint takes hold. High and flat means the chunk is still open-loop noise. |
| `eval/coherence` | QC's Figure-4 metric: end-effector distance travelled over 5 steps. Should be **higher** than a single-step run. If it isn't, the chunk distribution isn't coherent — fix `bc_alpha`, don't touch chunk length or learning rate. |
| `diagnosis/intra_chunk_reward_first` / `_last` | Reward leak check (§3). |
| `sac/actor_bc_term` | If it goes to ~0, `alpha` is so strong the actor is pure imitation. If it grows without bound, too weak. |
| `diagnosis/critic_ensemble_spread` | Disagreement across the 10 critics. Collapsing to zero early means overconfidence. |
| `sac/bc_flow_loss` | Should fall and plateau. Doesn't converge → the RSSM latent is still moving too fast under the behavior model (expected early; if it persists, train the flow model less often or with a lower lr). |

**`eef_slice` in the config defaults to `[0, 3]` and is not verified.** Check it
against the actual OGBench observation layout before trusting the coherence
numbers — a wrong slice produces a plausible-looking but meaningless curve.

---

## 7. Eval output

Each arm writes `<out_dir>/eval_log.csv`, one row per eval:

```
arm, env, seed, chunk_len, env_step, gradient_updates,
mean_return, success_rate, coherence, mean_episode_len, wall_time_s
```

Then:

```
python helpers/compare_to_paper.py train_sac_chunked_wm_out/eval_log.csv train_sac_chunked_out/eval_log.csv
```

Prints per-seed and per-arm final/best success, coherence, and **environment
steps to first cross 25% / 50% / 75% success**.

That last one is the number that matters. Every advantage the world model has
here is a sample-efficiency argument — none of them predicts a higher ceiling.
If both arms eventually solve cube-double, comparing endpoint scores will show
nothing, and that's the most likely way this experiment produces a null result
for the wrong reason.

`gradient_updates` is logged because the two arms update at different cadences —
equal environment steps is **not** equal compute, and any comparison should say
so out loud.

### On the paper reference numbers

`helpers/compare_to_paper.py` prints approximate cube-double figures read off Figure 2
of the QC paper by eye. They are not a benchmark: the paper does 1M offline
pretraining then 1M online steps, while these runs seed the buffer from the
offline data and go straight online. Order of magnitude only.

Also worth knowing: OGBench cube domains have 5 task variants, and the paper
tunes on `task2` then reports the average over all 5. If you want the closest
comparison, run `task2`. Verify the exact env id string before a long run.

---

## 8. Running it

```bash
python train_sac_chunked_wm.py --seed=0
python train_sac_chunked.py   --seed=0
```

Defaults are `cube-double-play-singletask-v0`, 500k steps, chunk length 5.
Override anything from the command line as before:

```bash
python train_sac_chunked_wm.py --train_sac_chunked_wm.chunk.bc_alpha=1000 --seed=1
```

**Sweep `bc_alpha` first, before anything else.** It is *the* hyperparameter of
this method — too low and chunks collapse to open-loop noise, too high and the
critic can't improve on the demonstrations. Try `{100, 300, 1000}` on short
runs and pick by `eval/coherence` and `sac/chunk_diversity`, not by return.

**Five seeds per arm.** With two, a 10-point gap is unfalsifiable, and the gap
here is expected to be smaller than your 85-vs-10 single-step result.

### A cheap check worth running first

Encode a real trajectory, unroll the model 5 steps with the *recorded* actions,
decode, and compare against the real states — `wm/check_rollout_animation.py` and
`bridge.decode_state` already do most of this. If 5-step rollouts track reality,
the causality concern about chunking is answered with data. If they don't,
you've learned something important before building on top.

---

## 9. What is and isn't verified

**Tested here, with unit and integration tests:**
chunk pooling and termination masking (including terminate-at-step-0 and
mid-chunk); the numpy/torch pooling twins agreeing; chunked λ-returns against
closed-form values at λ=0 and λ=1 and with `cont=0`; `chunk_pair_indices`
latent↔chunk alignment and episode-boundary rejection; the coherence metric;
actor/critic/flow shapes, gradient flow and parameter updates; deterministic
eval; the diversity diagnostic firing when the actor is forced to ignore `z`;
the full rollout against a mock bridge (shapes, monotone weight decay,
zero-weight after termination); the replay buffer's window filtering and offline
episode-id construction; `helpers/compare_to_paper.py` on synthetic logs.

An end-to-end learning test on a toy imagined MDP: imagined reward climbed
0.30 → 0.76 over 400 updates while intra-chunk jerk fell 0.47 → 0.25 and chunk
diversity stayed healthy. The loop learns, and the behavior constraint smooths
chunks as intended.

**Not verified — no JAX, `embodied`, `ogbench` or GPU available here:**
everything touching the RSSM bridge (`seed_pool` / `place_seed` / `img_step` /
`get_feat` shapes and dtypes in the chunked loop), the `elements` config
plumbing for the two new sections, and the real environment interaction.

Expect a debugging pass at the JAX/PyTorch boundary in `imagine_chunk_rollout` —
that's where your past bugs have lived, and it's the one part I couldn't run.
