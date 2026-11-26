""" Unit tests for the tree-exploration components (acceptance tests A2,
    A4, A5, A6, A7, A8, A9). CPU, tiny networks, seconds each. """

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import elements
import numpy as np
import pytest
import torch

from tdmpc.agent import TDMPC2Model
from wm.chunk_selector import ChunkSelector
from wm.chunk_proposer import ChunkProposer
from wm.episode_bandit import EpisodeBandit
from wm.tree_selector import TreeSelector, TreeSwitchSelector
from wm.visit_memory import VisitMemory

DEVICE = torch.device('cpu')
OBS, ACT, CL = 8, 2, 3
CHUNK_DIM = ACT * CL


def tiny_tdmpc_cfg(num_dyn=4):
    return elements.Config(dict(
        horizon=4, latent_dim=16, mlp_dim=16, enc_dim=16, enc_layers=2,
        simnorm_dim=8, num_q=2, dropout=0.0, num_bins=11, vmin=-10.0,
        vmax=10.0, num_dyn=num_dyn, lr=3e-4, enc_lr_scale=0.3, tau=0.01,
        rho=0.5, consistency_coef=20.0, reward_coef=0.1, value_coef=0.1,
        entropy_coef=1e-4, grad_clip_norm=20.0, q_mode='step',
        chunk_boot_agg='mean', compile_nets=False))


def tree_cfg(**over):
    base = dict(beta=1.0, num_dyn=4, depth=3, beam=2, branch=2, root_bc=4,
                trust_cap=2.0, reach_cap=2.0, min_amount=0.05, ref_every=100,
                ref_rate=0.1, memory_offline=64, memory_online=64,
                memory_refresh_every=100000, memory_k=2, proposer_hidden=16,
                proposer_lr=3e-4, bandit=True, bandit_window=40, bandit_c=1.0)
    base.update(over)
    return elements.Config(base)


def switch_cfg(**over):
    base = dict(signal='both', lookback_chunks=2, rank_window=100,
                min_history=1, constant_stuck=0.5)
    base.update(over)
    return elements.Config(base)


class DummyPolicy:
    """ The policy surface the selector touches, deterministic weights. """

    def __init__(self, obs_dim=OBS, chunk_dim=CHUNK_DIM):
        self.chunk_dim = chunk_dim
        g = torch.Generator().manual_seed(99)
        self.w = torch.randn(chunk_dim, 1, generator=g)

    def noise(self, batch):
        return torch.randn(batch, self.chunk_dim)

    def sample_chunk(self, feat_n, noises=None):
        if noises is None:
            noises = self.noise(feat_n.shape[0])
        return torch.clamp(noises + 0.01 * feat_n[:, :1], -1.0, 1.0)

    def compute_flow_actions(self, feat_n, noises):
        return torch.clamp(0.5 * noises + 0.01 * feat_n[:, :1], -1.0, 1.0)

    def critic(self, feat_n, cands):
        q = cands @ self.w
        return torch.stack([q, q + 0.1], dim=0)

    def _agg(self, qs):
        return qs.mean(0)

    def act(self, state, eval_mode):
        raise AssertionError('selector should be enabled in these tests')


def fake_replay(n=64, obs_dim=OBS):
    rng = np.random.default_rng(0)
    rep = types.SimpleNamespace()
    rep.obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    rep.offline_size = n
    return rep


def make_memory(model, obs_dim=OBS, k=2):
    mem = VisitMemory(obs_dim, DEVICE, offline_size=64, online_size=64, k=k)
    mem.fill_offline(fake_replay(obs_dim=obs_dim), np.random.default_rng(0))
    mem.refresh(model.encode)
    return mem


def make_model(num_dyn=4, obs_dim=OBS, action_dim=ACT, chunk_len=CL):
    return TDMPC2Model(obs_dim, action_dim, DEVICE, tiny_tdmpc_cfg(num_dyn),
                       0.99, chunk_len=chunk_len)


def make_tree_selector(model, policy, tree=None, switch=None, bandit=None,
                       gamma=0.99, n=4):
    tree = tree if tree is not None else tree_cfg()
    proposer = ChunkProposer(model.cfg.latent_dim, CHUNK_DIM, DEVICE,
                             hidden=tree.proposer_hidden, lr=tree.proposer_lr)
    memory = make_memory(model)
    cls = TreeSelector if switch is None else TreeSwitchSelector
    kwargs = {} if switch is None else {'switch': switch}
    return cls(model, policy, ACT, CL, n, gamma, DEVICE, tree=tree,
               proposer=proposer, memory=memory, bandit=bandit, seed=0,
               **kwargs)


# ------------------------------------------------------------------- A2

def test_a2_plain_path_equals_combined():
    """ Same state, same torch seed: TreeSelector with amount 0 returns the
        same chunk as ChunkSelector, and the global torch RNG state after
        the call is identical. """
    state = np.linspace(-1, 1, OBS).astype(np.float32)

    torch.manual_seed(5)
    np.random.seed(5)
    plain = ChunkSelector(None, DummyPolicy(), ACT, CL, 4, 0.99, DEVICE)
    plain.begin_episode()
    torch.manual_seed(11)
    chunk_plain = plain.select(state)
    rng_plain = torch.get_rng_state().clone()

    torch.manual_seed(5)
    np.random.seed(5)
    model = make_model()
    tree = make_tree_selector(model, DummyPolicy(),
                              tree=tree_cfg(beta=0.0), bandit=None)
    tree.begin_episode()
    torch.manual_seed(11)
    chunk_tree = tree.select(state)
    rng_tree = torch.get_rng_state().clone()

    assert np.array_equal(chunk_plain, chunk_tree)
    assert torch.equal(rng_plain, rng_tree)

    torch.manual_seed(11)
    eval_plain = plain.select(state, eval_mode=True)
    torch.manual_seed(11)
    eval_tree = tree.select(state, eval_mode=True)
    assert np.array_equal(eval_plain, eval_tree)


# ------------------------------------------------------------------- A4

def test_a4_imagine_chunk_equals_step_loop():
    model = make_model()
    torch.manual_seed(0)
    z = torch.randn(6, model.cfg.latent_dim)
    z = model.net.encode(torch.randn(6, OBS))     # a latent on the simplices
    actions = torch.rand(6, CL, ACT) * 2 - 1
    z_end, dis = model.imagine_chunk(z, actions)
    z_ref, dis_ref = z.clone(), torch.zeros(6, 1)
    for k in range(CL):
        z_ref, d = model._step(z_ref, actions[:, k])
        dis_ref = dis_ref + d / CL
    assert torch.allclose(z_end, z_ref, atol=1e-6)
    assert torch.allclose(dis, dis_ref.squeeze(-1), atol=1e-6)


# ------------------------------------------------------------------- A5

def test_a5_unvisitedness():
    mem = VisitMemory(4, DEVICE, offline_size=32, online_size=8, k=1)
    rep = types.SimpleNamespace()
    rep.obs = np.eye(4, dtype=np.float32).repeat(8, axis=0)
    rep.offline_size = 32
    mem.fill_offline(rep, np.random.default_rng(0))
    mem.refresh(lambda x: x)                      # identity encoder
    assert mem.typical_distance is not None
    assert np.isfinite(mem.typical_distance) and mem.typical_distance > 0
    stored = torch.as_tensor(rep.obs[:1])
    far = torch.full((1, 4), 50.0)
    u_stored = float(mem.unvisitedness(stored))
    u_far = float(mem.unvisitedness(far))
    assert u_stored == pytest.approx(0.0, abs=1e-5)
    assert u_far > 10.0


# ------------------------------------------------------------------- A6

def test_a6_trust_cut():
    torch.manual_seed(3)
    np.random.seed(3)
    model = make_model(num_dyn=4)
    sel = make_tree_selector(model, DummyPolicy(), tree=tree_cfg(),
                             bandit=None)          # amount = beta = 1
    sel.begin_episode()
    sel.reference = [1e-12] * int(sel.tree.depth)  # reference forced tiny
    state = np.linspace(-1, 1, OBS).astype(np.float32)
    chunk = sel.select(state)
    stats = sel.pop_stats()
    assert stats['select/tree_grown_frac'] == 1.0
    assert stats['select/reach_max'] == 0.0
    assert stats['select/reach_picked'] == 0.0
    assert stats['select/bonus_over_spread'] == pytest.approx(0.0, abs=1e-9)
    assert stats['select/pick_changed'] == 0.0     # pick equals critic argmax
    assert 'select/trusted_frac_d2' not in stats   # no level-2 nodes
    assert chunk.shape == (CL, ACT)


# ------------------------------------------------------------------- A7

class PathModel:
    """ Deterministic fake: latent = obs, imagine_chunk shifts the latent by
        the chunk's first action and never disagrees. """

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.q_mode = 'step'

    def encode(self, obs):
        return obs.clone()

    def imagine_chunk(self, z, chunk_actions):
        shift = torch.zeros_like(z)
        shift[:, 0] = chunk_actions[:, 0, 0]
        return z + shift, torch.zeros(z.shape[0])


class PathMemory:
    typical_distance = 1.0

    def unvisitedness(self, latents):
        return latents[:, 0].clamp_min(0.0)

    def add_online(self, obs, latent):
        pass

    def refresh(self, encode, generator=None):
        pass


class PathProposer:
    """ Proposes a constant chunk whose first action is +0.5. """

    def __init__(self, chunk_dim):
        self.chunk_dim = chunk_dim

    def propose(self, latents, generator):
        out = torch.zeros(latents.shape[0], self.chunk_dim)
        out[:, 0] = 0.5
        return out


class PathPolicy(DummyPolicy):
    """ Two fixed root candidates with identical critic scores. """

    def sample_chunk(self, feat_n, noises=None):
        out = torch.zeros(feat_n.shape[0], self.chunk_dim)
        out[0, 0] = 0.6                            # root 0 ends at latent 0.6
        out[1, 0] = 0.5                            # root 1 ends at latent 0.5
        return out

    def critic(self, feat_n, cands):
        q = torch.zeros(cands.shape[0], 1)
        return torch.stack([q, q], dim=0)


def test_a7_backup_and_carry():
    tree = tree_cfg(depth=3, beam=1, branch=1, root_bc=0, trust_cap=2.0,
                    reach_cap=10.0, min_amount=0.05)
    model = PathModel(latent_dim=OBS)
    sel = TreeSelector(model, PathPolicy(), ACT, CL, 2, 1.0, DEVICE,
                       tree=tree, proposer=PathProposer(CHUNK_DIM),
                       memory=PathMemory(), bandit=None, seed=0)
    sel.begin_episode()
    sel.reference = [1.0, 1.0, 1.0]                # everything trusted
    state = np.zeros(OBS, dtype=np.float32)
    chunk = sel.select(state)
    stats = sel.pop_stats()
    # Hand calculation (gamma = 1, start latent 0): root 0's level-1 node
    # scores 0.6, root 1's 0.5. Beam 1 expands root 0 only; each deeper
    # chunk adds +0.5 to the first latent coordinate, so its depth-2 node
    # scores 1.1 and its depth-3 node 1.6. Reach: root 0 = 1.6, root 1 =
    # 0.5 (its own level-1 node).
    assert stats['select/reach_picked'] == pytest.approx(1.6)
    assert stats['select/reach_max'] == pytest.approx(1.6)
    assert stats['select/reach_mean'] == pytest.approx((1.6 + 0.5) / 2)
    assert stats['select/best_depth_picked'] == 3.0
    # Executed root is 0 (bonus breaks the critic tie); the carried chunk is
    # the depth-2 chunk on the best path: the proposer's constant chunk.
    assert chunk[0, 0] == pytest.approx(0.6)
    assert sel.carried is not None
    assert float(sel.carried[0]) == pytest.approx(0.5)


# ------------------------------------------------------------------- A8

def make_switch_selector(gamma=0.9, **sw_over):
    tree = tree_cfg(depth=1, root_bc=0, beta=1.0, min_amount=1e9)
    model = PathModel(latent_dim=OBS)
    sel = TreeSwitchSelector(model, PathPolicy(), ACT, CL, 2, gamma, DEVICE,
                             tree=tree, proposer=PathProposer(CHUNK_DIM),
                             memory=PathMemory(), bandit=None, seed=0,
                             switch=switch_cfg(**sw_over))
    sel.begin_episode()
    return sel


def test_a8_expected_progress_zero():
    gamma = 0.9
    sel = make_switch_selector(gamma=gamma)
    crit = torch.tensor([-5.0, -10.0])             # current value -5
    sel.explore_amount(crit)                       # first decision: no reward
    sel._observe_reward(-5.0 * (1 - gamma))        # reward / (1-gamma) = -5
    sel.explore_amount(crit)
    stats = sel.pop_stats()
    assert stats['select/switch_expected_progress'] == pytest.approx(0.0, abs=1e-9)


def test_a8_broken_promise_hand_calculation():
    gamma = 0.9
    sel = make_switch_selector(gamma=gamma, lookback_chunks=2)
    v1, v2, v3 = -4.0, -6.0, -7.0
    sel.explore_amount(torch.tensor([v1, v1 - 1]))   # decision 1 (no reward yet)
    sel._observe_reward(-1.0)                        # r1
    sel.explore_amount(torch.tensor([v2, v2 - 1]))   # decision 2: 1 earlier decision
    sel._observe_reward(-2.0)                        # r2
    sel.explore_amount(torch.tensor([v3, v3 - 1]))   # decision 3: lookback hits
    stats = sel.pop_stats()
    delivered = -1.0 + gamma * -2.0 + gamma ** 2 * v3
    expected_bp = max(0.0, v1 - delivered)
    assert stats['select/switch_broken_promise'] * 2 == pytest.approx(expected_bp)
    # (decisions 2 and 3 accumulated -- decision 1 has no reward yet and
    # logs nothing; decision 2's broken promise is 0)


def test_a8_rank_conventions():
    sel = make_switch_selector(gamma=0.9)
    sel._progress_window = [1.0, 2.0, 3.0]
    sel._promise_window = [0.0, 0.0, 0.0]
    sel._observe_reward(0.0)
    sel._decisions = []                              # broken promise stays 0
    crit = torch.tensor([2.0, 0.0])                  # expected progress = 2.0
    sel.explore_amount(crit)
    stats = sel.pop_stats()
    # strictly greater: only 3.0 > 2.0 -> rank 1/3
    assert stats['select/switch_low_progress_rank'] == pytest.approx(1 / 3)
    # zeros in the broken-promise window are NOT strictly less than 0
    assert stats['select/switch_broken_promise_rank'] == pytest.approx(0.0)


def test_a8_constant_signal_respects_bandit_arm():
    sel = make_switch_selector(gamma=0.9, signal='constant',
                               constant_stuck=0.5, min_history=0)
    sel.bandit = EpisodeBandit(4, 1.0)
    sel.bandit.arm = 0
    sel._observe_reward(-1.0)
    assert sel.explore_amount(torch.tensor([0.0, -1.0])) == 0.0
    sel.bandit.arm = 1
    sel._observe_reward(-1.0)
    assert sel.explore_amount(torch.tensor([0.0, -1.0])) == pytest.approx(0.5)


# ------------------------------------------------------------------- A7b: bandit behaviour unchanged

def test_bandit_untried_first_then_ucb():
    b = EpisodeBandit(window=40, c=1.0)
    assert b.choose() == 0                           # untried arm 0 first
    b.report(-100.0)
    assert b.choose() == 1                           # then untried arm 1
    b.report(0.0)
    assert b.choose() == 1                           # higher mean wins early


# ------------------------------------------------------------------- A9

def test_a9_old_checkpoint_loads():
    path = pathlib.Path('/tmp/old_wm_explore_model.pt')
    if not path.exists():
        pytest.skip('old-shaped checkpoint not present on this machine')
    import ruamel.yaml as yaml
    root = pathlib.Path(__file__).resolve().parent.parent
    cfgs = yaml.YAML(typ='safe').load(open(root / 'configs.yaml'))
    cfg = elements.Config(cfgs['defaults']['tdmpc']).update({'compile_nets': False})
    model = TDMPC2Model(46, 5, DEVICE, cfg, 0.99, num_dyn=4, chunk_len=5)
    state = torch.load(path, map_location=DEVICE)
    assert 'proposer' not in state                   # genuinely old-shaped
    model.load_state_dict_all(state)
    z, dis = model.imagine_chunk(model.encode(torch.zeros(2, 46)),
                                 torch.zeros(2, 5, 5))
    assert z.shape == (2, cfg.latent_dim) and dis.shape == (2,)
