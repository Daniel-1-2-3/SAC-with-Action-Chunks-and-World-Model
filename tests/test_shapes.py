""" Shape checks with random tensors: no env, no dataset, CPU, seconds.
    Small nets so the whole file runs in well under 5 s. """

import sys
import pathlib

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tdmpc.agent import TDMPC2Model  # noqa: E402
from wm.chunk_selector import ChunkSelector  # noqa: E402
from sac_chunked.qc_agent import QCAgent  # noqa: E402

OBS, ACT, CHUNK, NDYN = 37, 5, 5, 5
DEV = torch.device('cpu')


class Cfg:
    """ The tdmpc block, small. """
    train_every = 2; batch_size = 8; horizon = 5
    latent_dim = 32; mlp_dim = 32; enc_dim = 32; enc_layers = 2; simnorm_dim = 8
    num_q = 2; dropout = 0.01; num_dyn = 1; online_frac = 0.5
    ref_mode = 'rollout'; reward_weight_shrink = 0.5
    num_bins = 101; vmin = -10.0; vmax = 10.0
    lr = 3e-4; enc_lr_scale = 0.3; tau = 0.01; rho = 0.5
    consistency_coef = 20.0; reward_coef = 0.1; value_coef = 0.1
    entropy_coef = 1e-4; grad_clip_norm = 20.0
    diag_windows = 0; diag_depth = 1; compile_nets = False


def make_model(ref_mode='rollout', novelty='reward', novelty_at='path', rollout_chunks=1):
    cfg = Cfg(); cfg.ref_mode = ref_mode
    torch.manual_seed(0)
    return TDMPC2Model(OBS, ACT, DEV, cfg, 0.99, num_dyn=NDYN, novelty=novelty,
                       novelty_at=novelty_at, rollout_chunks=rollout_chunks, chunk_len=CHUNK)


class StubPolicy:
    """ Random chunks and random critic scores of the right shapes. """
    chunk_len, action_dim = CHUNK, ACT
    ensemble = 2

    def sample_chunk(self, feat):
        return torch.rand(feat.shape[0], CHUNK * ACT) * 2 - 1

    def critic(self, feat, chunk):
        return torch.randn(self.ensemble, feat.shape[0], 1)

    def _agg(self, qs):
        return qs.mean(0)

    def act(self, state, eval_mode=False):
        return np.zeros((CHUNK, ACT), np.float32)


# ------------------------------------------------------------------ model

@pytest.mark.parametrize('ref_mode', ['step', 'rollout'])
def test_update_novelty_reference(ref_mode):
    m = make_model(ref_mode=ref_mode)
    B, H = 8, Cfg.horizon
    zs = torch.rand(B, H + 1, Cfg.latent_dim)
    action = torch.rand(B, H, ACT) * 2 - 1
    valid = torch.ones(B, H, 1); valid[0, 2:] = 0.0       # one window crosses an episode end
    var0 = torch.rand(B, Cfg.latent_dim)
    before = m.data_disagreement
    out = m.update_novelty_reference(zs, action, valid, var0=var0)
    assert isinstance(out, float) and out > 0 and out == m.data_disagreement
    assert out != before
    # rollout mode with no fully valid window: unchanged
    if ref_mode == 'rollout':
        v = m.update_novelty_reference(zs, action, torch.zeros(B, H, 1), var0=var0)
        assert v == out


def test_reward_weights_mean_one_per_row():
    m = make_model()
    z = torch.rand(6, Cfg.latent_dim); a = torch.rand(6, ACT) * 2 - 1
    w = m.reward_weights(z, a)
    assert w.shape == (6, Cfg.latent_dim)
    assert torch.allclose(w.mean(-1), torch.ones(6), atol=1e-5)
    assert (w >= 0).all()


@pytest.mark.parametrize('novelty', ['mean', 'reward'])
def test_reduce_disagreement_shape(novelty):
    m = make_model(novelty=novelty)
    z = torch.rand(6, Cfg.latent_dim); a = torch.rand(6, ACT) * 2 - 1
    d = m.reduce_disagreement(torch.rand(6, Cfg.latent_dim), z, a)
    assert d.shape == (6, 1) and (d >= 0).all()


@pytest.mark.parametrize('novelty_at,rollout_chunks', [('path', 1), ('end', 1), ('path', 2)])
def test_path_disagreement_shape(novelty_at, rollout_chunks):
    m = make_model(novelty_at=novelty_at, rollout_chunks=rollout_chunks)
    z = m.encode(torch.randn(4, OBS))
    d = m.path_disagreement(z, torch.rand(4, CHUNK, ACT) * 2 - 1)
    assert d.shape == (4,) and (d >= 0).all()


def test_model_update_shapes():
    m = make_model()
    B, H = 8, Cfg.horizon
    obs = torch.randn(B, H, OBS); nxt = torch.randn(B, H, OBS)
    act = torch.rand(B, H, ACT) * 2 - 1
    rew = -torch.ones(B, H, 1); mask = torch.ones(B, H, 1); valid = torch.ones(B, H, 1)
    metrics = m.update(obs, nxt, act, rew, mask, valid, metrics_on=True)
    assert 'loss_total' in metrics and np.isfinite(metrics['loss_total'])
    assert m.data_disagreement > 0


# --------------------------------------------------------------- selector

def make_selector(model, **kw):
    return ChunkSelector(model, StubPolicy(), ACT, CHUNK, n=16, gamma=0.99, device=DEV,
                         bonus_beta=1.0, **kw)


@pytest.mark.parametrize('controller', ['gate', 'bandit'])
@pytest.mark.parametrize('bonus_scale', ['unc', 'spread'])
@pytest.mark.parametrize('novelty', ['model', 'none'])
def test_select_uncertainty_scaled_shape(bonus_scale, novelty, controller):
    m = make_model()
    sel = make_selector(m, bonus_scale=bonus_scale, novelty=novelty, controller=controller)
    sel.begin_episode()
    feat_n = torch.randn(1, OBS).repeat(16, 1)
    cands = sel.policy.sample_chunk(feat_n)
    qs = sel.policy.critic(feat_n, cands)
    out = sel._select_uncertainty_scaled(feat_n, cands, qs, qs.mean(0).squeeze(-1))
    assert out.shape == (CHUNK, ACT)
    stats = sel.pop_stats()
    assert 'select/novelty_mean' in stats and 'select/pick_changed' in stats


def test_select_end_to_end_train_and_eval():
    m = make_model()
    sel = make_selector(m, bonus_scale='spread', novelty='model')
    state = np.random.randn(OBS).astype(np.float32)
    assert sel.select(state).shape == (CHUNK, ACT)
    assert sel.select(state, eval_mode=True).shape == (CHUNK, ACT)


# --------------------------------------------------------------- QC agent

def make_qc(n=8):
    torch.manual_seed(0)
    return QCAgent(repr_dim=OBS, action_dim=ACT, chunk_len=CHUNK, device=DEV, lr=3e-4,
                   hidden_dim=32, num_layers=2, critic_target_tau=0.005, ensemble=2,
                   num_samples=n, flow_steps=10)


def test_qc_sample_chunks_bc_shape_and_clip():
    ag = make_qc()
    c = ag.sample_chunks_bc(torch.randn(1, OBS), 8)
    assert c.shape == (8, CHUNK * ACT)
    assert c.min() >= -1.0 and c.max() <= 1.0
    assert ag.sample_chunks_bc(torch.randn(3, OBS), 8).shape == (3 * 8, CHUNK * ACT)


def test_qc_best_of_n_picks_the_max_q_candidate():
    ag = make_qc()
    feat = torch.randn(1, OBS)
    torch.manual_seed(1); cands = ag.sample_chunks_bc(feat, 8)
    q = ag._agg(ag.critic(feat.expand(8, -1), cands)).squeeze(-1)
    torch.manual_seed(1); pick = ag.best_of_n(feat)
    assert pick.shape == (1, CHUNK * ACT)
    assert torch.allclose(pick[0], cands[q.argmax()])


def test_qc_act_target_and_updates():
    ag = make_qc()
    a = ag.act(np.random.randn(OBS).astype(np.float32), eval_mode=True)
    assert a.shape == (CHUNK, ACT) and np.abs(a).max() <= 1.0
    B = 6
    obs = torch.randn(B, OBS); chunk = torch.rand(B, CHUNK * ACT) * 2 - 1
    tv = ag.chunk_target_values(torch.randn(B, OBS))
    assert tv.shape == (B, 1)
    m = ag.update_critic(obs, chunk, -torch.ones(B, 1) + 0.99 ** CHUNK * tv, torch.ones(B, 1))
    assert np.isfinite(m['critic_loss'])
    m = ag.update_actor(obs, torch.ones(B, 1), bc_feat=obs, bc_chunk=chunk,
                        bc_valid=torch.ones(B, CHUNK))
    assert set(m) == {'actor_loss', 'bc_flow_loss'}
    ag.update_target()
    assert ag.chunk_diversity(obs) >= 0.0
    assert sorted(ag.state_dict_all()) == ['actor_bc_flow', 'critic', 'critic_target']
