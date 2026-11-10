"""The differentiable integrator reproduces the implicit reference solver."""
import numpy as np
import pytest
import torch

from ht1b.config import Circuit, Controls
from s6.circuit_scan import CircuitScan, POTS

fast_solver = pytest.importorskip('ht1b.fast_solver')

RATE = 48000
CONTROLS = (Controls(threshold=.95, ratio=.8, attack=0., release=.9),
            Controls(threshold=.6, ratio=.3, attack=.5, release=.2, makeup_db=3.))


def burst(seconds=.3, seed=0):
    """Loud tone bursts separated by near silence: exercises attack, release and clipping."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds*RATE))/RATE
    envelope = np.where((t % .1) < .04, .8, .002)
    return envelope*np.sin(2*np.pi*220*t)+1e-4*rng.standard_normal(t.size)


def reference(x, controls):
    solver = fast_solver.FastCircuitSolver(Circuit(), controls, RATE)
    y, states = [], []
    for v in x:  # sample by sample to read the state trajectory
        y.append(solver.process(np.array([v]))[0])
        states.append(solver.state.copy())
    return np.array(y), np.array(states)


def run(x, divisor, chunk=None):
    scan = CircuitScan(divisor=divisor)
    dry = torch.tensor(np.stack([x, x]), dtype=torch.float64)
    pots = torch.tensor([[getattr(c, k) for k in POTS] for c in CONTROLS], dtype=torch.float64)
    chunk = chunk or dry.shape[1]
    state, wet, states = None, [], []
    with torch.no_grad():
        for o in range(0, dry.shape[1], chunk):
            y, st, state = scan(dry[:, o:o+chunk], pots, state)
            wet.append(y); states.append(st)
    return torch.cat(wet, 1).numpy(), torch.cat(states, 1).numpy()


def esr(a, b):
    return float(np.sum((a-b)**2)/np.sum(b**2))


def test_divisor_one_matches_implicit_solver():
    x = burst()[:9600]
    wet, states = run(x, 1)
    for lane, controls in enumerate(CONTROLS):
        y, st = reference(x, controls)
        assert esr(wet[lane], y) < 1e-4
        # The detector sees the conductance one control step late: a few percent
        # during the first milliseconds of a fast attack, then sub-percent.
        err = np.abs(states[lane, :, 0]-st[:, 0])/max(st[:, 0].max(), 1e-3)
        assert err.max() < .05 and np.median(err) < .005


@pytest.mark.parametrize('divisor', [4, 16])
def test_decimated_control_path_stays_close(divisor):
    x = burst()
    wet, _ = run(x, divisor)
    for lane, controls in enumerate(CONTROLS):
        y, _ = reference(x, controls)
        assert esr(wet[lane], y) < 1e-2


def test_chunked_streaming_equals_one_shot():
    x = burst(.1)[:4096]
    whole, whole_states = run(x, 16)
    parts, part_states = run(x, 16, chunk=1024)
    np.testing.assert_allclose(parts, whole, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(part_states, whole_states, rtol=1e-12, atol=1e-14)


def test_implicit_gradient_matches_finite_difference():
    # Slower attack, so C3 tracks the detector and every learnable parameter matters.
    x = torch.tensor(burst(.1)[:4096][None], dtype=torch.float64)
    pots = torch.tensor([[getattr(CONTROLS[1], k) for k in POTS]], dtype=torch.float64)
    scan = CircuitScan(divisor=16)

    def loss():
        y, st, _ = scan(x, pots)
        return y.square().mean()+st[..., 0].mean()
    value = loss()
    value.backward()
    for name in ('pot_taper', 'gre_half', 'rectifier_gain'):
        raw = scan.raw[name]
        grad, h = raw.grad.item(), 1e-6
        with torch.no_grad():
            raw += h; up = loss().item(); raw -= 2*h; down = loss().item(); raw += h
        assert np.isfinite(grad) and grad != 0
        assert grad == pytest.approx((up-down)/(2*h), rel=1e-3, abs=1e-9)
