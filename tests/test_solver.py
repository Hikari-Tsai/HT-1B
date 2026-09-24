"""Numerical behavior: an incorrect state update or topology must fail these."""
import importlib.util
import numpy as np
import pytest


def test_solver_package_exists():
    assert importlib.util.find_spec('ht1b') is not None, 'Implement the HT-1B circuit package'


def components():
    from ht1b.config import Circuit, Controls
    from ht1b.solver import CircuitSolver
    return Circuit, Controls, CircuitSolver


def test_silence_and_block_boundaries():
    Circuit, Controls, Solver = components()
    x = np.r_[np.zeros(10), .4*np.sin(np.arange(160)*.17), np.zeros(30)]
    full = Solver(Circuit(), Controls(), 8000)
    expected = full.process(x)
    chunked = Solver(Circuit(), Controls(), 8000)
    actual = np.r_[chunked.process(x[:77]), chunked.process(x[77:])]
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    np.testing.assert_array_equal(Solver(Circuit(), Controls(), 8000).process(np.zeros(30)), 0)
    assert np.isfinite(expected).all()
    assert full.max_residual < 1e-7


def test_optical_release_matches_analytic_backward_euler():
    Circuit, Controls, Solver = components()
    p = Circuit(gre_release_fast=.05, gre_release_slow=.5)
    s = Solver(p, Controls(mode='fixed'), 1000, initial_state=[0, .8, .4])
    s.process(np.zeros(100))
    # With zero drive the target is exactly zero: scalar RC decay.
    np.testing.assert_allclose(s.state[1:], [.8*(1+.001/.05)**-100, .4*(1+.001/.5)**-100], rtol=2e-6)


def test_passive_front_network_and_switch_modes():
    from ht1b.equations import evaluate, NumpyOps
    Circuit, Controls, _ = components()
    p = Circuit()
    for mode in ['fixed', 'manual', 'fix-man']:
        a = evaluate(np.array([.2,.1,.3]), 1., p, Controls(mode=mode), NumpyOps)
        assert 0 <= a['front_b'] <= a['front_a'] <= 1
        assert a['front_kcl_a'] == pytest.approx(0, abs=1e-15)
        assert a['front_kcl_b'] == pytest.approx(0, abs=1e-15)
    c = Controls(mode='fix-man')
    a = evaluate(np.array([.2,.1,.3]), 1., p, c, NumpyOps)
    assert a['control'] == max(a['detector'], .2)


def test_refining_timestep_converges():
    Circuit, Controls, Solver = components()
    x = np.full(120, .3)
    results = [Solver(Circuit(), Controls(), 4000, substeps=n).process(x) for n in [1,2,4]]
    assert np.linalg.norm(results[2]-results[1]) < np.linalg.norm(results[1]-results[0])


def test_reject_bad_inputs():
    Circuit, Controls, Solver = components()
    for kw in [{'mode':'typo'}, {'attack':-1}, {'ratio':2}]:
        with pytest.raises(ValueError): Controls(**kw)
    with pytest.raises(ValueError): Circuit(c3=0)
    s = Solver(Circuit(), Controls(), 8000)
    for x in [np.array([np.nan]), np.ones((4,2))]:
        with pytest.raises(ValueError): s.process(x)


@pytest.mark.parametrize('mode',['fixed','manual','fix-man'])
def test_extreme_pots_are_finite_and_recover(mode):
    Circuit, Controls, Solver=components()
    for attack,release in [(0.,0.),(1.,1.)]:
        s=Solver(Circuit(),Controls(threshold=1,ratio=1,attack=attack,release=release,mode=mode),8000)
        loud=s.process(np.full(80,1.))
        before=s.state.copy()
        tail=s.process(np.zeros(80))
        assert np.isfinite(loud).all() and np.isfinite(tail).all()
        assert np.all(s.state>=0) and np.all(s.state[1:]<=1)
        # Stored GRE states may continue charging while C3 releases, so test the
        # externally observable zero-input behavior instead of assuming monotonic states.
        np.testing.assert_array_equal(tail,0)
        assert before[1]>0
