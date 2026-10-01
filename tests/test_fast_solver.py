import pytest
pytest.importorskip('numba')
import numpy as np
from ht1b.config import Circuit,Controls
from ht1b.solver import CircuitSolver
from ht1b.fast_solver import FastCircuitSolver


@pytest.mark.parametrize('position',[0.,.5,1.])
@pytest.mark.parametrize('substeps',[1,2])
def test_compiled_solver_matches_reference_and_preserves_state(position,substeps):
    p=Circuit();c=Controls(threshold=position,ratio=position,attack=position,release=1-position)
    x=np.random.default_rng(19).normal(0,.2,700);x[200:400]=0
    initial=[.4,.2,.1]
    reference=CircuitSolver(p,c,48000,substeps=substeps,initial_state=initial)
    compiled=FastCircuitSolver(p,c,48000,substeps=substeps,initial_state=initial)
    expected=reference.process(x)
    actual=np.r_[compiled.process(x[:321]),compiled.process(x[321:])]
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(compiled.state,reference.state,rtol=1e-7,atol=1e-9)


def test_changed_physical_parameters_match_reference():
    p=Circuit(gre_span=.003,gre_half=.7,amp_gain=3.,gre_attack_fast=.02,gre_attack_slow=.005,
              gre_release_fast=.012,gre_release_slow=2.)
    c=Controls(threshold=.9,ratio=.8,attack=.02,release=.4)
    x=np.random.default_rng(7).normal(0,.7,2000);x[1000:]=0
    reference=CircuitSolver(p,c,48000)
    compiled=FastCircuitSolver(p,c,48000)
    np.testing.assert_allclose(compiled.process(x),reference.process(x),rtol=1e-7,atol=1e-9)
