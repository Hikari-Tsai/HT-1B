import numpy as np
import pytest
pytest.importorskip('physicsnemo')
import torch
from ht1b.config import Circuit, Controls


@pytest.mark.parametrize('mode',['fixed','manual','fix-man'])
def test_symbolic_equations_zero_on_numerical_rates_and_backpropagate(mode):
    from ht1b.physicsnemo_model import CircuitPDE, LearnedCircuit, residuals
    from ht1b.equations import evaluate, NumpyOps
    p,c=Circuit(),Controls(mode=mode)
    model=LearnedCircuit(p)
    states=torch.tensor([[.3,.01,.02],[.1,.3,.4],[0.,0.,0.]],dtype=torch.float64)
    x=torch.tensor([[.2],[.4],[0.]],dtype=torch.float64)
    rates=np.array([evaluate(s,u[0],p,c,NumpyOps)['rhs'] for s,u in zip(states.numpy(),x.numpy())])
    equations=CircuitPDE(p,c)
    result=residuals(equations,states,torch.tensor(rates),x,model.values())
    for v in result.values(): torch.testing.assert_close(v,torch.zeros_like(v),atol=1e-10,rtol=0)
    # Nonzero derivative mismatch excites time-constant gradients even when
    # the fixed detector is below threshold (zero drive cannot identify GRE gain).
    bad=residuals(equations,states,torch.tensor(rates)*.5,x,model.values())
    loss=sum((v*v).mean() for v in bad.values())
    loss.backward()
    assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in model.parameters())


def test_trajectory_hard_initial_state_and_bounds():
    from ht1b.physicsnemo_model import TrajectoryPINN
    net=TrajectoryPINN(2,width=16,layers=2).double()
    t=torch.tensor([[0.],[.01],[.02]],dtype=torch.float64)
    initial=torch.tensor([[.2,.3,.4]],dtype=torch.float64)
    s=net(t,0,.1,initial)
    torch.testing.assert_close(s[0],initial[0])
    assert (s>=0).all() and (s[:,1:]<=1).all()
