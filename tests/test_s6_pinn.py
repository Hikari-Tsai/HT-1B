"""Check shared physics, conditioning, causal history and loss connectivity."""
from types import SimpleNamespace
import torch
from ht1b.config import Circuit, Controls
from ht1b.physicsnemo_model import (CircuitPDE, LearnedCircuit, TorchOps,
                                   circuit_residual_terms, residuals, waveform)
from s6.pinn_model import S6PINN


def test_shared_batched_terms_match_symbolic_pinn():
    torch.manual_seed(19)
    p = LearnedCircuit(Circuit()).values()
    states = torch.rand(2, 6, 3, dtype=torch.float64)
    deriv = torch.randn_like(states)
    dry = torch.randn(2, 6, dtype=torch.float64)
    settings = [Controls(.1,.4,.2,.8,0), Controls(.9,.7,.6,.1,3)]
    pots = SimpleNamespace(**{k:torch.tensor([[getattr(c,k)] for c in settings], dtype=torch.float64)
        for k in ('threshold','ratio','attack','release','makeup_db')}, mode='manual')
    actual, predicted = circuit_residual_terms(tuple(states.unbind(-1)), tuple(deriv.unbind(-1)), dry, p, pots, TorchOps)
    for i,c in enumerate(settings):
        expected = residuals(CircuitPDE(Circuit(),c), states[i], deriv[i], dry[i,:,None], p)
        for k in expected:
            torch.testing.assert_close(actual[k][i], expected[k].squeeze(-1), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(predicted[i], waveform(states[i],dry[i,:,None],p,c).squeeze(-1))


def test_causality_chunk_history_and_gradients():
    torch.manual_seed(7)
    model = S6PINN(architecture='s6_tfilm',width=4,state_dim=3,blocks=2,context=8,
                   spectrum_dim=3,conv_kernel=3,expansion=1)
    torch.nn.init.normal_(model.output.weight, std=.001)
    dry = torch.randn(2,64)*.1
    controls = torch.rand(2,4)
    a, _ = model.trajectory(dry, controls)
    changed = dry.clone(); changed[:,32:] += .4
    b, _ = model.trajectory(changed, controls)
    torch.testing.assert_close(a[:,:32], b[:,:32], rtol=0,atol=0)
    first, hidden = model.trajectory(dry[:,:32], controls)
    last, _ = model.trajectory(dry[:,32:], controls, hidden)
    torch.testing.assert_close(a, torch.cat((first,last),dim=1),rtol=1e-5,atol=1e-7)
    pots = torch.tensor([[.2,.3,.4,.5,0],[.8,.7,.6,.1,0]])
    wet, residual = model.predict_window(dry,controls,pots,warmup=32,chunk=16)
    assert wet.shape == (2,32)
    assert all(torch.isfinite(v).all() for v in residual.values())
    assert ((a[...,0] >= 0) & (a[...,0] <= 30)).all()
    assert ((a[...,1:] >= 0) & (a[...,1:] <= 1)).all()
    loss = wet.square().mean()+sum(v.square().mean() for v in residual.values())+model.prior_loss()
    loss.backward()
    for param in (model.input.weight, model.output.weight, model.physical.raw['gre_span']):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().max() > 0
