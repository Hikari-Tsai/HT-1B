"""Causality and recurrent-history checks for the GRU trajectory model."""
import torch

from pinn.train_gru_pinn import CausalGRUTrajectory, FRAME


def _model():
    torch.manual_seed(11)
    return CausalGRUTrajectory().to(dtype=torch.float64)


def test_future_audio_cannot_change_earlier_prediction():
    model = _model()
    controls = torch.zeros((1, 5), dtype=torch.float64)
    first = torch.zeros(FRAME * 2, dtype=torch.float64)
    second = first.clone()
    second[100:] = .7
    a, _ = model(first, controls, 0, 220., None)
    b, _ = model(second, controls, 0, 220., None)
    torch.testing.assert_close(a[:100], b[:100], rtol=0, atol=0)


def test_frame_aligned_chunks_preserve_gru_history():
    model = _model()
    controls = torch.tensor([[.2, .8, .5, .7, 0.]], dtype=torch.float64)
    dry = torch.linspace(-.5, .5, FRAME * 3, dtype=torch.float64)
    whole, _ = model(dry, controls, 0, 220., None)
    first, hidden = model(dry[:FRAME], controls, 0, 220., None)
    second, _ = model(dry[FRAME:], controls, FRAME, 220., hidden)
    torch.testing.assert_close(whole, torch.cat((first, second)), rtol=1e-12, atol=1e-12)
