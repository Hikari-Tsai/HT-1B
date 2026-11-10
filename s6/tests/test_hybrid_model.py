"""S6Circuit: corrections around the integrated circuit, streaming and batched training."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('physicsnemo')

from s6.tests.test_batched import MODEL, SPANS, wav, write_config, write_manifest  # noqa: E402

PARTS = (slice(0, 32), slice(32, 64))
CONTROLS = torch.tensor([[.5, .5, 0., 1.], [.2, .7, .6, .3]], dtype=torch.float64)
POTS = torch.tensor([[.9, .5, 0., .5, 0.], [.7, .2, .5, .25, 3.]], dtype=torch.float64)


def model(corrections='audio', scale=0.):
    from s6.hybrid_model import S6Circuit
    torch.manual_seed(3)
    m = S6Circuit(corrections, architecture='s6_tfilm', **MODEL).double().eval()
    with torch.no_grad():
        m.output.weight.normal_(std=scale)
    return m


def signal():
    torch.manual_seed(1)
    return torch.randn(2, 64, dtype=torch.float64)*.5


def test_zero_init_is_the_plain_circuit():
    m, x = model(), signal()
    with torch.no_grad():
        wet, residual, _ = m.stream(x, CONTROLS, POTS)
        plain, _, _ = m.physical(x, POTS)
    assert residual == {}
    torch.testing.assert_close(wet, plain)


def test_audio_corrections_ignore_knob_labels_but_knob_variant_does_not():
    x = signal()
    with torch.no_grad():
        for corrections, same in (('audio', True), ('knobs', False)):
            m = model(corrections, scale=.5)
            a, _, _ = m.stream(x, CONTROLS, POTS)
            b, _, _ = m.stream(x, CONTROLS.flip(0), POTS)
            assert torch.equal(a, b) == same


def test_stream_carries_network_and_circuit_state_across_chunks():
    m, x = model(scale=.5), signal()
    with torch.no_grad():
        whole, _, _ = m.stream(x, CONTROLS, POTS)
        a, _, state = m.stream(x[:, PARTS[0]], CONTROLS, POTS)
        b, _, _ = m.stream(x[:, PARTS[1]], CONTROLS, POTS, state)
    torch.testing.assert_close(torch.cat((a, b), 1), whole)


def test_checkpoint_round_trip(tmp_path):
    from s6.models import AudioModel
    m, x = model('knobs', scale=.5), signal()
    m.save(str(tmp_path/'m.mdlus'))
    loaded = AudioModel.from_checkpoint(str(tmp_path/'m.mdlus')).double().eval()
    assert type(loaded).__name__ == 'S6Circuit' and loaded.corrections == 'knobs'
    with torch.no_grad():
        torch.testing.assert_close(loaded.stream(x, CONTROLS, POTS)[0], m.stream(x, CONTROLS, POTS)[0])


@pytest.mark.parametrize('architecture', ['s6_tfilm_circuit', 's6_tfilm_circuit_none'])
def test_batched_training_learns_physical_parameters_with_prior(tmp_path, architecture):
    from s6.batched import train_batched
    from s6.models import AudioModel
    m = write_manifest(tmp_path, wav(tmp_path), SPANS)
    cfg = write_config(tmp_path, architecture, batch_size=2, eval_batch_size=2, selection='macro_esr', prior_weight=1e-2)
    doc = json.loads(cfg.read_text())
    doc['training']['epochs'] = 2
    cfg.write_text(json.dumps(doc))
    result = train_batched(m, cfg, tmp_path/'run')
    assert np.isfinite(result['validation']['macro_esr']) and np.isfinite(result['train']['prior'])
    assert 'physics_mse' not in result['validation']
    saved = AudioModel.from_checkpoint(tmp_path/'run/checkpoints/epoch-000002/model.mdlus')
    assert type(saved).__name__ == 'S6Circuit'
    initial = model('none').physical
    moved = [k for k in saved.physical.raw if not torch.isclose(saved.physical.raw[k].double(), initial.raw[k])]
    assert moved, 'physical parameters should train'
    state = torch.load(tmp_path/'run/checkpoints/epoch-000002/training.pt', weights_only=True)
    assert [g['lr'] for g in state['optimizer']['param_groups']] == [.001, 1e-4]
