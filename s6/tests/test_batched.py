"""Equivalence checks for the multi-lane trainer and the fused selective scan.

Small synthetic models on CPU; the fused-scan parity tests need CUDA + Triton.
"""
import json

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip('torch')
pytest.importorskip('physicsnemo')

RATE = 8000
MODEL = dict(width=4, state_dim=3, blocks=2, film_block=4, context=8,
             spectrum_dim=3, conv_kernel=3)


def wav(tmp_path, frames=420, seed=0):
    rng = np.random.default_rng(seed)
    dry = (rng.standard_normal(frames)*.2).astype(np.float32)
    wet = np.tanh(2*dry)*.6
    path = tmp_path / 'TubeTech_a_1_r_3_r_6_t_-20_g_0.wav'
    sf.write(path, np.column_stack((dry, wet)), RATE, subtype='FLOAT')
    return path


def write_manifest(tmp_path, path, spans):
    rows = [dict(id=f'{split}-{i}', stereo_pair=path.name, split=split, start_sample=lo,
                 stop_sample=hi, delay_samples=0) for i, (split, lo, hi) in enumerate(spans)]
    dest = tmp_path / 'manifest.json'
    dest.write_text(json.dumps(dict(schema_version=1, records=rows)))
    return dest


def write_config(tmp_path, architecture, name='config.json', **training):
    dest = tmp_path / name
    dest.write_text(json.dumps(dict(
        model=dict(architecture=architecture, **MODEL),
        training=dict(sample_rate=RATE, chunk_samples=16, warmup_samples=16, epochs=1,
                      fft_sizes=[8, 16], log_every_chunks=1000, **training))))
    return dest


def small_model(architecture):
    from s6.models import AudioModel
    return AudioModel(architecture=architecture, **MODEL)


# Two train records (one ends in a 4-sample partial chunk) and one validation record.
SPANS = [('train', 0, 100), ('train', 110, 190), ('validation', 200, 300)]


@pytest.mark.parametrize('architecture', ['s4_tfilm', 's6_tfilm'])
def test_batch_size_one_reproduces_reference_training(tmp_path, architecture):
    from s6.models import AudioModel
    from s6.training import train
    from s6.batched import train_batched
    m = write_manifest(tmp_path, wav(tmp_path), SPANS)
    reference = train(m, write_config(tmp_path, architecture), tmp_path/'reference')
    batched = train_batched(m, write_config(tmp_path, architecture, 'batched.json', batch_size=1,
                                            eval_batch_size=2), tmp_path/'batched')
    assert batched['updates'] == reference['updates']
    assert batched['validation']['esr'] == pytest.approx(reference['validation']['esr'], rel=1e-5)
    for key in ('loss', 'l1', 'esr', 'spectral'):
        assert batched['train'][key] == pytest.approx(reference['train'][key], rel=1e-5)
    a = AudioModel.from_checkpoint(tmp_path/'reference/checkpoints/epoch-000001/model.mdlus')
    b = AudioModel.from_checkpoint(tmp_path/'batched/checkpoints/epoch-000001/model.mdlus')
    for (name, p), q in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(q, p, atol=1e-6, rtol=1e-5, msg=name)


@pytest.mark.parametrize('architecture', ['s4_tfilm', 's6_tfilm'])
def test_batched_validation_matches_streaming_reference(tmp_path, architecture):
    from s6.data import load_records
    from s6.training import evaluate_records
    from s6.batched import evaluate_batched
    spans = [('validation', 0, 100), ('validation', 110, 150), ('validation', 160, 300),
             ('validation', 310, 337)]
    records = load_records(write_manifest(tmp_path, wav(tmp_path), spans), RATE)
    torch.manual_seed(5)
    model = small_model(architecture).eval()
    expected = evaluate_records(model, records, chunk_samples=16, warmup_samples=16)
    actual = evaluate_batched(model, records, chunk_samples=16, warmup_samples=16, lanes=3,
                              block_chunks=2)
    assert actual['samples'] == expected['samples']
    for key in ('mse', 'mae', 'esr'):
        assert actual[key] == pytest.approx(expected[key], rel=1e-5)
    for name, row in expected['records'].items():
        assert actual['records'][name]['samples'] == row['samples']
        assert actual['records'][name]['esr'] == pytest.approx(row['esr'], rel=1e-5)


@pytest.mark.parametrize('architecture', ['s4_tfilm', 's6_tfilm'])
def test_mask_state_resets_only_selected_lanes(architecture):
    from s6.batched import mask_state
    torch.manual_seed(2)
    model = small_model(architecture).eval()
    x = torch.randn(2, 32)*.2
    p = torch.tensor([[.5, .5, 0., 1.], [.2, .7, .6, .3]])
    with torch.no_grad():
        _, state = model(x[:, :16], p)
        state = mask_state(state, torch.tensor([False, True]))
        both, _ = model(x[:, 16:], p, state)
        fresh, _ = model(x[:1, 16:], p[:1])
        _, carried = model(x[1:, :16], p[1:])
        continued, _ = model(x[1:, 16:], p[1:], carried)
    torch.testing.assert_close(both[:1], fresh, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(both[1:], continued, atol=1e-6, rtol=1e-5)


def test_scored_loss_is_mean_of_per_chunk_losses():
    from s6.losses import AudioLoss
    from s6.batched import scored_loss
    criterion = AudioLoss(fft_sizes=(8, 16))
    torch.manual_seed(0)
    prediction, target = torch.randn(4, 16), torch.randn(4, 16)
    valid = np.array([16, 16, 5, 16])
    rows = [0, 2, 3]
    loss, sums, samples = scored_loss(criterion, prediction, target, rows, valid, 16)
    per_row = [criterion(prediction[i:i+1, :valid[i]], target[i:i+1, :valid[i]])[0] for i in rows]
    torch.testing.assert_close(loss, torch.stack(per_row).mean())
    assert samples == 37
    assert sums['loss'] == pytest.approx(sum(l.item()*valid[i] for l, i in zip(per_row, rows)), rel=1e-6)


def test_batched_training_resumes_and_rejects_changed_lane_count(tmp_path):
    from s6.batched import train_batched
    m = write_manifest(tmp_path, wav(tmp_path), SPANS)
    cfg = write_config(tmp_path, 's6_tfilm', 'b.json', batch_size=2, eval_batch_size=2)
    first = train_batched(m, cfg, tmp_path/'run')
    assert first['epoch'] == 1 and first['batch_size'] == 2
    doc = json.loads(cfg.read_text())
    doc['training']['epochs'] = 2
    cfg.write_text(json.dumps(doc))
    second = train_batched(m, cfg, tmp_path/'run', resume=tmp_path/'run/checkpoints/epoch-000001')
    assert second['epoch'] == 2
    doc['training'].update(epochs=3, batch_size=3)
    cfg.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='config'):
        train_batched(m, cfg, tmp_path/'run', resume=tmp_path/'run/checkpoints/epoch-000002')


class _NanGrad(torch.autograd.Function):
    """Identity forward, NaN backward: a finite loss whose gradients are not."""
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad*float('nan')


def poison_updates(monkeypatch, calls):
    """Make the scored_loss calls numbered in `calls` (1-based) produce NaN gradients."""
    import s6.batched as batched
    original, count = batched.scored_loss, [0]

    def scored_loss(*args):
        loss, sums, samples = original(*args)
        count[0] += 1
        return (_NanGrad.apply(loss) if count[0] in calls else loss), sums, samples
    monkeypatch.setattr(batched, 'scored_loss', scored_loss)


def test_nonfinite_gradient_skips_update_and_continues(tmp_path, monkeypatch):
    from s6.models import AudioModel
    from s6.batched import train_batched
    m = write_manifest(tmp_path, wav(tmp_path), SPANS)
    cfg = write_config(tmp_path, 's6_tfilm', 'b.json', batch_size=2, eval_batch_size=2)
    clean = train_batched(m, cfg, tmp_path/'clean')
    poison_updates(monkeypatch, {2})
    report = train_batched(m, cfg, tmp_path/'run')
    assert report['skipped_updates'] == 1
    assert report['skipped'][0]['after_update'] == 1 and report['skipped'][0]['reset_lanes'] == 0
    assert report['updates'] == clean['updates'] - 1
    model = AudioModel.from_checkpoint(tmp_path/'run/checkpoints/epoch-000001/model.mdlus')
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert np.isfinite(report['validation']['esr'])


def test_consecutive_nonfinite_updates_stop_training(tmp_path, monkeypatch):
    import s6.batched as batched
    m = write_manifest(tmp_path, wav(tmp_path), SPANS)
    cfg = write_config(tmp_path, 's6_tfilm', 'b.json', batch_size=2, eval_batch_size=2)
    monkeypatch.setattr(batched, 'MAX_CONSECUTIVE_SKIPS', 1)
    poison_updates(monkeypatch, {1, 2})
    with pytest.raises(ValueError, match='consecutive nonfinite'):
        batched.train_batched(m, cfg, tmp_path/'run')


def test_nonfinite_lanes_are_found_and_reset():
    from s6.batched import finite_lanes, mask_state
    lstm = torch.ones(2, 3, 4)        # [layers, batch, hidden]
    ssm = torch.ones(3, 5, dtype=torch.complex64)
    lstm[1, 2, 0] = float('inf')
    ssm[0, 3] = float('nan')
    state = [dict(h=lstm), (ssm, None)]
    finite = finite_lanes(state, 3, 'cpu')
    assert finite.tolist() == [False, True, False]
    lstm_out, ssm_out = mask_state(state, finite)[0]['h'], mask_state(state, finite)[1][0]
    assert torch.isfinite(lstm_out).all() and torch.isfinite(ssm_out).all()
    assert (lstm_out[:, 1] == 1).all() and (lstm_out[:, [0, 2]] == 0).all()
    assert (ssm_out[1] == 1).all() and (ssm_out[[0, 2]] == 0).all()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='fused scan needs CUDA')


@cuda
@pytest.mark.parametrize('shape', [(3, 37, 5, 3), (2, 1024, 16, 16), (1, 1, 2, 4)])
def test_fused_scan_matches_reference_forward_and_gradients(shape):
    pytest.importorskip('triton')
    from s6.layers import SelectiveSSM
    from s6.fused_scan import fused_forward
    batch, length, width, n = shape
    torch.manual_seed(11)
    layer = SelectiveSSM(width, n).cuda()
    with torch.no_grad():
        layer.skip.add_(torch.randn(width, device='cuda')*.1)
    u = (torch.randn(batch, length, width, device='cuda')*.5).requires_grad_()
    s0 = (torch.randn(batch, width, n, device='cuda')*.1).requires_grad_()
    weights = torch.randn(batch, length, width, device='cuda')
    final_weights = torch.randn(batch, width, n, device='cuda')

    def run(forward):
        layer.zero_grad()
        y, end = forward(layer, u, s0)
        ((y*weights).sum() + (end*final_weights).sum()).backward()
        grads = [u.grad.clone(), s0.grad.clone()] + [p.grad.clone() for p in layer.parameters()]
        u.grad = s0.grad = None
        return y.detach(), end.detach(), grads

    ref_y, ref_end, ref_grads = run(SelectiveSSM.forward)
    y, end, grads = run(fused_forward)
    torch.testing.assert_close(y, ref_y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(end, ref_end, atol=1e-5, rtol=1e-5)
    for name, g, r in zip(['u', 's0', 'log_a', 'bc', 'dt.weight', 'dt.bias', 'skip'], grads, ref_grads):
        torch.testing.assert_close(g, r, atol=2e-4, rtol=2e-4, msg=name)


@cuda
def test_accelerated_model_matches_reference_model():
    pytest.importorskip('triton')
    from s6.fused_scan import accelerate_training
    torch.manual_seed(4)
    reference = small_model('s6_tfilm').cuda()
    fused = small_model('s6_tfilm').cuda()
    fused.load_state_dict(reference.state_dict())
    accelerate_training(fused)
    x = torch.randn(4, 64, device='cuda')*.2
    p = torch.rand(4, 4, device='cuda')
    outputs = []
    for model in (reference, fused):
        y, state = model(x[:, :32], p)
        y2, _ = model(x[:, 32:], p, state)
        torch.cat((y, y2), 1).square().mean().backward()
        outputs.append((torch.cat((y, y2), 1).detach(), [q.grad.clone() for q in model.parameters()]))
    torch.testing.assert_close(outputs[1][0], outputs[0][0], atol=1e-5, rtol=1e-5)
    for g, r in zip(outputs[1][1], outputs[0][1]):
        torch.testing.assert_close(g, r, atol=2e-4, rtol=2e-4)
