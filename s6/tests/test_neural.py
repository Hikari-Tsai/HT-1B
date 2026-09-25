"""Contract checks only: no trainer runs or optimizer updates in this suite."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip('torch')
pytest.importorskip('physicsnemo')


def pair(tmp_path, frames=320, rate=8000):
    path = tmp_path / 'TubeTech_a_1_r_3_r_6_t_-20_g_0.wav'
    x = np.linspace(-.3, .3, frames, dtype=np.float32)
    sf.write(path, np.column_stack((x, .5*x)), rate, subtype='FLOAT')
    return path, x


def manifest(tmp_path, path, **overrides):
    row = dict(id='one', stereo_pair=path.name, split='train',
               start_sample=0, stop_sample=32, delay_samples=2)
    row.update(overrides)
    dest = tmp_path / 'manifest.json'
    dest.write_text(json.dumps(dict(schema_version=1, records=[row])))
    return dest


def test_stereo_order_alignment_and_raw_knobs(tmp_path):
    from s6.data import load_records, iter_chunks
    path, x = pair(tmp_path)
    record = load_records(manifest(tmp_path, path), sample_rate=8000)[0]
    chunk = next(iter_chunks(record, 16))
    np.testing.assert_array_equal(chunk.dry, x[:16])
    np.testing.assert_array_equal(chunk.wet, .5*x[2:18])
    np.testing.assert_allclose(record.controls, [.5, .5, .25, .75])


def test_temporal_split_has_guard_and_never_overlaps(tmp_path):
    from s6.data import prepare_manifest, load_records
    path, _ = pair(tmp_path)
    dest = tmp_path / 'split.json'
    prepare_manifest([path], dest, sample_rate=8000, validation_fraction=.25,
                     gap_samples=16)
    train, val = load_records(dest, 8000)
    assert (train.start, train.stop, val.start, val.stop) == (0, 240, 256, 320)
    with pytest.raises(ValueError, match='exists'):
        prepare_manifest([path], dest, sample_rate=8000)
    doc = json.loads(dest.read_text())
    doc['records'][1]['start_sample'] = 230
    dest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='overlap'):
        load_records(dest, 8000)


def test_reader_rejects_mono_rate_mismatch_and_nonfinite(tmp_path):
    from s6.data import load_records, iter_chunks
    path, _ = pair(tmp_path)
    m = manifest(tmp_path, path)
    with pytest.raises(ValueError, match='sample rate'):
        load_records(m, 48000)
    sf.write(path, np.zeros(40), 8000)
    with pytest.raises(ValueError, match='stereo'):
        load_records(m, 8000)
    bad = np.zeros((40, 2), np.float32)
    bad[3, 1] = np.nan
    sf.write(path, bad, 8000, subtype='FLOAT')
    with pytest.raises(ValueError, match='finite'):
        next(iter_chunks(load_records(m, 8000)[0], 16))


def small_model(architecture):
    from s6.models import AudioModel
    return AudioModel(architecture=architecture, width=4, state_dim=3,
                      blocks=2, film_block=4, context=8, spectrum_dim=3,
                      conv_kernel=3)


@pytest.mark.parametrize('architecture', ['s4_tfilm', 's6_tfilm'])
def test_state_carry_matches_whole_sequence_and_gradients(architecture):
    from s6.layers import detach_state
    torch.manual_seed(3)
    model = small_model(architecture)
    x = torch.randn(2, 24) * .1
    p = torch.tensor([[.5, .5, 0., 1.], [.2, .7, .6, .3]])
    whole, _ = model(x, p)
    a, state = model(x[:, :8], p)
    b, state = model(x[:, 8:16], p, state)
    c, _ = model(x[:, 16:], p, state)
    torch.testing.assert_close(torch.cat((a, b, c), 1), whole, atol=2e-6, rtol=2e-5)
    reset, _ = model(x[:, 8:16], p)
    assert not torch.allclose(reset, b, atol=1e-8, rtol=1e-6)
    detached = detach_state(state)
    def visit(value):
        if torch.is_tensor(value):
            assert value.grad_fn is None
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
    visit(detached)
    whole.square().mean().backward()  # gradient verification only; no optimizer
    grads = [v.grad for v in model.parameters() if v.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_s6_is_causal_and_gain_form_preserves_zero_input():
    torch.manual_seed(1)
    model = small_model('s6_tfilm')
    x = torch.randn(1, 20)
    p = torch.tensor([[.5, .5, 0., 1.]])
    y, _ = model(x, p)
    altered = x.clone()
    altered[:, 10:] += 10
    z, _ = model(altered, p)
    torch.testing.assert_close(y[:, :10], z[:, :10])
    zero, _ = model(torch.zeros_like(x), p)
    assert torch.count_nonzero(zero) == 0


def test_s4_tfilm_only_looks_within_current_block():
    model = small_model('s4_tfilm')
    x = torch.randn(1, 16)
    p = torch.zeros(1, 4)
    y, _ = model(x, p)
    changed = x.clone()
    changed[:, 8:] += 20
    z, _ = model(changed, p)
    torch.testing.assert_close(y[:, :8], z[:, :8], atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='multiple'):
        model(x[:, :7], p)


@pytest.mark.parametrize('architecture', ['s4_tfilm', 's6_tfilm'])
def test_physicsnemo_checkpoint_prediction_roundtrip(tmp_path, architecture):
    from s6.models import AudioModel
    model = small_model(architecture).eval()
    checkpoint = tmp_path / 'untrained.mdlus'
    model.save(checkpoint)
    loaded = AudioModel.from_checkpoint(checkpoint).eval()
    x, p = torch.randn(1, 16), torch.ones(1, 4)*.5
    torch.testing.assert_close(model(x, p)[0], loaded(x, p)[0])


def test_loss_rejects_shape_and_rewards_correct_waveform():
    from s6.losses import AudioLoss
    criterion = AudioLoss(fft_sizes=(8, 16))
    y = torch.randn(1, 32)*.1
    exact, _ = criterion(y, y)
    wrong, parts = criterion(y*.5, y)
    assert exact.item() < 1e-6
    assert wrong > exact
    assert parts['esr'].item() == pytest.approx(.25, abs=1e-5)
    silent, _ = criterion(torch.zeros_like(y), torch.zeros_like(y))
    assert torch.isfinite(silent)
    with pytest.raises(ValueError, match='shape'):
        criterion(y[:, :16], y)


def config(tmp_path, architecture):
    dest = tmp_path / 'config.json'
    dest.write_text(json.dumps(dict(
        model=dict(architecture=architecture, width=4, state_dim=3, blocks=2,
                   film_block=4, context=8, spectrum_dim=3, conv_kernel=3),
        training=dict(sample_rate=8000, chunk_samples=16, warmup_samples=16,
                      epochs=2, fft_sizes=[8, 16]))))
    return dest


@pytest.mark.parametrize('architecture', ['s4_tfilm', 's6_tfilm'])
def test_check_cli_never_creates_training_artifacts(tmp_path, architecture):
    from s6.data import prepare_manifest
    path, _ = pair(tmp_path)
    m = tmp_path / 'split.json'
    prepare_manifest([path], m, sample_rate=8000, gap_samples=16)
    cfg = config(tmp_path, architecture)
    before = {p.name for p in tmp_path.iterdir()}
    result = subprocess.run([sys.executable, '-m', 's6', 'check',
                             str(m), '--config', str(cfg)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'no optimizer' in result.stdout
    assert {p.name for p in tmp_path.iterdir()} == before


def test_validation_streaming_metrics_include_partial_tail(tmp_path):
    from s6.data import load_records
    from s6.training import evaluate_records
    path, x = pair(tmp_path, frames=43)
    m = manifest(tmp_path, path, stop_sample=43, delay_samples=0, split='validation')
    model = small_model('s4_tfilm').eval()
    records = load_records(m, 8000)
    report = evaluate_records(model, records, chunk_samples=16, warmup_samples=16,
                              device='cpu')
    dry = torch.from_numpy(np.pad(x, (0, 1)))[None]
    p = torch.tensor(records[0].controls)[None]
    with torch.no_grad():
        expected = model(dry, p)[0][0, 16:43].numpy()
    assert report['samples'] == 27
    assert report['mse'] == pytest.approx(float(np.mean((expected-.5*x[16:])**2)), rel=1e-4)


def test_config_rejects_invalid_chunk_before_any_training(tmp_path):
    from s6.training import load_config
    cfg = config(tmp_path, 's4_tfilm')
    doc = json.loads(cfg.read_text())
    doc['training']['chunk_samples'] = 7
    cfg.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='multiple'):
        load_config(cfg)


def test_s4_impulse_matches_known_exponential_response():
    # A=-1, dt=1, B=1, 2*Re(C)=1, D=0: y[k]=(1-exp(-1))*exp(-k).
    from s6.layers import DiagonalSSM
    layer = DiagonalSSM(1, 1)
    with torch.no_grad():
        layer.log_decay.zero_()
        layer.frequency.zero_()
        layer.log_dt.zero_()
        layer.readout.copy_(torch.tensor([[[.5, 0.]]]))
        layer.skip.zero_()
    impulse = torch.zeros(1, 4, 1)
    impulse[0, 0, 0] = 1
    output, _ = layer(impulse)
    torch.testing.assert_close(output.flatten(), torch.tensor([
        .6321205588, .2325441579, .0855482149, .0314714295]), atol=1e-7, rtol=1e-6)


def test_epoch_bundle_and_resume_guards_without_optimizer_updates(tmp_path):
    from s6.training import load_config, save_epoch, load_resume
    cfg = load_config(config(tmp_path, 's6_tfilm'))
    model = small_model('s6_tfilm')
    optimizer = torch.optim.AdamW(model.parameters())  # construct only; never step
    output = tmp_path / 'run'
    epoch = save_epoch(output, model, optimizer, epoch=1, config=cfg,
                       signature='audio-content-hash', best_esr=.4, report={'epoch': 1})
    (output/'latest.json').write_text(json.dumps({'checkpoint': str(epoch.relative_to(output))}))
    restored = load_resume(output, epoch, cfg, 'audio-content-hash')
    assert restored['epoch'] == 1 and restored['best_esr'] == .4
    assert (epoch/'model.mdlus').is_file()
    with pytest.raises(ValueError, match='signature'):
        load_resume(output, epoch, cfg, 'different-audio')
    changed = json.loads(json.dumps(cfg))
    changed['training']['epochs'] = 10
    assert load_resume(output, epoch, changed, 'audio-content-hash')['epoch'] == 1
    changed['training']['learning_rate'] = .5
    with pytest.raises(ValueError, match='config'):
        load_resume(output, epoch, changed, 'audio-content-hash')
    with pytest.raises(ValueError, match='exists'):
        save_epoch(output, model, optimizer, epoch=1, config=cfg,
                   signature='audio-content-hash', best_esr=.4, report={})
