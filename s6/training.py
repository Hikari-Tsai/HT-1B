"""Single-device chronological TBPTT with PhysicsNeMo model checkpoints.

Only train() creates an optimizer or changes weights. check() and evaluation
are inference-only. Reference implementation: one recording stream at a time.
"""
import json
import math
from pathlib import Path
import random
import tempfile
import shutil
import time

import torch
from torch.nn import functional as F

from .data import load_records, iter_chunks, data_signature
from .layers import detach_state
from .losses import AudioLoss
from .models import AudioModel


MODEL_DEFAULTS = dict(architecture='s4_tfilm', width=16, state_dim=32, blocks=8,
                      film_block=128, context=64, spectrum_dim=8, conv_kernel=4, expansion=2)
TRAIN_DEFAULTS = dict(sample_rate=48000, chunk_samples=1024, warmup_samples=49152,
                      epochs=50, learning_rate=.001, weight_decay=0., grad_clip=1.,
                      seed=7, fft_sizes=[256, 512, 1024], l1_weight=1., esr_weight=.1,
                      spectral_weight=.1, log_every_chunks=100)


def load_config(path):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or set(document)-{'model', 'training'}:
        raise ValueError('Config accepts only model and training sections')
    result = {}
    for section, defaults in [('model', MODEL_DEFAULTS), ('training', TRAIN_DEFAULTS)]:
        values = document.get(section, {})
        if not isinstance(values, dict) or set(values)-set(defaults):
            raise ValueError(f'Unknown {section} config fields')
        result[section] = {**defaults, **values}
    m, t = result['model'], result['training']
    if m['architecture'] == 's6_tfilm' and 'blocks' not in document.get('model', {}):
        m['blocks'] = 2
    for name in set(m)-{'architecture'}:
        _positive_integer(m[name], f'model.{name}')
    if m['architecture'] not in ('s4_tfilm', 's6_tfilm') or (m['architecture'] == 's6_tfilm' and m['blocks'] != 2):
        raise ValueError('Choose s4_tfilm or s6_tfilm (exactly two blocks for S6)')
    for name in ('sample_rate', 'chunk_samples', 'epochs', 'log_every_chunks'):
        _positive_integer(t[name], name)
    for name in ('warmup_samples', 'seed'):
        if isinstance(t[name], bool) or not isinstance(t[name], int) or t[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if t['warmup_samples'] % t['chunk_samples']:
        raise ValueError('warmup_samples must be a multiple of chunk_samples')
    if m['architecture'] == 's4_tfilm' and t['chunk_samples'] % m['film_block']:
        raise ValueError('chunk_samples must be a multiple of film_block')
    for name in ('learning_rate', 'weight_decay', 'grad_clip', 'l1_weight', 'esr_weight', 'spectral_weight'):
        value = t[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if t['learning_rate'] == 0 or t['grad_clip'] == 0 or sum(t[k] for k in ('l1_weight', 'esr_weight', 'spectral_weight')) == 0:
        raise ValueError('learning_rate, grad_clip and total loss weight must be positive')
    if not isinstance(t['fft_sizes'], list) or not t['fft_sizes']:
        raise ValueError('fft_sizes must be a nonempty list')
    for size in t['fft_sizes']:
        _positive_integer(size, 'fft size')
        if size < 4:
            raise ValueError('fft sizes must be >= 4')
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def resolve_device(device):
    if device not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda; MPS/AMP are not supported by this reference implementation')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device(device)


def validate_records(records, options, require_train=True):
    splits = {r.split for r in records}
    if require_train and not {'train', 'validation'} <= splits:
        raise ValueError('Training/check requires both train and validation records; prepare a neural manifest first')
    for r in records:
        if r.stop-r.start <= options['warmup_samples']:
            raise ValueError(f'{r.id}: no scored samples remain after warmup_samples')


def _inputs(chunk, record, model, device):
    x = torch.from_numpy(chunk.dry).to(device)[None]
    y = torch.from_numpy(chunk.wet).to(device)[None]
    # Pad ONLY a final partial TFiLM block. Padded targets never enter the loss.
    padding = (-x.shape[1]) % model.required_multiple
    x = F.pad(x, (0, padding))
    controls = torch.tensor(record.controls, device=device, dtype=torch.float32)[None]
    gain = torch.tensor([record.gain_db], device=device, dtype=torch.float32)
    return x, y, controls, gain


def check(manifest, config_path, device='cpu'):
    """Read metadata + one chunk per split; no directory, optimizer, or checkpoint."""
    config = load_config(config_path)
    records = load_records(manifest, config['training']['sample_rate'])
    validate_records(records, config['training'])
    device = resolve_device(device)
    torch.manual_seed(config['training']['seed'])
    model = AudioModel(**config['model']).to(device).eval()
    shapes = {}
    with torch.inference_mode():
        for split in sorted({r.split for r in records}):
            r = next(r for r in records if r.split == split)
            chunk = next(iter_chunks(r, config['training']['chunk_samples']))
            x, target, controls, gain = _inputs(chunk, r, model, device)
            predicted, _ = model(x, controls, gain_db=gain)
            if not torch.isfinite(predicted).all():
                raise ValueError('Untrained forward produced nonfinite output')
            shapes[split] = dict(input=list(x.shape), target=list(target.shape), output=list(predicted.shape))
    return dict(status='check only: no optimizer, no training, no checkpoints',
                architecture=model.architecture, parameters=sum(p.numel() for p in model.parameters()),
                records=len(records), shapes=shapes, sample_rate=config['training']['sample_rate'],
                note='Checks metadata and first chunks only; does not measure model accuracy.')


@torch.no_grad()
def evaluate_records(model, records, *, chunk_samples, warmup_samples, device='cpu'):
    """Sample-weighted waveform metrics; states reset between records, never chunks."""
    if not records:
        raise ValueError('No recordings match selected split')
    model.eval()
    total_error = total_abs = total_energy = 0.
    count = 0
    per_record = {}
    for record in records:
        state = None
        square = absolute = energy = 0.
        samples = 0
        for chunk in iter_chunks(record, chunk_samples):
            x, target, controls, gain = _inputs(chunk, record, model, device)
            prediction, state = model(x, controls, state, gain)
            first = max(0, warmup_samples-chunk.offset)
            if first >= target.shape[1]:
                continue
            prediction, target = prediction[:, first:target.shape[1]], target[:, first:]
            if not torch.isfinite(prediction).all():
                raise ValueError(f'{record.id}: nonfinite model output')
            error = (prediction-target).double()
            square += error.square().sum().item()
            absolute += error.abs().sum().item()
            energy += target.double().square().sum().item()
            samples += target.numel()
        if not samples:
            raise ValueError(f'{record.id}: no samples after warmup')
        per_record[record.id] = dict(samples=samples, mse=square/samples,
                                     mae=absolute/samples, esr=square/max(energy, samples*1e-8))
        total_error += square
        total_abs += absolute
        total_energy += energy
        count += samples
    return dict(samples=count, mse=total_error/count, mae=total_abs/count,
                esr=total_error/max(total_energy, count*1e-8), records=per_record)


def _json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_epoch(output, model, optimizer, *, epoch, config, signature, best_esr, report):
    """Commit model and optimizer together; pointers update only after complete save."""
    root = Path(output)/'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    destination = root/f'epoch-{epoch:06d}'
    if destination.exists():
        raise ValueError(f'Checkpoint already exists: {destination}')
    temporary = Path(tempfile.mkdtemp(prefix='.saving-', dir=root))
    try:
        model.save(temporary/'model.mdlus')
        torch.save(dict(format_version=1, epoch=epoch, config=config, data_signature=signature,
                        optimizer=optimizer.state_dict(), best_esr=best_esr,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                        report=report), temporary/'training.pt')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _resume_config(config):
    # Increasing the TOTAL epoch target is the only allowed configuration change.
    return dict(model=config['model'], training={k: v for k, v in config['training'].items() if k != 'epochs'})


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    return value


def load_resume(output, resume, config, signature):
    """Read and validate an epoch bundle before creating an optimizer."""
    output, resume = Path(output).resolve(), Path(resume).resolve()
    if resume.parent != output/'checkpoints':
        raise ValueError('Resume must be an epoch directory inside this output/checkpoints')
    pointer = json.loads((output/'latest.json').read_text())
    if (output/pointer['checkpoint']).resolve() != resume:
        raise ValueError('Resume the latest completed epoch, or use a separate experiment')
    restored = torch.load(resume/'training.pt', map_location='cpu', weights_only=True)
    if restored.get('format_version') != 1 or restored['data_signature'] != signature:
        raise ValueError('Resume data signature mismatch')
    if _resume_config(restored['config']) != _resume_config(config):
        raise ValueError('Resume config mismatch; only total epochs may change')
    return restored


def train(manifest, config_path, output, *, device='cpu', resume=None):
    """Explicit opt-in training. Resume is a completed epoch DIRECTORY, not .mdlus."""
    config = load_config(config_path)
    options = config['training']
    records = load_records(manifest, options['sample_rate'])
    validate_records(records, options)
    device = resolve_device(device)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('Output is nonempty; use a new directory or resume its latest epoch')
    signature = data_signature(manifest, records)
    torch.manual_seed(options['seed'])
    start_epoch, best = 0, math.inf
    restored = None
    if resume is not None:
        resume = Path(resume).resolve()
        restored = load_resume(output, resume, config, signature)
        start_epoch, best = restored['epoch'], restored['best_esr']
        model = AudioModel.from_checkpoint(resume/'model.mdlus').to(device)
    else:
        model = AudioModel(**config['model']).to(device)
    if options['epochs'] <= start_epoch:
        raise ValueError('epochs must exceed the last completed epoch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'],
                                  weight_decay=options['weight_decay'])
    if restored is not None:
        optimizer.load_state_dict(restored['optimizer'])
        for key, value in optimizer.state.items():
            optimizer.state[key] = _move(value, device)
        torch.set_rng_state(restored['torch_rng'])
        if device.type == 'cuda' and restored['cuda_rng']:
            torch.cuda.set_rng_state_all(restored['cuda_rng'])
    criterion = AudioLoss(options['fft_sizes'], options['l1_weight'], options['esr_weight'],
                          options['spectral_weight'])
    output.mkdir(parents=True, exist_ok=True)
    _json_atomic(output/'config.json', config)
    _json_atomic(output/'manifest.json', json.loads(Path(manifest).read_text()))
    train_records = [r for r in records if r.split == 'train']
    validation_records = [r for r in records if r.split == 'validation']
    for epoch in range(start_epoch+1, options['epochs']+1):
        begun = time.monotonic()
        model.train()
        ordered = train_records.copy()
        random.Random(options['seed']+epoch).shuffle(ordered)  # shuffle records, NEVER their chunks
        sums = dict(loss=0., l1=0., esr=0., spectral=0.)
        samples = updates = 0
        for record in ordered:
            state = None
            for chunk in iter_chunks(record, options['chunk_samples']):
                x, target, controls, gain = _inputs(chunk, record, model, device)
                if chunk.offset < options['warmup_samples']:
                    with torch.no_grad():
                        _, state = model(x, controls, state, gain)
                    continue
                optimizer.zero_grad(set_to_none=True)
                prediction, state = model(x, controls, state, gain)
                loss, parts = criterion(prediction[:, :target.shape[1]], target)
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss at {record.id}, sample {record.start+chunk.offset}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                state = detach_state(state)  # TBPTT preserves values, cuts only gradient history
                n = target.numel()
                samples += n
                updates += 1
                sums['loss'] += loss.item()*n
                for key, value in parts.items():
                    sums[key] += value.item()*n
                if updates % options['log_every_chunks'] == 0:
                    print(json.dumps(dict(epoch=epoch, updates=updates, loss=sums['loss']/samples)), flush=True)
        validation = evaluate_records(model, validation_records, chunk_samples=options['chunk_samples'],
                                      warmup_samples=options['warmup_samples'], device=device)
        improved = validation['esr'] < best
        best = min(best, validation['esr'])
        report = dict(epoch=epoch, updates=updates, train={k: v/samples for k, v in sums.items()},
                      validation=validation, best_validation_esr=best, seconds=time.monotonic()-begun)
        checkpoint = save_epoch(output, model, optimizer, epoch=epoch, config=config,
                                signature=signature, best_esr=best, report=report)
        pointer = dict(epoch=epoch, checkpoint=str(checkpoint.relative_to(output)), validation_esr=validation['esr'])
        _json_atomic(output/'latest.json', pointer)
        if improved:
            _json_atomic(output/'best.json', pointer)
        _json_atomic(output/'report.json', report)
        with (output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(report, allow_nan=False)+'\n')
        print(json.dumps(report), flush=True)
    return report


def evaluate_checkpoint(manifest, checkpoint, *, split='validation', device='cpu'):
    checkpoint = Path(checkpoint)
    metadata = torch.load(checkpoint/'training.pt', map_location='cpu', weights_only=True)
    options = metadata['config']['training']
    records = [r for r in load_records(manifest, options['sample_rate']) if r.split == split]
    validate_records(records, options, require_train=False)
    device = resolve_device(device)
    model = AudioModel.from_checkpoint(checkpoint/'model.mdlus').to(device)
    return evaluate_records(model, records, chunk_samples=options['chunk_samples'],
                            warmup_samples=options['warmup_samples'], device=device)
