"""Multi-lane chronological TBPTT: `batch_size` record streams advance together.

Each lane streams one record exactly as training.train() does: chunks in time
order, state carried across chunks and reset only when the lane starts a new
record, the first warmup_samples of each record run without loss. Lanes take
records from the same per-epoch shuffled order. One optimizer step averages the
per-chunk losses of every scored lane, so batch_size=1 reproduces train()
update for update. Larger batches change the optimisation (fewer, averaged
updates per epoch), not the model, data split, loss or checkpoint format.

Validation streams `eval_batch_size` records at once with the same per-record,
sample-weighted metrics as training.evaluate_records.
"""
from collections import deque
import json
import math
from pathlib import Path
import queue
import random
import threading
import time

import numpy as np
import soundfile as sf
import torch

from .data import load_records, data_signature
from .layers import detach_state
from .losses import AudioLoss
from .models import AudioModel
from .training import (load_config, resolve_device, validate_records, save_epoch,
                       load_resume, _json_atomic, _move)


BATCH_DEFAULTS = dict(batch_size=1, eval_batch_size=256, read_block_chunks=32)


def load_batched_config(path):
    config = load_config(path, extra_training=BATCH_DEFAULTS)
    for name in BATCH_DEFAULTS:
        value = config['training'][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    return config


class _Lane:
    """Sequential buffered reader for one record (same samples as data.iter_chunks)."""

    def __init__(self, record, slot, block):
        self.record, self.slot, self.block = record, slot, block
        self.length = record.stop - record.start
        self.position = 0
        self.dry_file = sf.SoundFile(record.path)
        self.dry_file.seek(record.start)
        self.wet_file = None
        if record.delay:
            self.wet_file = sf.SoundFile(record.path)
            self.wet_file.seek(record.start + record.delay)
        self.buffer = None
        self.cursor = 0

    @property
    def done(self):
        return self.position >= self.length

    def _fill(self):
        count = min(self.block, self.length - self.position)
        pair = self.dry_file.read(count, dtype='float32', always_2d=True)
        dry = pair[:, 0]
        wet = (self.wet_file.read(count, dtype='float32', always_2d=True)[:, 1]
               if self.wet_file is not None else pair[:, 1])
        if len(dry) != count or len(wet) != count:
            raise ValueError(f'{self.record.id}: recording changed or ended early')
        if not (np.isfinite(dry).all() and np.isfinite(wet).all()):
            raise ValueError(f'{self.record.id}: audio must be finite')
        self.buffer, self.cursor = (dry, wet), 0

    def next(self, chunk):
        count = min(chunk, self.length - self.position)
        if self.buffer is None or self.cursor + count > len(self.buffer[0]):
            self._fill()
        dry, wet = self.buffer
        offset = self.position
        span = slice(self.cursor, self.cursor + count)
        self.cursor += count
        self.position += count
        return dry[span], wet[span], offset

    def close(self):
        self.dry_file.close()
        if self.wet_file is not None:
            self.wet_file.close()


def lane_batches(records, order, lanes, chunk, block_chunks):
    """Yield [lanes, chunk] steps until every record index in `order` is fully read.

    index: position of the lane's record in `records` (-1 = idle lane).
    offset: chunk start relative to the record start. valid: real samples.
    reset: the lane starts a new record (or is idle) at this step.
    """
    pending = deque(order)
    active = [None]*lanes
    block = chunk*block_chunks
    try:
        while True:
            reset = np.zeros(lanes, dtype=bool)
            for i in range(lanes):
                if active[i] is None or active[i].done:
                    if active[i] is not None:
                        active[i].close()
                    slot = pending.popleft() if pending else None
                    active[i] = None if slot is None else _Lane(records[slot], slot, block)
                    reset[i] = True
            if all(lane is None for lane in active):
                return
            batch = dict(dry=np.zeros((lanes, chunk), dtype=np.float32),
                         wet=np.zeros((lanes, chunk), dtype=np.float32),
                         valid=np.zeros(lanes, dtype=np.int64),
                         offset=np.zeros(lanes, dtype=np.int64),
                         index=np.full(lanes, -1, dtype=np.int64),
                         controls=np.zeros((lanes, 4), dtype=np.float32),
                         gain=np.zeros(lanes, dtype=np.float32), reset=reset)
            for i, lane in enumerate(active):
                if lane is None:
                    continue
                dry, wet, batch['offset'][i] = lane.next(chunk)
                batch['valid'][i] = len(dry)
                batch['dry'][i, :len(dry)] = dry
                batch['wet'][i, :len(wet)] = wet
                batch['index'][i] = lane.slot
                batch['controls'][i] = lane.record.controls
                batch['gain'][i] = lane.record.gain_db
            yield batch
    finally:
        for lane in active:
            if lane is not None:
                lane.close()


def prefetch(iterator, depth=8):
    """Read audio in a background thread while the GPU computes."""
    items = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def work():
        try:
            for item in iterator:
                if stop.is_set():
                    return
                items.put(item)
            items.put(done)
        except BaseException as error:  # re-raised in the consumer thread
            items.put(error)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while thread.is_alive():
            try:
                items.get_nowait()
            except queue.Empty:
                thread.join(.05)


def mask_state(state, keep):
    """Zero the state of lanes where keep==0 (a fresh record), keep the others.

    Zero equals the model's own initial state for every S4/S6 state tensor.
    Batch is dim 0, except LSTM/GRU states [layers, batch, hidden] (dim 1).
    """
    if state is None:
        return None
    if torch.is_tensor(state):
        batch = keep.shape[0]
        if state.shape[0] == batch:
            dim = 0
        elif state.ndim >= 2 and state.shape[1] == batch:
            dim = 1
        else:
            raise ValueError(f'Cannot locate batch dimension in state {tuple(state.shape)}')
        shape = [1]*state.ndim
        shape[dim] = batch
        return state*keep.view(shape).to(state.dtype)
    if isinstance(state, tuple):
        return tuple(mask_state(v, keep) for v in state)
    if isinstance(state, list):
        return [mask_state(v, keep) for v in state]
    if isinstance(state, dict):
        return {k: mask_state(v, keep) for k, v in state.items()}
    return state


def _to_device(batch, model, device):
    x = torch.from_numpy(batch['dry']).to(device, non_blocking=True)
    y = torch.from_numpy(batch['wet']).to(device, non_blocking=True)
    if x.shape[1] % model.required_multiple:
        raise ValueError('chunk_samples must be a multiple of the model block size')
    controls = torch.from_numpy(batch['controls']).to(device)
    gain = torch.from_numpy(batch['gain']).to(device)
    keep = torch.from_numpy(~batch['reset']).to(device)
    return x, y, controls, gain, keep


def scored_loss(criterion, prediction, target, rows, valid, chunk):
    """Mean of per-chunk losses, each computed exactly as train() does for one chunk.

    Full-length rows share one batched criterion call (identical to the mean of
    per-row calls); a record's final partial chunk is scored on its own.
    Returns (loss, sample-weighted part sums, scored samples).
    """
    full = [i for i in rows if valid[i] == chunk]
    partial = [i for i in rows if valid[i] < chunk]
    total = prediction.new_zeros(())
    sums = dict(loss=0., l1=0., esr=0., spectral=0.)
    samples = 0
    pieces = []
    if full:
        index = torch.tensor(full, device=prediction.device)
        pieces.append((len(full), len(full)*chunk, *criterion(prediction[index], target[index])))
    for i in partial:
        n = int(valid[i])
        pieces.append((1, n, *criterion(prediction[i:i+1, :n], target[i:i+1, :n])))
    for weight, n, loss, parts in pieces:
        total = total + weight*loss
        samples += n
        sums['loss'] += loss.item()*n
        for key, value in parts.items():
            sums[key] += value.item()*n
    return total/len(rows), sums, samples


@torch.no_grad()
def evaluate_batched(model, records, *, chunk_samples, warmup_samples, lanes,
                     block_chunks=32, device='cpu'):
    """Same metrics as training.evaluate_records, streaming `lanes` records at once."""
    if not records:
        raise ValueError('No recordings match selected split')
    model.eval()
    device = torch.device(device)
    square = np.zeros(len(records))
    absolute = np.zeros(len(records))
    energy = np.zeros(len(records))
    counts = np.zeros(len(records), dtype=np.int64)
    positions = torch.arange(chunk_samples, device=device)
    state = None
    lanes = min(lanes, len(records))
    for batch in prefetch(lane_batches(records, range(len(records)), lanes,
                                       chunk_samples, block_chunks)):
        x, target, controls, gain, keep = _to_device(batch, model, device)
        state = mask_state(state, keep)
        prediction, state = model(x, controls, state, gain)
        valid = torch.from_numpy(batch['valid']).to(device)
        offset = torch.from_numpy(batch['offset']).to(device)
        mask = (positions[None] < valid[:, None]) & (positions[None]+offset[:, None] >= warmup_samples)
        if not torch.isfinite(prediction[mask]).all():
            raise ValueError('Nonfinite model output during validation')
        weight = mask.double()
        error = (prediction-target).double()*weight
        rows = torch.stack((error.square().sum(1), error.abs().sum(1),
                            (target.double().square()*weight).sum(1), weight.sum(1))).cpu().numpy()
        live = batch['index'] >= 0
        index = batch['index'][live]
        np.add.at(square, index, rows[0][live])
        np.add.at(absolute, index, rows[1][live])
        np.add.at(energy, index, rows[2][live])
        np.add.at(counts, index, rows[3][live].astype(np.int64))
    per_record = {}
    for i, record in enumerate(records):
        if not counts[i]:
            raise ValueError(f'{record.id}: no samples after warmup')
        n = int(counts[i])
        # Plain floats: reports are stored in weights_only checkpoints.
        per_record[record.id] = dict(samples=n, mse=float(square[i]/n), mae=float(absolute[i]/n),
                                     esr=float(square[i]/max(energy[i], n*1e-8)))
    total = int(counts.sum())
    return dict(samples=total, mse=float(square.sum()/total), mae=float(absolute.sum()/total),
                esr=float(square.sum()/max(energy.sum(), total*1e-8)), records=per_record)


def _accelerate(model, device):
    if device.type == 'cuda':
        from .fused_scan import accelerate_training
        accelerate_training(model)
    return model


def train_batched(manifest, config_path, output, *, device='cpu', resume=None):
    """Explicit opt-in training; same artifacts and resume rules as training.train()."""
    config = load_batched_config(config_path)
    options = config['training']
    chunk, lanes = options['chunk_samples'], options['batch_size']
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
    _accelerate(model, device)
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
    report = None
    for epoch in range(start_epoch+1, options['epochs']+1):
        begun = time.monotonic()
        model.train()
        order = list(range(len(train_records)))
        random.Random(options['seed']+epoch).shuffle(order)  # same permutation as train()
        sums = dict(loss=0., l1=0., esr=0., spectral=0.)
        samples = updates = scored_chunks = 0
        state = None
        for batch in prefetch(lane_batches(train_records, order, lanes, chunk,
                                           options['read_block_chunks'])):
            x, target, controls, gain, keep = _to_device(batch, model, device)
            state = mask_state(state, keep)
            rows = [i for i in range(lanes) if batch['index'][i] >= 0 and
                    batch['offset'][i] >= options['warmup_samples']]
            if not rows:  # every live lane is still warming up its record
                with torch.no_grad():
                    _, state = model(x, controls, state, gain)
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(x, controls, state, gain)
            loss, step_sums, step_samples = scored_loss(criterion, prediction, target, rows,
                                                        batch['valid'], chunk)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch {epoch}, update {updates+1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            state = detach_state(state)  # TBPTT keeps values, cuts gradient history
            updates += 1
            scored_chunks += len(rows)
            samples += step_samples
            for key in sums:
                sums[key] += step_sums[key]
            if updates % options['log_every_chunks'] == 0:
                print(json.dumps(dict(epoch=epoch, updates=updates, scored_chunks=scored_chunks,
                                      loss=sums['loss']/samples,
                                      elapsed=round(time.monotonic()-begun, 1))), flush=True)
        validation = evaluate_batched(model, validation_records, chunk_samples=chunk,
                                      warmup_samples=options['warmup_samples'],
                                      lanes=options['eval_batch_size'],
                                      block_chunks=options['read_block_chunks'], device=device)
        improved = validation['esr'] < best
        best = min(best, validation['esr'])
        report = dict(epoch=epoch, updates=updates, scored_chunks=scored_chunks, batch_size=lanes,
                      train={k: v/samples for k, v in sums.items()},
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
        print(json.dumps({k: v for k, v in report.items() if k != 'validation'} |
                         dict(validation_esr=validation['esr'])), flush=True)
    return report


def evaluate_checkpoint_batched(manifest, checkpoint, *, split='validation', device='cpu', lanes=256):
    checkpoint = Path(checkpoint)
    metadata = torch.load(checkpoint/'training.pt', map_location='cpu', weights_only=True)
    options = metadata['config']['training']
    records = [r for r in load_records(manifest, options['sample_rate']) if r.split == split]
    validate_records(records, options, require_train=False)
    device = resolve_device(device)
    model = _accelerate(AudioModel.from_checkpoint(checkpoint/'model.mdlus').to(device), device)
    return evaluate_batched(model, records, chunk_samples=options['chunk_samples'],
                            warmup_samples=options['warmup_samples'], lanes=lanes,
                            block_chunks=options.get('read_block_chunks', 32), device=device)
