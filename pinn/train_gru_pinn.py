"""Train a causal GRU + PhysicsNeMo inverse PINN on every train sample.

The GRU sees only Dry audio and controls, including across held-out intervals.
Wet is used for gradient updates in train intervals only. The trajectory network
identifies shared circuit parameters; deployment still uses the circuit solver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from physicsnemo.models.mlp.fully_connected import FullyConnected

from ht1b.config import Controls, read_config, write_config
from ht1b.corpus import load_corpus
from ht1b.physicsnemo_model import CircuitPDE, LearnedCircuit, residuals, waveform


RATE = 48000
FRAME = 480  # 10 ms recurrent context; sample-rate physics/data losses remain intact.
HIDDEN = 64


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


class CausalGRUTrajectory(nn.Module):
    """Dry-history GRU and PhysicsNeMo samplewise decoder for latent e/f/s."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=7, hidden_size=HIDDEN, batch_first=True)
        self.decoder = FullyConnected(in_features=HIDDEN + 7, out_features=3,
                                      layer_size=64, num_layers=3,
                                      activation_fn='tanh', weight_norm=False)

    def forward(self, dry: torch.Tensor, controls: torch.Tensor,
                sample_start: int, duration: float,
                hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        n = dry.numel()
        padding = (-n) % FRAME
        frames = F.pad(dry, (0, padding)).reshape(-1, FRAME)
        counts = torch.full((len(frames),), FRAME, device=dry.device, dtype=dry.dtype)
        if padding:
            counts[-1] -= padding
        rms = torch.sqrt(frames.square().sum(dim=1) / counts + 1e-12)
        peak = frames.abs().amax(dim=1)
        context_in = torch.cat((rms[:, None], peak[:, None],
                                controls.expand(len(frames), -1)), dim=1)
        updated_context, next_hidden = self.gru(context_in[None], hidden)
        # A frame's RMS/peak contains later samples in that 10 ms frame. Use
        # its GRU state only for the *next* frame, so the decoder is causal.
        prior_context = (hidden[0] if hidden is not None else
                         torch.zeros((1, HIDDEN), device=dry.device, dtype=dry.dtype))
        context = torch.cat((prior_context, updated_context[0, :-1]), dim=0)
        context = context.repeat_interleave(FRAME, dim=0)[:n]
        time_column = (torch.arange(sample_start + 1, sample_start + n + 1,
                                    device=dry.device, dtype=dry.dtype) / RATE)[:, None]
        raw = self.decoder(torch.cat((dry[:, None], time_column / duration,
                                      controls.expand(n, -1), context), dim=1))
        bounded = torch.cat((30 * torch.sigmoid(raw[:, :1] - 6),
                             torch.sigmoid(raw[:, 1:] - 3)), dim=1)
        states = bounded * time_column / (time_column + .001)
        return states, next_hidden


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    if args.chunk_samples < 1 or args.epochs != 1:
        raise ValueError('This exhaustive sequential experiment requires one epoch and a positive chunk size')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    corpus, root = load_corpus(args.manifest, args.dataset)
    circuit, _ = read_config(args.config)
    total = corpus['frames_by_split']['train']
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    options = dict(architecture='causal_gru_pinn_v1', chunk_samples=args.chunk_samples,
                   frame_samples=FRAME, gru_hidden=HIDDEN, decoder_width=64,
                   decoder_layers=3, seed=args.seed, epochs=args.epochs,
                   network_lr=args.network_lr, physical_lr=args.physical_lr)
    network = CausalGRUTrajectory().to(device='cuda', dtype=torch.float64)
    physical = LearnedCircuit(circuit).to(device='cuda', dtype=torch.float64)
    optimizer = torch.optim.Adam([{'params': network.parameters(), 'lr': args.network_lr},
                                  {'params': physical.parameters(), 'lr': args.physical_lr}])
    prior = torch.stack([v.detach().clone() for v in physical.raw.values()])
    order = np.random.default_rng(args.seed).permutation(len(corpus['records']))
    record_cursor = position = step = covered = chunks = 0
    hidden = last_state = None
    per_record: dict[str, int] = {}
    per_group: dict[str, int] = {}
    latest_loss: dict[str, float] = {}
    if args.resume:
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if (saved['manifest_signature'] != corpus['signature'] or
                saved['options'] != options or saved['circuit'] != asdict(circuit)):
            raise ValueError('Resume manifest/options/circuit mismatch')
        network.load_state_dict(saved['network'])
        physical.load_state_dict(saved['physical'])
        optimizer.load_state_dict(saved['optimizer'])
        # Optimizer tensors must follow the CUDA parameters after CPU checkpoint load.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to('cuda')
        record_cursor = saved['record_cursor']
        position = saved['position']
        step = saved['step']
        covered = saved['covered']
        chunks = saved['chunks']
        per_record = saved['per_record']
        per_group = saved['per_group']
        latest_loss = saved['latest_loss']
        hidden = saved['hidden'].to('cuda') if saved['hidden'] is not None else None
        last_state = saved['last_state'].to('cuda') if saved['last_state'] is not None else None
    started = time.perf_counter()
    session_start_covered = covered
    last_save = started
    equations: dict[int, CircuitPDE] = {}
    write_json(output / 'run-config.json', dict(options=options,
        manifest=str(args.manifest.resolve()), manifest_signature=corpus['signature'],
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, records=len(corpus['records']),
        frames_per_epoch=total, hours_per_epoch=total/RATE/3600,
        architecture='Previous 10 ms Dry RMS/peak + controls -> PyTorch GRU(64) context -> PhysicsNeMo FullyConnected(64 x 3) with current Dry -> e/f/s; no within-frame lookahead.',
        objective='Physics residual + train-Wet block-energy-normalized waveform MSE + physical prior.',
        holdout='Validation/test Wet never enters gradient updates; Dry advances recurrent history.',
        smoke_chunks=args.smoke_chunks))

    def status(phase: str, **extra: object) -> None:
        elapsed = time.perf_counter() - started
        speed = (covered - session_start_covered) / max(elapsed, 1e-6)
        row = corpus['records'][int(order[record_cursor])] if record_cursor < len(order) else None
        payload = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(),
            gpu=torch.cuda.get_device_name(), cuda=True, record_cursor=record_cursor,
            records_total=len(order), current_record=row['id'] if row else None,
            sample_position=position, step=step, chunks=chunks,
            frames_covered=covered, frames_planned=total,
            coverage_percent=100*covered/total, files_visited=len(per_record),
            seconds=elapsed, samples_per_second=speed,
            estimated_remaining_seconds=(total-covered)/speed if speed>0 else None,
            latest_loss=latest_loss, peak_cuda_bytes=torch.cuda.max_memory_allocated(), **extra)
        write_json(output / 'status.json', payload)
        print(json.dumps(payload), flush=True)

    def save() -> None:
        state = dict(format_version=1, manifest_signature=corpus['signature'],
            options=options, circuit=asdict(circuit), network=network.state_dict(),
            physical=physical.state_dict(), optimizer=optimizer.state_dict(),
            record_cursor=record_cursor, position=position, step=step,
            covered=covered, chunks=chunks, per_record=per_record,
            per_group=per_group, latest_loss=latest_loss,
            hidden=hidden.detach().cpu() if hidden is not None else None,
            last_state=last_state.detach().cpu() if last_state is not None else None)
        temp = output / 'checkpoint.tmp'
        torch.save(state, temp)
        temp.replace(output / 'last.pt')
        write_config(output / 'fitted.json', physical.export(), Controls(),
                     dict(step=step, source='causal GRU + PhysicsNeMo inverse PINN',
                          complete_training_pass=covered == total,
                          manifest_signature=corpus['signature']))
        write_json(output / 'coverage.json', dict(frames_covered=covered,
            frames_per_epoch=total, per_record=per_record, per_group=per_group,
            holdout_wet_training_samples=0))

    status('training')
    try:
        while record_cursor < len(order):
            index = int(order[record_cursor])
            row = corpus['records'][index]
            controls = Controls(**row['controls'])
            control_vector = torch.tensor([[row['controls'][key] for key in
                ('threshold', 'ratio', 'attack', 'release', 'makeup_db')]],
                device='cuda', dtype=torch.float64)
            if index not in equations:
                equations[index] = CircuitPDE(circuit, controls)
            pde = equations[index]
            duration = row['frames'] / RATE
            with sf.SoundFile(root / row['path']) as source:
                source.seek(position)
                for interval in row['intervals']:
                    if interval['stop_sample'] <= position:
                        continue
                    if position < interval['start_sample']:
                        raise ValueError('Recording cursor skipped a partition')
                    while position < interval['stop_sample']:
                        n = min(args.chunk_samples, interval['stop_sample'] - position)
                        audio = source.read(n, dtype='float64', always_2d=True)
                        if len(audio) != n or not np.isfinite(audio).all():
                            raise ValueError('Truncated/nonfinite recording')
                        dry = torch.from_numpy(audio[:, 0].copy()).to('cuda')
                        train = interval['split'] == 'train'
                        if train:
                            states, hidden = network(dry, control_vector, position, duration, hidden)
                            previous = torch.cat((last_state if last_state is not None else
                                torch.zeros((1,3), device='cuda', dtype=torch.float64), states[:-1]), dim=0)
                            derivatives = (states - previous) * RATE
                            params = physical.values()
                            volts = dry[:, None] * circuit.input_volts_per_fs
                            physics = residuals(pde, states, derivatives, volts, params)
                            physics_loss = sum(value.square().mean() for value in physics.values()) / len(physics)
                            wet = torch.from_numpy(audio[:, 1].copy()).to('cuda')[:, None]
                            energy = max(float(np.mean(audio[:, 1]**2)), 1e-8)
                            data_loss = (waveform(states, volts, params, controls) - wet).square().mean() / energy
                            regularizer = (torch.stack(list(physical.raw.values())) - prior).square().mean()
                            loss = physics_loss + data_loss + 1e-4*regularizer
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite GRU-PINN loss')
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(list(network.parameters()) +
                                list(physical.parameters()), 10., error_if_nonfinite=True)
                            optimizer.step()
                            step += 1
                            covered += n
                            per_record[row['id']] = per_record.get(row['id'], 0) + n
                            group = str(interval['group'])
                            per_group[group] = per_group.get(group, 0) + n
                            latest_loss = dict(total=float(loss.detach()),
                                physics=float(physics_loss.detach()), data=float(data_loss.detach()))
                        else:
                            with torch.no_grad():
                                states, hidden = network(dry, control_vector, position, duration, hidden)
                        last_state = states[-1:].detach()
                        hidden = hidden.detach()
                        position += n
                        chunks += 1
                        if time.perf_counter() - last_save >= 60 or (args.smoke_chunks and chunks >= args.smoke_chunks):
                            save()
                            status('training')
                            last_save = time.perf_counter()
                        if args.smoke_chunks and chunks >= args.smoke_chunks:
                            status('smoke_complete', note='Infrastructure check only; not a full pass')
                            return
            if position != row['frames']:
                raise RuntimeError('Recording not fully traversed')
            record_cursor += 1
            position = 0
            hidden = last_state = None
        if covered != total:
            raise RuntimeError(f'Incomplete training coverage: {covered} != {total}')
        expected = {r['id']:sum(i['stop_sample']-i['start_sample'] for i in r['intervals']
                             if i['split']=='train') for r in corpus['records']}
        if per_record != expected:
            raise RuntimeError('Per-record training coverage mismatch')
        save()
        status('training_complete', evaluation_pending=True)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/default.json'))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--chunk-samples', type=int, default=15360)
    parser.add_argument('--network-lr', type=float, default=.001)
    parser.add_argument('--physical-lr', type=float, default=.0001)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--smoke-chunks', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
