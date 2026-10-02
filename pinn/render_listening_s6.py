"""Render the frozen S6 checkpoint for the five historical audition clips.

The original stereo recording supplies Dry from sample zero. Only the exact
historical [16, 95984) comparison interval is written. No target-based fitting,
gain matching, alignment, or retraining is performed.
"""
from __future__ import annotations

import hashlib
import json
import struct
import wave
from pathlib import Path

import numpy as np
import torch

from s6.models import AudioModel
from s6.inference_scan import accelerate

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'runs/pretrained-comparison'
MANIFEST = ROOT / 'runs/full-corpus-gpu-20260925/manifest.json'
COMMON = ROOT / 'runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json'
OUTPUT = HISTORICAL / 'audio/s6_tfilm'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_original(path, stop):
    with wave.open(str(path), 'rb') as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (48000, 2, 2):
            raise ValueError(f'Unexpected source WAV format: {path}')
        stereo = np.frombuffer(handle.readframes(stop), dtype='<i2').reshape(-1, 2)
    if stereo.shape[0] != stop:
        raise ValueError(f'Short original recording: {path}')
    return stereo.astype(np.float32) / 32768


def read_float_wav(path):
    raw = path.read_bytes()
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError(f'Invalid comparison WAV: {path}')
    pos = 12
    while pos + 8 <= len(raw):
        kind, size = struct.unpack_from('<4sI', raw, pos)
        start = pos + 8
        if kind == b'data':
            return np.frombuffer(raw[start:start + size], dtype='<f4')
        pos = start + size + (size & 1)
    raise ValueError(f'No audio data: {path}')


def write_float_wav(path, samples):
    audio = np.asarray(samples, dtype='<f4').tobytes()
    fmt = struct.pack('<HHIIHH', 3, 1, 48000, 48000 * 4, 4, 32)
    data = (b'RIFF' + struct.pack('<I', 4 + 8 + len(fmt) + 8 + len(audio)) + b'WAVE'
            + b'fmt ' + struct.pack('<I', len(fmt)) + fmt
            + b'data' + struct.pack('<I', len(audio)) + audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.wav.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def render():
    comparison = read_json(HISTORICAL / 'comparison.json')
    records = {r['id']: r for r in read_json(MANIFEST)['records']}
    source = read_json(COMMON)['sources']['s6']
    checkpoint = ROOT / source['path']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('S6 checkpoint hash differs from the evaluated source')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the tested fused S6 inference path')
    torch.set_num_threads(2)
    model = accelerate(AudioModel.from_checkpoint(str(checkpoint)).cuda().eval())
    summary = []
    with torch.inference_mode():
        for item in comparison['records']:
            original = records[item['id']]
            begin, stop = item['start_sample'], item['stop_sample']
            pair = read_original(ROOT / 'data/tubetech-cl-1b-v1' / original['path'], stop)
            dry = read_float_wav(HISTORICAL / 'audio/reference_dry' / f"{item['id']}.wav")
            wet = read_float_wav(HISTORICAL / 'audio/reference_wet' / f"{item['id']}.wav")
            if not np.array_equal(pair[begin:stop, 0], dry) or not np.array_equal(pair[begin:stop, 1], wet):
                raise ValueError(f'Original recording does not match historical audition: {item["id"]}')
            labels = original['labels']
            controls = torch.tensor([[-labels['threshold_db'] / 40,
                                      (labels['ratio'] - 2) / 8,
                                      labels['attack_index'] / 4,
                                      labels['release_index'] / 4]], dtype=torch.float32, device='cuda')
            gain = torch.tensor([labels['gain_db']], dtype=torch.float32, device='cuda')
            state, chunks = None, []
            for start in range(0, stop, 8192):
                x = torch.from_numpy(pair[start:min(start + 8192, stop), 0].copy()).to('cuda')[None, :]
                output, state = model(x, controls, state, gain_db=gain)
                chunks.append(output[0].cpu().numpy())
            predicted = np.concatenate(chunks)[begin:stop]
            if len(predicted) != len(wet) or not np.isfinite(predicted).all():
                raise ValueError(f'Invalid S6 output: {item["id"]}')
            target = OUTPUT / f"{item['id']}.wav"
            write_float_wav(target, predicted)
            esr = float(np.sum((predicted.astype(np.float64) - wet) ** 2) /
                        max(float(np.sum(wet.astype(np.float64) ** 2)), len(wet) * 1e-8))
            summary.append(dict(id=item['id'], samples=len(predicted), esr=esr, wav=str(target.relative_to(ROOT))))
            print(item['id'], f'ESR={esr:.6f}', flush=True)
    (OUTPUT / 'provenance.json').write_text(json.dumps(dict(checkpoint=source, sample_rate=48000,
        interval='historical [16, 95984); original Dry processed from sample 0',
        method='frozen S6+TFiLM; no fitting, gain matching, or temporal alignment',
        records=summary), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    render()
