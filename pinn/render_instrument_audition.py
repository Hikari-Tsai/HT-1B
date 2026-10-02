"""Render synchronized model auditions from distinct regions of one CL-1B file.

This creates listening excerpts only; it does not calculate benchmark scores,
retrain models, fit controls, align audio, or normalize track loudness.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ht1b.config import Controls, read_config
from ht1b.fast_solver import FastCircuitSolver
from pinn.build_unified_comparison import wav_samples, waveform_summary
from pinn.compare_pretrained import AuthorCL1B
from pinn.render_listening_all import model_for, neural_audio
from pinn.render_listening_s6 import write_float_wav

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'site/audio/instrument-excerpts'
MANIFEST = ROOT/'runs/full-corpus-gpu-20260925/manifest.json'
COMMON = ROOT/'runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json'
LONG = ROOT/'runs/long-clip-evaluation-20260929/comparison.json'
DATA = ROOT/'data/tubetech-cl-1b-v1'

# Each excerpt is 3 seconds from the corresponding already-reviewed study region.
# Instrument tags remain provisional because the automatic labels have not had a
# dedicated human listening pass.
SEGMENTS = (
    dict(id='sweep', title='掃頻', study='A1', start=90.0, stop=93.0),
    dict(id='drums', title='鼓組', study='A2', start=96.0, stop=99.0),
    dict(id='bass', title='Bass／低音', study='A3', start=101.0, stop=104.0),
    dict(id='vocal', title='人聲', study='C', start=72.0, stop=75.0),
    dict(id='guitar', title='吉他', study='E', start=35.0, stop=38.0),
)
TRACKS = dict(reference_dry='reference_dry', reference_wet='reference_wet',
              mlp_full='mlp', gru_full='gru', s4_tfilm='s4', s6_tfilm='s6',
              s6_tfilm_pinn='s6_pinn', author_pretrained='riccardovib',
              pure_algorithm='baseline')
LABELS = dict(reference_dry='Dry input', reference_wet='實測 Wet',
              mlp_full='MLP＋PINN · 完整訓練', gru_full='GRU＋PINN · 完整訓練',
              s4_tfilm='S4＋TFiLM · 第 14 輪', s6_tfilm='S6＋TFiLM · 第 8 輪',
              s6_tfilm_pinn='S6＋TFiLM＋PINN · 第 15 輪',
              author_pretrained='RiccardoVib', pure_algorithm='純演算法',
              ours_5000='歷史 MLP · 5,000 更新')


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def circuit_output(path: Path, dry: np.ndarray, controls: dict) -> np.ndarray:
    circuit, _ = read_config(path)
    return FastCircuitSolver(circuit, Controls(**controls), 48000).process(dry.astype(np.float64))


def write_manifest():
    manifest = read(MANIFEST)
    long = read(LONG)
    records_by_path = {row['path']:row for row in manifest['records']}
    output_records = []
    for segment in SEGMENTS:
        candidate = long['studies'][segment['study']]['candidate']
        source = records_by_path[candidate['reference_path']]
        start,stop=round(segment['start']*48000),round(segment['stop']*48000)
        audio,waveforms={},{}
        wet_path=OUT/segment['id']/'reference_wet.wav'
        wet=wav_samples(wet_path)
        for track,label in LABELS.items():
            path=OUT/segment['id']/f'{track}.wav'
            if not path.is_file():
                raise FileNotFoundError(f'Missing audition audio: {path}')
            samples=wav_samples(path)
            if len(samples)!=stop-start:
                raise ValueError(f'Incorrect excerpt duration: {path}')
            audio[track]=dict(label=label,path=f'audio/instrument-excerpts/{segment["id"]}/{track}.wav')
            waveforms[track]=waveform_summary(samples,wet)
        output_records.append(dict(id=f'instrument-{segment["id"]}',asset_id=segment['id'],title=segment['title'],
            start_seconds=segment['start'],stop_seconds=segment['stop'],seconds=(stop-start)/48000,
            samples=stop-start,source_record=source['id'],source_path=source['path'],
            study=segment['study'],machine_label_status='automatic candidate; not human verified',
            audio=audio,waveform=waveforms))
    report=dict(version=1,source_record=output_records[0]['source_record'],sample_rate=48000,
        scope='listening-only excerpts; existing source, not a new independent test',
        label_status='Instrument names are provisional automatic identifications, pending human listening.',
        records=output_records)
    temp=OUT/'manifest.json.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(OUT/'manifest.json')
    print(f'Complete: {len(output_records)} instrument excerpts, {sum(len(r["audio"]) for r in output_records)} WAV files.')
    return report


@torch.inference_mode()
def riccardovib_audio(pair: np.ndarray, row: dict, source: dict) -> np.ndarray:
    checkpoint = ROOT/source['path']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('RiccardoVib checkpoint differs from common validation')
    model = AuthorCL1B(checkpoint).double().cuda().eval()
    calibration = source['conditioning']
    labels = row['labels']
    controls = np.asarray([labels['attack_index']/4, labels['release_index']/4,
                           (labels['ratio']-2)/8, abs(labels['threshold_db'])/40], dtype=np.float32)
    if calibration['threshold_flipped']:
        controls[3] = 1-controls[3]
    controls = torch.as_tensor(controls[calibration['permutation']], device='cuda', dtype=torch.float64)
    dry = pair[:, 0]
    outputs = {}
    for segment in SEGMENTS:
        start, stop = round(segment['start']*48000), round(segment['stop']*48000)
        windows = torch.as_tensor(dry[start-16:stop].copy(), device='cuda', dtype=torch.float64).unfold(0,32,16)
        chunks = []
        for offset in range(0, len(windows), 4096):
            batch = windows[offset:offset+4096]
            chunks.append(model(batch, controls.expand(len(batch),-1)).reshape(-1).cpu().numpy())
        outputs[segment['id']] = np.concatenate(chunks)
    return outputs


def render():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the frozen S6 and RiccardoVib inference paths')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(MANIFEST)
    common = read(COMMON)
    long = read(LONG)
    records_by_path = {row['path']:row for row in manifest['records']}
    chosen = []
    for segment in SEGMENTS:
        study = long['studies'][segment['study']]['candidate']
        path = study['reference_path']
        if path not in records_by_path:
            raise ValueError(f'Missing representative source recording: {path}')
        row = records_by_path[path]
        if segment['start'] < 16/48000 or segment['stop']*48000 > row['frames']:
            raise ValueError(f'Excerpt outside source file: {segment["id"]}')
        chosen.append((segment,row))
    source_ids = {row['id'] for _,row in chosen}
    if len(source_ids) != 1:
        raise ValueError('For this audition, all snippets must share one continuous source recording')
    row = chosen[0][1]
    max_stop = max(round(item['stop']*48000) for item,_ in chosen)
    pair,sample_rate = sf.read(DATA/row['path'],frames=max_stop,dtype='float32',always_2d=True)
    if sample_rate != 48000 or pair.shape != (max_stop,2):
        raise ValueError('Source WAV is not the expected stereo 48 kHz recording')
    dry = pair[:,0]
    source_by_track = {}

    # Physics-only baselines and the historical 5,000-update fitted circuit.
    circuit_sources = dict(baseline=ROOT/common['sources']['baseline']['path'],
        mlp=ROOT/common['sources']['mlp']['path'], gru=ROOT/common['sources']['gru']['path'],
        ours_5000=ROOT/'runs/gpu-continuation-2000-to-5000/step-5000/fitted.json')
    for track,model_key in circuit_sources.items():
        output = circuit_output(model_key,dry,row['controls'])
        source_by_track[track] = {s['id']:output[round(s['start']*48000):round(s['stop']*48000)].copy()
                                  for s,_ in chosen}
        print('Rendered',track,flush=True)

    # Frozen neural checkpoints; scan from sample zero so recurrent state is retained.
    for model_key in ('s4','s6','s6_pinn'):
        model = model_for(model_key,common['sources'][model_key])
        output = neural_audio(model_key,model,pair,row['labels'],row['controls'])
        source_by_track[model_key] = {s['id']:output[round(s['start']*48000):round(s['stop']*48000)].copy()
                                      for s,_ in chosen}
        del model,output
        torch.cuda.empty_cache()
        print('Rendered',model_key,flush=True)

    source_by_track['riccardovib'] = riccardovib_audio(pair,row,common['sources']['riccardovib'])
    print('Rendered riccardovib',flush=True)

    for segment,_ in chosen:
        start,stop = round(segment['start']*48000),round(segment['stop']*48000)
        dry_clip,wet_clip = pair[start:stop,0].copy(),pair[start:stop,1].copy()
        arrays = dict(reference_dry=dry_clip,reference_wet=wet_clip)
        for track,model_key in TRACKS.items():
            if track in ('reference_dry','reference_wet'):
                continue
            arrays[track] = source_by_track[model_key][segment['id']]
        if 'ours_5000' not in arrays:
            arrays['ours_5000'] = source_by_track['ours_5000'][segment['id']]
        folder = OUT/segment['id']
        for track,samples in arrays.items():
            if len(samples) != stop-start or not np.isfinite(samples).all():
                raise ValueError(f'Invalid rendered track {segment["id"]}/{track}')
            path = folder/f'{track}.wav'
            write_float_wav(path,samples)
        print('Wrote',segment['id'],f'{(stop-start)/48000:.1f}s',flush=True)
    write_manifest()


if __name__ == '__main__':
    write_manifest() if '--manifest-only' in sys.argv else render()
