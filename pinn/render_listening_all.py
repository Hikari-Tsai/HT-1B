"""Render all current comparison models on the five matched audition clips.

Uses frozen comparison checkpoints and original Dry from sample zero, then saves
only the historical common interval. Does not fit, align, normalize, or retrain.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ht1b.config import Circuit, Controls, read_config
from ht1b.equations import evaluate
from ht1b.fast_solver import FastCircuitSolver
from s6.models import AudioModel
from s6.inference_scan import accelerate

from pinn.render_listening_s6 import ROOT, HISTORICAL, MANIFEST, COMMON, read_json, read_original, read_float_wav, write_float_wav

KEYS = ('mlp', 'gru', 's4', 's6', 's6_pinn')
FOLDERS = dict(mlp='mlp_full', gru='gru_full', s4='s4_tfilm', s6='s6_tfilm', s6_pinn='s6_tfilm_pinn')

# Runtime-compatible copies of the fixed physical head; the evaluation's
# PhysicsNeMo-Sym import is unavailable in the current inference-only venv.
BOUNDS = {'gre_span': (1e-6,.02), 'gre_half': (.001,3.),
          'gre_attack_fast': (.0001,.05), 'gre_attack_slow': (.001,.5),
          'gre_release_fast': (.002,2.), 'gre_release_slow': (.02,20.),
          'amp_gain': (.25,12.)}


class TorchOps:
    @staticmethod
    def maximum(a,b):
        if not torch.is_tensor(a):a=torch.as_tensor(a)
        return torch.maximum(a,torch.as_tensor(b,dtype=a.dtype,device=a.device))
    @staticmethod
    def minimum(a,b):
        if not torch.is_tensor(a):a=torch.as_tensor(a)
        return torch.minimum(a,torch.as_tensor(b,dtype=a.dtype,device=a.device))
    absolute = staticmethod(torch.abs)
    tanh = staticmethod(torch.tanh)
    @staticmethod
    def where(condition,a,b):
        a=torch.as_tensor(a,device=condition.device)
        b=torch.as_tensor(b,device=condition.device)
        return torch.where(condition,a,b)


class LearnedCircuitCompat(nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.base = circuit
        self.raw = nn.ParameterDict({name: nn.Parameter(torch.tensor(
            math.log(((getattr(circuit,name)-lo)/(hi-lo))/(1-(getattr(circuit,name)-lo)/(hi-lo))),
            dtype=torch.float64)) for name,(lo,hi) in BOUNDS.items()})

    def values(self):
        values = asdict(self.base)
        for name,(lo,hi) in BOUNDS.items():
            values[name] = lo+(hi-lo)*torch.sigmoid(self.raw[name])
        return SimpleNamespace(**values)


class S6PINNCompat(AudioModel):
    def __init__(self, **options):
        super().__init__(**options)
        self.output = nn.Linear(self.output.in_features,3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.physical = LearnedCircuitCompat(Circuit())
        self.register_buffer('physical_prior',torch.stack([p.detach().clone() for p in self.physical.raw.values()]))

    def trajectory(self,dry,controls,state=None):
        previous={} if state is None else state
        history=previous.get('history')
        if history is None:history=dry.new_zeros(dry.shape[0],self.context-1)
        audio=torch.cat((history,dry),dim=1)
        windows=audio.unfold(1,self.context,1)
        z=self.input(windows)
        spectrum=self.spectrum(torch.fft.rfft(windows,n=2*self.context).abs()/self.context)
        z,first=self.first(z,previous.get('first'))
        z,condition=self.condition(z,spectrum,controls,previous.get('condition'))
        z,second=self.second(z,previous.get('second'))
        raw=self.output(z).double()
        states=torch.cat((30*torch.sigmoid(raw[...,:1]-6),torch.sigmoid(raw[...,1:]-3)),dim=-1)
        recurrent=dict(history=audio[:,-(self.context-1):] if self.context>1 else audio[:,:0],
                       first=first,condition=condition,second=second)
        return states,recurrent


def label_controls(labels):
    return ([-labels['threshold_db']/40, (labels['ratio']-2)/8,
             labels['attack_index']/4, labels['release_index']/4], labels['gain_db'])


def model_for(key, source):
    checkpoint = ROOT / source['path']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError(f'Checkpoint differs from common evaluation: {key}')
    if key == 's6_pinn':
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = S6PINNCompat(**saved['config']['model'])
        model.load_state_dict(saved['model'])
    else:
        model = AudioModel.from_checkpoint(str(checkpoint))
    model = model.cuda().eval()
    if key in ('s6', 's6_pinn'):
        accelerate(model)
    return model


@torch.inference_mode()
def neural_audio(key, model, pair, labels, physical_controls):
    controls, gain_db = label_controls(labels)
    controls = torch.tensor([controls], device='cuda', dtype=torch.float32)
    gain = torch.tensor([gain_db], device='cuda', dtype=torch.float32)
    if key == 's6_pinn':
        pots = SimpleNamespace(**{name: torch.tensor([physical_controls[name]], dtype=torch.float64, device='cuda')[:, None]
                                  for name in ('threshold', 'ratio', 'attack', 'release', 'makeup_db')}, mode='manual')
        params = model.physical.values()
    state, chunks = None, []
    for start in range(0, len(pair), 8192):
        actual = min(8192, len(pair)-start)
        padded = ((actual+127)//128)*128 if key == 's4' else actual
        dry = np.zeros(padded, dtype=np.float32)
        dry[:actual] = pair[start:start+actual, 0]
        x = torch.from_numpy(dry)[None].to('cuda')
        if key == 's6_pinn':
            latent, state = model.trajectory(x, controls, state)
            wet = evaluate(tuple(latent.unbind(-1)), x.double()*params.input_volts_per_fs,
                           params, pots, TorchOps)['output']/params.output_volts_per_fs
        else:
            wet, state = model(x, controls, state, gain_db=gain)
        chunks.append(wet[0, :actual].cpu().numpy())
    return np.concatenate(chunks)


def render():
    comparison = read_json(HISTORICAL/'comparison.json')
    manifest = {r['id']: r for r in read_json(MANIFEST)['records']}
    sources = read_json(COMMON)['sources']
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the frozen S6/S6+PINN scan path')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = []
    for item in comparison['records']:
        record = manifest[item['id']]
        begin, stop = item['start_sample'], item['stop_sample']
        pair = read_original(ROOT/'data/tubetech-cl-1b-v1'/record['path'], stop)
        wet = read_float_wav(HISTORICAL/'audio/reference_wet'/f"{item['id']}.wav")
        dry = read_float_wav(HISTORICAL/'audio/reference_dry'/f"{item['id']}.wav")
        if not np.array_equal(pair[begin:stop, 0], dry) or not np.array_equal(pair[begin:stop, 1], wet):
            raise ValueError(f'Historical source mismatch: {item["id"]}')
        prepared.append((item, record, pair, wet))

    all_results = {}
    for key in KEYS:
        source = sources[key]
        checkpoint = ROOT/source['path']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Artifact differs from evaluation: {key}')
        model = model_for(key, source) if key not in ('mlp', 'gru') else None
        if key in ('mlp', 'gru'):
            circuit, _ = read_config(checkpoint)
        results = []
        for item, record, pair, wet in prepared:
            if model is None:
                full = FastCircuitSolver(circuit, Controls(**record['controls']), 48000).process(pair[:, 0].astype(np.float64))
            else:
                full = neural_audio(key, model, pair, record['labels'], record['controls'])
            predicted = np.asarray(full[item['start_sample']:item['stop_sample']], dtype=np.float64)
            if len(predicted) != len(wet) or not np.isfinite(predicted).all():
                raise ValueError(f'Invalid {key} output: {item["id"]}')
            target = HISTORICAL/'audio'/FOLDERS[key]/f"{item['id']}.wav"
            write_float_wav(target, predicted)
            residual = predicted-wet.astype(np.float64)
            esr = float((residual@residual)/max(float(wet.astype(np.float64)@wet.astype(np.float64)),len(wet)*1e-8))
            results.append(dict(id=item['id'], samples=len(predicted), esr=esr, wav=str(target.relative_to(ROOT))))
            print(key, item['id'], f'ESR={esr:.6f}', flush=True)
        all_results[key] = dict(source=source, records=results)
        if model is not None:
            del model
            torch.cuda.empty_cache()
    summary = dict(sample_rate=48000, interval='same five historical clips, [16, 95984); original Dry processed from sample 0',
                   method='frozen six-model comparison artifacts; no fitting, alignment, normalization, or retraining',
                   models=all_results, riccardovib='existing pretrained audition WAVs retained')
    (HISTORICAL/'audio/current-model-provenance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    render()
