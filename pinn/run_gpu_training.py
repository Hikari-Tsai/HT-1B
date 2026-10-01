"""Run a bounded CUDA pilot and compare default/fitted circuit solvers.

Run from the repository root in the Linux CUDA environment. Raw audio remains
unchanged. Outputs are local under runs/. This is a prefix/settings experiment,
not a full-dataset or independent-source generalization benchmark.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from ht1b.cli import evaluate_manifest
from ht1b.config import read_config
from ht1b.dataset import parse_filename, prepare_manifest
from ht1b.training import train


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('data/tubetech-cl-1b-v1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--seconds',type=float,default=2.)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; this runner never falls back to CPU training')
    # Execute and synchronize a real float64 GPU operation before preparing data.
    probe=torch.randn((128,128),device='cuda',dtype=torch.float64)
    assert torch.isfinite(probe@probe.T).all().item()
    torch.cuda.synchronize()
    args.output.mkdir(parents=True,exist_ok=False)
    out=args.output.resolve()
    def status(phase, **extra):
        write_json(out/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**extra))
        print(f'PHASE {phase}',flush=True)
    try:
        status('preparing')
        inventory={}
        for path in sorted(args.dataset.rglob('*.wav')):
            p=parse_filename(path)
            if p['gain_db']==0:
                key=(p['attack_index'],p['release_index'],p['ratio'],p['threshold_db'])
                if key in inventory: raise ValueError(f'Duplicate recording settings: {key}')
                inventory[key]=path.resolve()
        # Balanced 25-setting grid: each knob level appears five times.
        selected=[]
        for attack in range(5):
            for release in range(5):
                key=(attack,release,(2,4,6,8,10)[(attack+release)%5],-10*((attack+2*release)%5))
                selected.append(inventory[key])
        circuit,_=read_config('configs/default.json')
        manifest=prepare_manifest(selected,out/'manifest.json',circuit,seconds=args.seconds)
        payload=json.loads(manifest.read_text())
        for row in payload['records']:
            labels=row['metadata']['filename_parameters']
            row['split']='validation' if labels['attack_index']==labels['release_index'] else 'train'
        write_json(manifest,payload)
        comparison_rows=[r for r in payload['records'] if r['split']=='validation']
        comparison_rows.append(next(r for r in payload['records'] if r['split']=='train'))
        comparison_manifest=out/'comparison-manifest.json'
        write_json(comparison_manifest,{'schema_version':1,'records':comparison_rows})
        config=out/'initial-config.json'
        config.write_text(Path('configs/default.json').read_text(),encoding='utf-8')
        settings=dict(steps=args.steps,batch_size=512,width=64,layers=3,learning_rate=.001,
                      seed=7,device='cuda',checkpoint_every=100)
        write_json(out/'run-config.json',dict(
            settings=settings,seconds_per_record=args.seconds,train_records=20,validation_records=5,
            comparison_records=len(comparison_rows),dataset=str(args.dataset.resolve()),
            gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,
            physicsnemo_version=version('nvidia-physicsnemo'),
            validation_scope='Held-out knob settings; independence of source audio is not established.',
            assumptions=['Manual mode','Zero integer delay','Released initial state','Uncalibrated knob mapping'],
            circuit=asdict(circuit)))
        status('training',device='cuda',gpu=torch.cuda.get_device_name(),steps=args.steps)
        report=train(manifest,out,circuit,**settings)
        # Evaluate the same six recordings with the same sample rate, alignment,
        # initial states, and controls; only physical circuit parameters differ.
        for label,config_path in [('baseline',config),('fitted',out/'fitted.json')]:
            status(f'evaluating_{label}',training_step=report['step'])
            evaluate_manifest(SimpleNamespace(manifest=comparison_manifest,config=config_path,
                output=out/f'{label}-evaluation.json',split='all',substeps=1,
                audio_dir=out/f'audio-{label}'))
        before=json.loads((out/'baseline-evaluation.json').read_text())['records']
        after=json.loads((out/'fitted-evaluation.json').read_text())['records']
        comparisons=[]
        for row in comparison_rows:
            name=row['id']
            x,sr=sf.read(row['stereo_pair'],frames=row['stop_sample'],always_2d=True)
            reference_dir=out/'audio-reference'
            reference_dir.mkdir(exist_ok=True)
            sf.write(reference_dir/f'{name}-dry.wav',x[:,0],sr,subtype='FLOAT')
            sf.write(reference_dir/f'{name}-wet.wav',x[:,1],sr,subtype='FLOAT')
            comparisons.append(dict(id=name,split=row['split'],
                baseline={k:before[name][k] for k in ('mse','mae','esr','seconds')},
                fitted={k:after[name][k] for k in ('mse','mae','esr','seconds')},
                esr_reduction_percent=100*(before[name]['esr']-after[name]['esr'])/before[name]['esr']))
        validation=[r for r in comparisons if r['split']=='validation']
        means={model:{metric:float(np.mean([r[model][metric] for r in validation]))
               for metric in ('mse','mae','esr')} for model in ('baseline','fitted')}
        write_json(out/'comparison.json',dict(records=comparisons,validation_macro_mean=means,
            note='Lower error is better. Held-out settings only; first prefixes, not unseen-source/full-length validation.'))
        lines=['# GPU pilot comparison','',
            f'GPU: {torch.cuda.get_device_name()}; PhysicsNeMo {version("nvidia-physicsnemo")}; '
            f'{args.steps} steps; 20 training + 5 held-out settings; first {args.seconds:g} seconds per recording.','',
            'Baseline: default circuit parameters with the numerical solver. Fitted: GPU PINN-identified parameters with the same solver.',
            'PINN trajectory reconstruction is reported separately in report.json and is not a deployment score.','',
            '| Recording | Split | Baseline ESR | Fitted ESR | Error reduction |',
            '|---|---|---:|---:|---:|']
        for r in comparisons:
            lines.append(f'| {r["id"]} | {r["split"]} | {r["baseline"]["esr"]:.6g} | {r["fitted"]["esr"]:.6g} | {r["esr_reduction_percent"]:.2f}% |')
        lines.extend(['',f'Validation macro mean ESR: {means["baseline"]["esr"]:.6g} -> {means["fitted"]["esr"]:.6g}.',
            '', 'A positive error reduction is an improvement; this is not a hardware fidelity percentage.',
            'The split holds out knob settings. Source-audio independence is not established. No full-length training was performed.',
            'Listen to audio-reference, audio-baseline and audio-fitted using identical playback gain.'])
        (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status('complete',training_step=report['step'],validation_macro_mean=means)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main()
