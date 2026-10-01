"""Continue a CUDA pilot to a total step count, validating each saved stage."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from ht1b.cli import evaluate_manifest
from ht1b.config import read_config
from ht1b.training import train


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-step',type=int,default=5000)
    parser.add_argument('--interval',type=int,default=1000)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is disabled')
    source=args.run.resolve()
    checkpoint=source/'last.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    start=saved['step']
    options=saved['options'].copy()
    if args.target_step<=start or args.interval<1: raise ValueError('Target must exceed the saved step; interval must be positive')
    original_report=json.loads((source/'report.json').read_text())
    if original_report['step']!=start: raise ValueError('Source report does not match checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    out=args.output.resolve()
    def status(phase,**extra):
        write_json(out/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**extra))
        print(json.dumps(dict(phase=phase,**extra)),flush=True)
    try:
        for name in ('manifest.json','initial-config.json','run-config.json'):
            shutil.copy2(source/name,out/name)
        manifest=out/'manifest.json'
        circuit,_=read_config(out/'initial-config.json')
        payload=json.loads(manifest.read_text())
        validation=[r for r in payload['records'] if r['split']=='validation']
        if not validation: raise ValueError('No validation recordings')
        validation_manifest=out/'validation-manifest.json'
        write_json(validation_manifest,{'schema_version':1,'records':validation})
        baseline=json.loads((source/'baseline-evaluation.json').read_text())['records']
        previous=json.loads((source/'fitted-evaluation.json').read_text())['records']
        names=[r['id'] for r in validation]
        def average(records):
            return {k:float(np.mean([records[name][k] for name in names])) for k in ('mse','mae','esr')}
        reference=average(baseline)
        stages=[dict(step=start,fixed_loss=original_report['last_loss'],validation=average(previous),
                     directory=str(source),training_seconds=original_report['seconds'])]
        write_json(out/'continuation-config.json',dict(source=str(source),start_step=start,
            target_step=args.target_step,interval=args.interval,options=options,device='cuda',
            gpu=torch.cuda.get_device_name(),selection_metric='validation macro mean ESR',
            validation_scope='Same five held-out knob settings; prefixes only, not independent-source validation.'))

        def save_summary():
            best=min(stages,key=lambda stage:stage['validation']['esr'])
            write_json(out/'progression.json',dict(baseline=reference,stages=stages,best_step=best['step'],
                selected_by='Lowest validation macro mean ESR among evaluated stages, including the starting checkpoint.'))
            lines=['# GPU 續訓比較','',
                f'GPU：{torch.cuda.get_device_name()}。沿用 20 組訓練、5 組保留設定，每檔前 2 秒；訓練選項與原 checkpoint 相同。','',
                '| 總步數 | 固定評估 loss | 保留設定 MSE | 保留設定 MAE | 保留設定 ESR |',
                '|---:|---:|---:|---:|---:|']
            for stage in stages:
                v=stage['validation']
                lines.append(f'| {stage["step"]} | {stage["fixed_loss"]:.6g} | {v["mse"]:.6g} | {v["mae"]:.6g} | {v["esr"]:.6g} |')
            lines.extend(['',f'未訓練演算法的保留設定平均 ESR：{reference["esr"]:.6g}。',
                f'目前保留設定 ESR 最佳步數：{best["step"]}。',
                '', 'best.pt / best-fitted.json 為保留設定 ESR 最佳版本；last.pt / fitted.json 為本次最後完成的版本。',
                '固定 loss 是訓練錄音上的固定取樣評估，與保留設定 ESR 不同。保留設定用於選模型，並非另有獨立測試集；尚未驗證跨素材或完整音檔。'])
            (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            for src,dst in [('last.pt','best.pt'),('fitted.json','best-fitted.json')]:
                shutil.copy2(Path(best['directory'])/src,out/dst)
            return best

        save_summary()
        current=start
        while current<args.target_step:
            target=min(current+args.interval,args.target_step)
            stage_dir=out/f'step-{target}'
            status('training',from_step=current,target_step=target,device='cuda')
            report=train(manifest,stage_dir,circuit,steps=target-current,resume=checkpoint,
                         device='cuda',checkpoint_every=100,**options)
            if report['step']!=target: raise RuntimeError('Unexpected final step')
            if not np.isclose(report['first_loss'],stages[-1]['fixed_loss'],rtol=1e-8,atol=1e-10):
                raise RuntimeError('Resumed fixed loss differs from the previous checkpoint')
            status('evaluating',step=target,validation_records=len(names))
            evaluation=stage_dir/'validation-evaluation.json'
            evaluate_manifest(SimpleNamespace(manifest=validation_manifest,config=stage_dir/'fitted.json',
                output=evaluation,split='validation',substeps=1,audio_dir=stage_dir/'audio-validation'))
            records=json.loads(evaluation.read_text())['records']
            if set(records)!=set(names): raise RuntimeError('Validation record set changed')
            stages.append(dict(step=target,fixed_loss=report['last_loss'],resume_fixed_loss=report['first_loss'],
                validation=average(records),directory=str(stage_dir),training_seconds=report['seconds']))
            checkpoint=stage_dir/'last.pt'
            current=target
            for name in ('last.pt','fitted.json','report.json'):
                shutil.copy2(stage_dir/name,out/name)
            best=save_summary()
            status('stage_complete',step=current,best_step=best['step'],validation=stages[-1]['validation'])
        status('complete',step=current,best_step=best['step'],validation=stages[-1]['validation'])
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main()
