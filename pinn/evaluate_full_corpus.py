"""Evaluate exported circuits with continuous dry history and held-out-only scores."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from ht1b.config import Circuit,Controls,read_config
from ht1b.corpus import load_corpus
from ht1b.fast_solver import FastCircuitSolver
from ht1b.audio import metrics
from ht1b.audio_metrics import mrstft,frame_analysis,region_metrics,METRIC_PROTOCOL


def evaluate_record(task):
    root,row,config,split=task
    solver=FastCircuitSolver(Circuit(**config),Controls(**row['controls']),48000)
    spans=[r for r in row['intervals'] if r['split']==split]
    predictions=[[] for _ in spans];targets=[[] for _ in spans];position=0
    with sf.SoundFile(Path(root)/row['path']) as source:
        while position<max(s['stop_sample'] for s in spans):
            block=source.read(480000,dtype='float64',always_2d=True)
            if not len(block):raise ValueError('Unexpected EOF')
            output=solver.process(block[:,0])
            for i,span in enumerate(spans):
                a=max(position,span['start_sample']);b=min(position+len(block),span['stop_sample'])
                if a<b:
                    predictions[i].append(output[a-position:b-position].copy())
                    targets[i].append(block[a-position:b-position,1].copy())
            position+=len(block)
    results=[]
    for span,p,t in zip(spans,predictions,targets):
        p=np.concatenate(p);t=np.concatenate(t)
        if len(p)!=span['stop_sample']-span['start_sample']:raise ValueError('Scoring coverage mismatch')
        values=metrics(p,t);values['mrstft']=mrstft(p,t)['total']
        values['rms_db_mae']=region_metrics(frame_analysis({'reference_wet':t,'prediction':p}),'prediction')['rms_db_mae']
        results.append(dict(**span,samples=len(p),metrics=values))
    total=sum(s['samples'] for s in results)
    # Waveform metrics are pooled by sample/energy; spectral and RMS window scores
    # are duration-weighted across noncontiguous spans without concatenation artifacts.
    mse=sum(s['samples']*s['metrics']['mse'] for s in results)/total
    energy=sum(s['samples']*s['metrics']['target_rms']**2 for s in results)/total
    summary={k:sum(s['samples']*s['metrics'][k] for s in results)/total for k in ('mae','mrstft','rms_db_mae')}
    summary.update(mse=mse,esr=mse/max(energy,1e-12))
    return dict(id=row['id'],split=split,samples=total,metrics=summary,intervals=results)


def evaluate(manifest,dataset,config,split,output,workers=4):
    corpus,root=load_corpus(manifest,dataset);circuit,_=read_config(config)
    if output.exists():raise ValueError('Evaluation output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[r for r in corpus['records'] if any(s['split']==split for s in r['intervals'])]
    records=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(evaluate_record,(str(root),r,asdict(circuit),split)) for r in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            if len(records)%25==0:print(json.dumps(dict(evaluating=split,completed=len(records),total=len(rows))),flush=True)
    records.sort(key=lambda r:r['id'])
    expected=corpus['frames_by_split'][split]
    if sum(r['samples'] for r in records)!=expected:raise ValueError('Incomplete holdout evaluation')
    report=dict(manifest_signature=corpus['signature'],config=str(config),circuit=asdict(circuit),split=split,
                records=records,samples_scored=expected,seconds=time.perf_counter()-started,
                macro_mean={k:float(np.mean([r['metrics'][k] for r in records])) for k in ('mse','mae','esr','mrstft','rms_db_mae')},
                metric_protocol=METRIC_PROTOCOL,history='Continuous dry-only forward simulation from sample 0; no target-state fitting, output gain or delay correction.',
                test_scope=corpus['independence'])
    temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(output)
    print(json.dumps(dict(complete=split,macro_mean=report['macro_mean'],seconds=report['seconds'])),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    evaluate(a.manifest,a.dataset,a.config,a.split,a.output,a.workers)
