"""Command-line audio processing and PhysicsNeMo training workflows."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import numpy as np
from .audio import load_manifest, read_audio, write_audio, metrics
from .config import Circuit, Controls, read_config, write_config
from .solver import CircuitSolver


def add_controls(parser):
    for name in ('threshold','ratio','attack','release','makeup-db'):
        parser.add_argument('--'+name,type=float)
    parser.add_argument('--mode',choices=['fixed','manual','fix-man'])


def resolve_controls(base,args):
    return replace(base,**{name:getattr(args,name) for name in asdict(base)
                           if getattr(args,name,None) is not None})


def simulate(args):
    p,c=read_config(args.config)
    c=resolve_controls(c,args)
    audio,rate=read_audio(args.input)
    if audio.shape[1]>1 and args.channel=='mono':
        raise ValueError('Stereo input requires --channel left or --channel right')
    channel=1 if args.channel=='right' else 0
    if channel>=audio.shape[1]: raise ValueError('Requested channel does not exist')
    if Path(args.input).resolve()==Path(args.output).resolve():
        raise ValueError('Output must not overwrite source audio')
    x=audio[:,channel]
    solver=CircuitSolver(p,c,rate,substeps=args.substeps)
    start=time.perf_counter()
    chunks=[]
    for begin in range(0,len(x),4096):
        chunks.append(solver.process(x[begin:begin+4096]))
        if begin and begin//4096%50==0:
            print(f'Processed {begin}/{len(x)} samples',file=sys.stderr,flush=True)
    y=np.concatenate(chunks)
    peak=write_audio(args.output,y,rate)
    report=dict(samples=len(y),sample_rate=rate,seconds=time.perf_counter()-start,
                max_scaled_residual=solver.max_residual,final_state=solver.state.tolist(),
                circuit=asdict(p),controls=asdict(c),**peak)
    Path(str(args.output)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def demo_data(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('Demo directory already contains a manifest')
    if args.sample_rate<1000 or args.seconds<=0 or not np.isfinite(args.seconds):
        raise ValueError('Use sample-rate >= 1000 and positive finite seconds')
    count=round(args.sample_rate*args.seconds)
    if count<2: raise ValueError('Duration too short')
    t=np.arange(count)/args.sample_rate
    p,c=Circuit(),Controls()
    records=[]
    for name,freq,level,phase in [('train',137,.3,0),('validation',211,.2,.3)]:
        envelope=(t>.05*args.seconds)*(t<.65*args.seconds)
        x=level*envelope*(.8*np.sin(2*np.pi*freq*t+phase)+.2*np.sin(2*np.pi*freq*1.7*t))
        y=CircuitSolver(p,c,args.sample_rate).process(x)
        write_audio(out/(name+'.wav'),np.column_stack([x,y]),args.sample_rate)
        records.append(dict(id=name,stereo_pair=name+'.wav',split=name,controls=asdict(c),
                            initial_state=[0,0,0],metadata={'source':'SYNTHETIC HT-1B reduced solver; not hardware'}))
    (out/'manifest.json').write_text(json.dumps({'schema_version':1,'records':records},indent=2)+'\n')
    write_config(out/'truth.json',p,c,{'source':'synthetic generator'})
    print(out/'manifest.json')


def evaluate_manifest(args):
    p,_=read_config(args.config)
    records=load_manifest(args.manifest)
    report={'model':'reduced-circuit direct solver','records':{}}
    for r in records:
        if args.split!='all' and r.split!=args.split: continue
        start=time.perf_counter()
        solver=CircuitSolver(p,r.controls,r.sample_rate,substeps=args.substeps,initial_state=r.initial_state)
        y=solver.process(r.dry)
        row=metrics(y,r.wet) if r.wet is not None else {'peak':float(np.max(abs(y)))}
        row.update(seconds=time.perf_counter()-start,max_scaled_residual=solver.max_residual,
                   metadata=r.metadata)
        report['records'][r.id]=row
        if args.audio_dir:
            safe_id=''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in r.id)
            write_audio(Path(args.audio_dir)/(safe_id+'.wav'),y,r.sample_rate)
    if not report['records']: raise ValueError('No recordings match selected split')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description='HT-1B reduced circuit / PhysicsNeMo research tools')
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('simulate',help='Offline implicit circuit solver -> FLOAT WAV')
    s.add_argument('input');s.add_argument('output');s.add_argument('--config')
    s.add_argument('--channel',choices=['mono','left','right'],default='mono')
    s.add_argument('--substeps',type=int,default=1);add_controls(s)
    d=sub.add_parser('demo-data',help='Generate labeled synthetic L/R training and validation recordings')
    d.add_argument('output');d.add_argument('--sample-rate',type=int,default=8000)
    d.add_argument('--seconds',type=float,default=.25)
    e=sub.add_parser('evaluate',help='Evaluate the deployed circuit solver, not PINN reconstruction')
    e.add_argument('manifest');e.add_argument('--config');e.add_argument('--output',required=True)
    e.add_argument('--split',choices=['train','validation','test','all'],default='validation')
    e.add_argument('--audio-dir');e.add_argument('--substeps',type=int,default=1)
    t=sub.add_parser('train',help='PhysicsNeMo inverse PINN using audio manifest')
    t.add_argument('manifest');t.add_argument('--output',required=True);t.add_argument('--config')
    for name,default in [('steps',500),('batch-size',256),('width',64),('layers',3),('seed',7),('checkpoint-every',100)]:
        t.add_argument('--'+name,type=int,default=default)
    t.add_argument('--learning-rate',type=float,default=.001)
    t.add_argument('--physics-weight',type=float,default=1.)
    t.add_argument('--data-weight',type=float,default=1.)
    t.add_argument('--device',choices=['cpu','cuda'],default='cpu');t.add_argument('--resume')
    prep=sub.add_parser('prepare',help='Build manifest from CL-1B stereo-pair filenames')
    prep.add_argument('files',nargs='+');prep.add_argument('--output',required=True);prep.add_argument('--config')
    prep.add_argument('--mode',choices=['fixed','manual','fix-man'],default='manual')
    prep.add_argument('--seconds',type=float);prep.add_argument('--delay-samples',type=int,default=0)
    exp=sub.add_parser('export',help='Export identified circuit parameters from a checkpoint')
    exp.add_argument('checkpoint');exp.add_argument('output')
    args=parser.parse_args(argv)
    try:
        if args.command=='simulate': simulate(args)
        elif args.command=='demo-data': demo_data(args)
        elif args.command=='evaluate': evaluate_manifest(args)
        elif args.command=='prepare':
            from .dataset import prepare_manifest
            p,_=read_config(args.config)
            print(prepare_manifest(args.files,args.output,p,args.mode,args.seconds,args.delay_samples))
        elif args.command=='export':
            from .training import export_checkpoint
            export_checkpoint(args.checkpoint,args.output)
        elif args.command=='train':
            from .training import train
            p,_=read_config(args.config)
            options={k:v for k,v in vars(args).items() if k not in ('command','manifest','output','config')}
            report=train(args.manifest,args.output,p,**options)
            print(json.dumps({'report':str(Path(args.output)/'report.json'),
                              'first_loss':report['first_loss'],'last_loss':report['last_loss']},indent=2))
    except (ValueError,OSError,RuntimeError) as error:
        parser.exit(2,f'ht1b: {error}\n')
