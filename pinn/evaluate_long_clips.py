"""Frozen-checkpoint, continuous-history evaluation of the reviewed clip proposal.

No fitting, optimizer, gain matching, or target-dependent alignment. Outputs are
atomic per-record metric checkpoints; the public page changes only on completion.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from ht1b.config import Controls, read_config
from ht1b.fast_solver import FastCircuitSolver
from ht1b.equations import evaluate
from ht1b.physicsnemo_model import TorchOps
from pinn.compare_pretrained import AuthorCL1B
from s6.data import encode_labels
from s6.models import AudioModel
from s6.pinn_model import S6PINN
from s6.inference_scan import accelerate
from s6.representative_experiment import write_json, utc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/long-clip-evaluation-20260929'
DATA=ROOT/'data/tubetech-cl-1b-v1'
COMMON=ROOT/'runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json'
KEYS=('s6','s6_pinn','s4','riccardovib','baseline','mlp','gru')
METRICS=('esr','mse','mae','mrstft','rms_db_mae')
CHUNK=8192

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def score_gpu(pred,target):
    p=torch.as_tensor(np.ascontiguousarray(pred),dtype=torch.float64,device='cuda')
    t=torch.as_tensor(np.ascontiguousarray(target),dtype=torch.float64,device='cuda')
    if p.shape!=t.shape or p.ndim!=1 or not bool(torch.isfinite(p).all() & torch.isfinite(t).all()):
        raise ValueError('Invalid paired audio')
    n=p.numel();e=p-t;energy=t.square().sum();sq=e.square().sum()
    spectral=[]
    for size in (512,1024,2048):
        win=torch.hann_window(size,dtype=p.dtype,device=p.device)
        def mag(x):
            z=torch.stft(x,size,hop_length=size//4,window=win,center=True,pad_mode='reflect',return_complex=True)
            return (z.real.square()+z.imag.square()).clamp_min(1e-8).sqrt()
        ps,ts=mag(p),mag(t)
        spectral.append(torch.linalg.vector_norm(ps-ts)/torch.linalg.vector_norm(ts)+(ps.log()-ts.log()).abs().mean())
    whole=n//480;remain=n%480
    def db(x):return 10*x.square().mean(-1).clamp_min(1e-10).log10()
    rms=(db(p[:whole*480].reshape(-1,480))-db(t[:whole*480].reshape(-1,480))).abs().sum()*480
    if remain:rms=rms+(db(p[-remain:])-db(t[-remain:])).abs()*remain
    values=torch.stack((sq/energy.clamp_min(n*1e-8),sq/n,e.abs().mean(),torch.stack(spectral).mean(),rms/n,energy,sq,e.abs().sum())).cpu().tolist()
    return dict(zip((*METRICS,'target_energy','squared_error','absolute_error'),values))

def metric_parity():
    from ht1b.audio_metrics import mrstft,frame_analysis,region_metrics
    rng=np.random.default_rng(39);results=[]
    for scale in (0.,1e-6,.03):
        t=rng.normal(0,scale,48037);p=t*.9+rng.normal(0,1e-6,len(t))
        actual=score_gpu(p,t);frame=frame_analysis({'reference_wet':t,'prediction':p})
        expected=dict(mse=float(np.mean((p-t)**2)),mae=float(np.abs(p-t).mean()),
          esr=float(np.sum((p-t)**2)/max(float(t@t),len(t)*1e-8)),
          mrstft=mrstft(p,t)['total'],rms_db_mae=region_metrics(frame,'prediction')['rms_db_mae'])
        for k in METRICS:np.testing.assert_allclose(actual[k],expected[k],rtol=2e-9,atol=2e-11)
        results.append(dict(scale=scale,max_abs=max(abs(actual[k]-expected[k]) for k in METRICS)))
    return results

def prepare():
    proposal=read(ROOT/'output/test-candidates/proposal.json')
    manifest=read(ROOT/'runs/full-corpus-gpu-20260925/manifest.json');common=read(COMMON)
    assert proposal['source_manifest_signature']==manifest['signature']==common['manifest_signature']
    sources=common['sources']
    for key in KEYS:assert sha(ROOT/sources[key]['path'])==sources[key]['sha256'],key
    windows=proposal['windows'].copy();candidates=proposal['candidates'].copy()
    # Parent A remains a continuous multi-material score; children are selectable,
    # never pooled back with A (which would double-count its samples).
    for cid,lo,hi,title in [('A1',90,95,'掃頻'),('A2',95,100,'鼓組'),('A3',100,107.7,'Bass')]:
        parent=next(c for c in candidates if c['id']=='A')
        candidates.append(dict(parent,id=cid,start=lo,stop=hi,seconds=hi-lo,title=title,
            content='A 的分素材子區段；聲音標籤為機器推定，待人工聽辨。',parent='A'))
        for w in proposal['windows']:
            if w['candidate']=='A':
                off=round(w['content_offset_seconds']*48000)
                windows.append(dict(w,candidate=cid,score_start_sample=round(lo*48000)+off,
                                    score_stop_sample=round(hi*48000)+off))
    byrecord=defaultdict(list)
    records={r['id']:r for r in manifest['records']}
    for w in windows:
        assert 16<=w['score_start_sample']<w['score_stop_sample']<=records[w['record']]['frames']
        assert w['score_start_sample']%16==w['score_stop_sample']%16==0
        byrecord[w['record']].append(w)
    assert len(byrecord)==620 and len(proposal['windows'])==4835
    signature=hashlib.sha256(json.dumps(windows,sort_keys=True).encode()).hexdigest()
    return proposal,manifest,sources,candidates,windows,byrecord,signature

def load_model(key,sources):
    path=ROOT/sources[key]['path']
    if key=='s6_pinn':
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=S6PINN(**saved['config']['model']);model.load_state_dict(saved['model'])
    elif key=='riccardovib':model=AuthorCL1B(path).float()
    else:model=AudioModel.from_checkpoint(path)
    model=model.cuda().eval()
    if key in ('s6','s6_pinn'):accelerate(model)
    if key=='riccardovib':
        ref=np.load(ROOT/'runs/pretrained-comparison/tensorflow-reference.npz')
        with torch.inference_mode():
            y=model(torch.as_tensor(ref['windows'],dtype=torch.float32,device='cuda'),
                    torch.as_tensor(ref['conditioning'],dtype=torch.float32,device='cuda')).cpu().numpy()
        np.testing.assert_allclose(y,ref['predictions'],rtol=1e-4,atol=1e-5)
    return model

def circuit_record(args):
    record,windows,sources=args
    stop=max(w['score_stop_sample'] for w in windows)
    signal,sr=sf.read(DATA/record['path'],stop=stop,dtype='float64',always_2d=True)
    assert sr==48000 and signal.shape==(stop,2)
    predictions={}
    for key in ('baseline','mlp','gru'):
        circuit,_=read_config(ROOT/sources[key]['path'])
        output=FastCircuitSolver(circuit,Controls(**record['controls']),48000).process(signal[:,0])
        predictions[key]={w['candidate']:output[w['score_start_sample']:w['score_stop_sample']].copy() for w in windows}
    targets={w['candidate']:signal[w['score_start_sample']:w['score_stop_sample'],1].copy() for w in windows}
    return record['id'],predictions,targets

def run(smoke=False):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    folder=OUT.with_name(OUT.name+'-smoke') if smoke else OUT
    folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'running.lock'
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    begun=time.monotonic();last_status=0
    def status(phase,**values):
        nonlocal last_status
        last_status=time.monotonic()
        row=dict(phase=phase,utc=utc(),pid=os.getpid(),elapsed_seconds=last_status-begun,
                 gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,**values)
        write_json(folder/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
    try:
        if (folder/'comparison.json').exists():raise FileExistsError('Completed result already exists')
        proposal,manifest,sources,candidates,windows,byrecord,signature=prepare()
        records=sorted(manifest['records'],key=lambda r:r['id'])
        if smoke:records=records[:1]
        protocol=dict(created_utc=utc(),sources=sources,manifest_signature=manifest['signature'],
            window_signature=signature,proposal_sha256=sha(ROOT/'output/test-candidates/proposal.json'),
            candidates=candidates,windows=windows,files=len(records),methods=list(KEYS),sample_rate=48000,
            esr_floor=1e-8,fft_sizes=[512,1024,2048],state='Continuous Dry from record sample 0; preserve recurrent states; never reset at scoring boundaries',
            riccardovib_history='Native 16 preceding samples per 16 output samples; no additional persistent state',
            padding='S4 final chunk zero-padded to 128 alignment, padded outputs discarded',
            scope='Exploratory reuse. A/B old holdouts; C/D/E training-overlap diagnostics; F/G/H special. No newly independent test.',
            metric_note='Double precision; periodic Hann; center reflect; hop n/4; power floor 1e-8; RMS 480 samples with weighted partial tail and -100 dBFS floor',
            no_fitting=True,smoke=smoke,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
        write_json(folder/'protocol.json',protocol)
        write_json(folder/'frozen-proposal.json',proposal)
        status('metric_parity');write_json(folder/'metric-parity.json',dict(checks=metric_parity()))
        def persist(key,record,predictions,targets):
            result={cid:score_gpu(pred,targets[cid]) for cid,pred in predictions.items()}
            write_json(folder/'metrics'/key/f'{record}.json',result)
        for key in KEYS[:4]:
            status('loading_model',model=key,files_completed=0,files_total=len(records))
            model=load_model(key,sources);phase_start=time.monotonic()
            for offset in range(0,len(records),16):
                batch=records[offset:offset+16];audio=[];predictions=[];targets=[]
                for r in batch:
                    ws=byrecord[r['id']];stop=max(w['score_stop_sample'] for w in ws)
                    pair,sr=sf.read(DATA/r['path'],stop=stop,dtype='float32',always_2d=True)
                    assert sr==48000 and pair.shape==(stop,2)
                    audio.append(pair[:,0].copy())
                    targets.append({w['candidate']:pair[w['score_start_sample']:w['score_stop_sample'],1].copy() for w in ws})
                    predictions.append({w['candidate']:np.empty(w['score_stop_sample']-w['score_start_sample'],dtype=np.float64) for w in ws})
                maximum=max(map(len,audio));controls=[];gains=[]
                for r in batch:
                    c,g=encode_labels(r['labels']);controls.append(c);gains.append(g)
                controls=torch.tensor(controls,dtype=torch.float32,device='cuda');gains=torch.tensor(gains,dtype=torch.float32,device='cuda')
                if key=='s6_pinn':
                    pots=SimpleNamespace(**{k:torch.tensor([r['controls'][k] for r in batch],dtype=torch.float64,device='cuda')[:,None] for k in ('threshold','ratio','attack','release','makeup_db')},mode='manual')
                    params=model.physical.values()
                state=None
                with torch.inference_mode():
                    if key=='riccardovib':
                        calibration=sources[key]['conditioning']
                        for i,r in enumerate(batch):
                            label=r['labels'];c=np.array([label['attack_index']/4,label['release_index']/4,(label['ratio']-2)/8,abs(label['threshold_db'])/40],np.float32)
                            if calibration['threshold_flipped']:c[3]=1-c[3]
                            c=torch.tensor(c[calibration['permutation']],device='cuda')
                            for w in byrecord[r['id']]:
                                start,stop=w['score_start_sample'],w['score_stop_sample']
                                x=torch.as_tensor(audio[i][start-16:stop].copy(),device='cuda').unfold(0,32,16)
                                parts=[model(x[j:j+4096],c.expand(min(4096,len(x)-j),-1)).reshape(-1).cpu().numpy() for j in range(0,len(x),4096)]
                                predictions[i][w['candidate']][:]=np.concatenate(parts)
                    else:
                        for start in range(0,maximum,CHUNK):
                            length=min(CHUNK,maximum-start);length=((length+127)//128)*128
                            x=np.zeros((len(batch),length),np.float32)
                            for i,dry in enumerate(audio):
                                part=dry[start:start+length];x[i,:len(part)]=part
                            x=torch.as_tensor(x,device='cuda')
                            if key=='s6_pinn':
                                latent,state=model.trajectory(x,controls,state)
                                y=evaluate(tuple(latent.unbind(-1)),x.double()*params.input_volts_per_fs,params,pots,TorchOps)['output']/params.output_volts_per_fs
                            else:y,state=model(x,controls,state,gain_db=gains)
                            # Only transfer scored ranges; the prefix still updates every recurrent state.
                            for i,r in enumerate(batch):
                                for w in byrecord[r['id']]:
                                    a,b=max(start,w['score_start_sample']),min(start+length,w['score_stop_sample'])
                                    if a<b:predictions[i][w['candidate']][a-w['score_start_sample']:b-w['score_start_sample']]=y[i,a-start:b-start].cpu().numpy()
                            if time.monotonic()-last_status>30:
                                fraction=(offset+len(batch)*min(1,(start+length)/maximum))/len(records)
                                status('continuous_inference',model=key,files_completed=offset,files_total=len(records),batch_files=len(batch),batch_audio_seconds=min(start+length,maximum)/48000,
                                    model_eta_seconds=(time.monotonic()-phase_start)*(1-fraction)/max(fraction,1e-9))
                for i,r in enumerate(batch):persist(key,r['id'],predictions[i],targets[i])
                done=offset+len(batch);elapsed=time.monotonic()-phase_start
                status('model_progress',model=key,files_completed=done,files_total=len(records),model_eta_seconds=elapsed*(len(records)-done)/done)
                del audio,predictions,targets
            del model;torch.cuda.empty_cache()
        # CPU numerical solvers in two bounded workers; GPU only computes metrics.
        status('circuit_inference',model='baseline/mlp/gru',files_completed=0,files_total=len(records))
        phase_start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={};next_submit=0
            def submit():
                nonlocal next_submit
                r=records[next_submit];pending[next_submit]=pool.submit(circuit_record,(r,byrecord[r['id']],sources));next_submit+=1
            for _ in range(min(2,len(records))):submit()
            for i in range(len(records)):
                record,pred,targets=pending.pop(i).result()
                if next_submit<len(records):submit()
                for key,values in pred.items():persist(key,record,values,targets)
                elapsed=time.monotonic()-phase_start
                status('circuit_inference',model='baseline/mlp/gru',files_completed=i+1,files_total=len(records),model_eta_seconds=elapsed*(len(records)-i-1)/(i+1))
        for key in KEYS:assert sha(ROOT/sources[key]['path'])==sources[key]['sha256']
        studies={}
        for candidate in candidates:
            cid=candidate['id'];rows=[]
            for r in records:
                w=next((w for w in byrecord[r['id']] if w['candidate']==cid),None)
                if w is None:continue
                values={k:read(folder/'metrics'/k/f'{r["id"]}.json')[cid] for k in KEYS}
                energies=[v['target_energy'] for v in values.values()]
                np.testing.assert_allclose(energies,energies[0],rtol=2e-6,atol=1e-16)
                rows.append(dict(id=r['id'],labels=r['labels'],samples=w['score_stop_sample']-w['score_start_sample'],target_energy=energies[0],
                    metrics={k:{m:v[m] for m in METRICS} for k,v in values.items()}))
            if not smoke:assert len(rows)==candidate['files']
            if not rows:continue
            total=sum(r['samples'] for r in rows);energy=sum(r['target_energy'] for r in rows)
            macro={k:{m:float(np.mean([r['metrics'][k][m] for r in rows])) for m in METRICS} for k in KEYS}
            pooled={k:{m:sum(r['metrics'][k][m]*r['samples'] for r in rows)/total for m in METRICS} for k in KEYS}
            for k in KEYS:pooled[k]['esr']=sum(r['metrics'][k]['mse']*r['samples'] for r in rows)/max(energy,total*1e-8)
            studies[cid]=dict(candidate=candidate,records=rows,macro_mean=macro,pooled=pooled,files=len(rows),windows=len(rows),samples_scored=total)
        report=dict(schema_version=1,completed_utc=utc(),protocol=protocol,studies=studies,sources=sources,window_signature=signature,manifest_signature=manifest['signature'])
        write_json(folder/'comparison.json',report)
        if not smoke:
            status('building_page')
            from pinn.build_comparison_page import build
            build(ROOT/'output/cl1b-comparison.html')
        status('smoke_complete' if smoke else 'complete',comparison=str(folder/'comparison.json'))
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(args.smoke)
