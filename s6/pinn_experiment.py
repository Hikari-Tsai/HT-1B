"""GPU S6/TFiLM + shared circuit PINN; fixed representative-window protocol.

Queues behind explicitly identified existing jobs. After training, reloads the
best checkpoint, computes the same five metrics as the six-method report, and
builds the offline comparison page. Never touches the test split.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .pinn_model import S6PINN
from .losses import AudioLoss
from .training import load_config
from .representative_experiment import windows, load_batch, write_json, utc, WARMUP, SCORE, WINDOW

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/s6-tfilm-pinn-gpu-20260929'
BASE = ROOT / 'runs/six-model-common-validation-20260925'
MANIFEST = ROOT / 'runs/full-corpus-gpu-20260925/manifest.json'


def process_identity(pid):
    try:
        # field 22, with command name (possibly containing spaces) removed.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def pots_for(rows, originals, device):
    keys = ('threshold','ratio','attack','release','makeup_db')
    controls = [originals[r['id']]['controls'] for r in rows]
    if any(c['mode'] != 'manual' for c in controls):
        raise ValueError('This circuit experiment requires manual controls')
    return torch.tensor([[c[k] for k in keys] for c in controls],device=device,dtype=torch.float64)


def get_batch(rows, originals, device):
    dry, wet, controls, _ = load_batch(rows, device)
    return dry, wet, controls, pots_for(rows, originals, device)


def losses(model, batch, audio_loss):
    dry, wet, controls, pots = batch
    predicted, residual = model.predict_window(dry, controls, pots)
    audio, parts = audio_loss(predicted.float(), wet[:, WARMUP:])
    physics = sum(v.square().mean() for v in residual.values())/len(residual)
    prior = model.prior_loss()
    total = audio+physics+1e-4*prior
    detail = dict(audio=float(audio.detach()),physics=float(physics.detach()),
                  prior=float(prior.detach()), **{k:float(v.detach()) for k,v in parts.items()},
                  **{k:float(v.square().mean().detach()) for k,v in residual.items()})
    return total, detail


def optimizer_for(model, options):
    physics_ids = {id(p) for p in model.physical.parameters()}
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in physics_ids],lr=options['learning_rate']),
        dict(params=model.physical.parameters(),lr=1e-4)],weight_decay=options['weight_decay'])


@torch.no_grad()
def validate(model, rows, originals, device, batch_size, progress, full_metrics=False):
    from pinn.evaluate_five_models import score, summarize, METRICS
    model.eval()
    grouped = defaultdict(list)
    scores = []
    physics_sums = defaultdict(float)
    for offset in range(0,len(rows),batch_size):
        selected = rows[offset:offset+batch_size]
        dry, wet, controls, pots = get_batch(selected,originals,device)
        predicted, residual = model.predict_window(dry,controls,pots)
        if not torch.isfinite(predicted).all() or any(not torch.isfinite(v).all() for v in residual.values()):
            raise FloatingPointError('Nonfinite validation output or physics residual')
        for k,v in residual.items():
            physics_sums[k] += float(v.square().sum())
        for row,p,t in zip(selected,predicted.cpu().numpy(),wet[:,WARMUP:].cpu().numpy()):
            if full_metrics:
                item = score(p,t)
            else:
                p,t = p.astype(np.float64),t.astype(np.float64)
                error = p-t
                item = dict(squared_error=float(error@error), absolute_error=float(np.abs(error).sum()),
                            target_energy=float(t@t), mrstft=0.,rms_db_mae=0.)
            scores.append(item)
            grouped[row['id']].append(item)
        progress(min(offset+batch_size,len(rows)),len(rows))
    records = {name:summarize(values) for name,values in grouped.items()}
    metrics = METRICS if full_metrics else ('esr','mse','mae')
    return dict(macro_mean={k:float(np.mean([v[k] for v in records.values()])) for k in metrics},
                pooled={k:v for k,v in summarize(scores).items() if k in metrics},
                physics_mse={k:v/(len(rows)*SCORE) for k,v in physics_sums.items()},
                records=records, intervals=scores, files=len(records),windows=len(rows))


def checkpoint(path,model,optimizer,config,epoch,manifest_signature):
    temp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
                    epoch=epoch,manifest_signature=manifest_signature),temp)
    temp.replace(path)


def add_common_comparison(output, result, rows, selected_epoch, manifest_signature):
    from pinn.evaluate_five_models import METRICS, sha
    report = json.loads((BASE/'comparison.json').read_text())
    original_report = json.loads((BASE/'comparison.json').read_text())
    index = json.loads((BASE/'windows.json').read_text())
    generated = [dict(id=r['id'], interval=r['interval'],warmup_start=r['start'],
        start_sample=r['start']+WARMUP,stop_sample=r['stop']) for r in rows]
    signature = hashlib.sha256(json.dumps(generated,sort_keys=True).encode()).hexdigest()
    if (generated != index['windows'] or signature != report['window_signature'] or
            manifest_signature != report['manifest_signature']):
        raise ValueError('New evaluation does not match the existing common windows')
    for row,values in zip(report['intervals'],result['intervals']):
        row['metrics']['s6_pinn'] = {k:values[k] for k in METRICS}
    for row in report['records']:
        energy = sum(v['target_energy'] for r,v in zip(rows,result['intervals']) if r['id'] == row['id'])
        np.testing.assert_allclose(energy,row['target_energy'],rtol=1e-12,atol=1e-12)
        row['metrics']['s6_pinn'] = result['records'][row['id']]
    report['macro_mean']['s6_pinn'] = result['macro_mean']
    report['pooled']['s6_pinn'] = result['pooled']
    report['sources']['s6_pinn'] = dict(path=str((output/'best.pt').relative_to(ROOT)),
        sha256=sha(output/'best.pt'),selected_epoch=selected_epoch,
        method='S6/TFiLM latent e/f/s + shared reduced-circuit ODE residual; direct neural trajectory inference; 4096 Dry warmup',
        protocol=str((output/'protocol.json').relative_to(ROOT)))
    report['s6_pinn_added_utc'] = utc()
    report['parent_comparison_sha256'] = sha(BASE/'comparison.json')
    # Existing six results remain byte-for-value equivalent.
    for old,new in zip(original_report['records'],report['records']):
        assert all(new['metrics'][key] == value for key,value in old['metrics'].items())
    write_json(output/'common-validation/windows.json',index)
    write_json(output/'common-validation/comparison.json',report)


def run(output=RUN, epochs=16, batch_size=256, wait_pids=(), start_queued=False):
    if output.exists() and any(output.iterdir()):
        if not start_queued:
            raise FileExistsError(f'Output directory already in use: {output}')
        previous = json.loads((output/'status.json').read_text())
        if (previous['phase'] != 'queued_for_gpu' or previous['epoch'] != 0 or
                process_identity(previous['pid']) is not None or
                {p.name for p in output.iterdir()} != {'status.json','protocol.json','queue.json'}):
            raise RuntimeError('Only a stopped, never-trained queue may be started directly')
        archive = output/'previous-queue'
        archive.mkdir()
        for name in ('status.json','protocol.json','queue.json'):
            shutil.copy2(output/name,archive/name)
    output.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    state = dict(phase='preparing',utc=utc(),pid=os.getpid(),architecture='s6_tfilm_pinn',
        epoch=0,epochs=epochs,train_intervals_seen=0,train_intervals_total=8680,
        validation_intervals_seen=0,validation_intervals_total=1240,batch_loss=None,eta_seconds=None)
    def update(**changes):
        state.update(changes,utc=utc(),elapsed_seconds=round(time.monotonic()-begun,1))
        write_json(output/'status.json',state)
    try:
        manifest = json.loads(MANIFEST.read_text())
        originals = {r['id']:r for r in manifest['records']}
        config = load_config(ROOT/'s6/configs/neural-s6-tfilm.json')
        options = config['training']
        train_probe = windows(manifest,ROOT/'data/tubetech-cl-1b-v1','train',0,epochs)
        validation = windows(manifest,ROOT/'data/tubetech-cl-1b-v1','validation',0,epochs)
        if epochs != 16 or len(train_probe) != 8680 or len(validation) != 1240:
            raise ValueError('Expected the fixed 16-epoch representative-window protocol')
        protocol = dict(manifest_signature=manifest['signature'],epochs=epochs,batch_size=batch_size,
            seed=7,architecture='S6 + TFiLM + PINN',config=config,
            warmup_samples=WARMUP,scored_samples_per_window=SCORE,window_samples=WINDOW,
            train_intervals=8680,validation_intervals=1240,train_files=620,validation_files=620,
            train_frames_in_full_manifest=manifest['frames_by_split']['train'],
            train_input_samples_per_epoch=len(train_probe)*WINDOW,train_scored_samples_per_epoch=len(train_probe)*SCORE,
            validation_scored_samples=len(validation)*SCORE,initialization='fresh network and original circuit; no pretrained weights',
            objective='AudioLoss (L1 + 0.1 ESR + 0.1 MR-STFT) + mean normalized C3/GRE residual MSE + 0.0001 physical prior',
            physical_learning_rate=1e-4,physics_precision='float64; backbone float32',
            equations='shared ht1b.physicsnemo_model.circuit_residual_terms, also used by MLP/GRU CircuitPDE',
            controls='network: normalized labels; physics: manifest physical pots, manual mode, makeup applied once inside circuit',
            state='reset neural hidden state per window; Dry-only warmup; preceding latent e/f/s carried into first scored sample; no zero physical state imposed mid-record',
            limitations=['85.33 ms warmup cannot establish long release history','sampled training, not full sample coverage',
                'reused validation for selection; no new independent test','output head and objective change together; not a loss-only ablation',
                'direct neural state inference differs from prior MLP/GRU fitted-parameter solver deployment'],
            split_scope='validation_model_selection_only',sampling='same windows, positions and shuffle as previous S6, 16 epochs')
        write_json(output/'protocol.json',protocol)
        identities = {str(pid):process_identity(pid) for pid in wait_pids}
        identities = {pid:start for pid,start in identities.items() if start is not None}
        write_json(output/'queue.json',dict(wait_for=identities,created_utc=utc()))
        if start_queued:
            write_json(output/'start-now.json',dict(utc=utc(),reason='User requested immediate shared-GPU training',
                prior_queue_pid=previous['pid'],other_jobs_untouched=True))
        while True:
            active = [pid for pid,start in identities.items() if process_identity(pid) == start]
            if not active:
                break
            update(phase='queued_for_gpu',waiting_pids=active)
            time.sleep(30)
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; CPU training is prohibited')
        device = torch.device('cuda')
        protocol.update(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
        write_json(output/'protocol.json',protocol)
        update(phase='gpu_smoke_test',gpu=protocol['gpu'],cuda=protocol['cuda'],waiting_pids=[])
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        model = S6PINN(**config['model']).to(device)
        audio_loss = AudioLoss(options['fft_sizes'],options['l1_weight'],options['esr_weight'],options['spectral_weight'])
        optimizer = optimizer_for(model,options)
        probe = get_batch(train_probe[:4],originals,device)
        smoke = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = losses(model,probe,audio_loss)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite GPU smoke loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),options['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            smoke.append(dict(loss=float(loss.detach()),gradient_norm=float(norm),parts=parts))
        write_json(output/'gpu-smoke.json',dict(gpu=protocol['gpu'],cuda=protocol['cuda'],steps=smoke))
        del optimizer,model,probe
        torch.cuda.empty_cache()
        # Smoke samples never contaminate the official model initialization.
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        model = S6PINN(**config['model']).to(device)
        optimizer = optimizer_for(model,options)
        history=[]; best=float('inf')
        for epoch in range(epochs):
            started=time.monotonic()
            rows=windows(manifest,ROOT/'data/tubetech-cl-1b-v1','train',epoch,epochs)
            random.Random(7+epoch).shuffle(rows)
            model.train(); sums=defaultdict(float)
            update(phase='training',epoch=epoch+1,train_intervals_seen=0,validation_intervals_seen=0)
            for offset in range(0,len(rows),batch_size):
                selected=rows[offset:offset+batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss,parts=losses(model,get_batch(selected,originals,device),audio_loss)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),options['grad_clip'],error_if_nonfinite=True)
                optimizer.step()
                value=float(loss.detach())
                for key,v in dict(total=value,**parts).items(): sums[key]+=v*len(selected)
                seen=offset+len(selected)
                remain=(len(rows)-seen)/seen*(time.monotonic()-started)
                update(train_intervals_seen=seen,batch_loss=value,batch_loss_parts=parts,eta_seconds=round(remain,1),
                       eta_scope='remaining training batches in current epoch only')
            update(phase='validation',eta_seconds=None)
            result=validate(model,validation,originals,device,batch_size,
                lambda seen,total:update(validation_intervals_seen=seen))
            current=result['macro_mean']['esr']
            if current < best:
                best=current
                checkpoint(output/'best.pt',model,optimizer,config,epoch+1,manifest['signature'])
                write_json(output/'best-validation.json',dict(selected_epoch=epoch+1,**{
                    k:v for k,v in result.items() if k not in ('records','intervals')}))
            checkpoint(output/'last.pt',model,optimizer,config,epoch+1,manifest['signature'])
            history.append(dict(epoch=epoch+1,train_mean_losses={k:v/len(rows) for k,v in sums.items()},
                validation_macro_mean=result['macro_mean'],validation_pooled=result['pooled'],
                validation_physics_mse=result['physics_mse'],best_esr=best,epoch_seconds=round(time.monotonic()-started,1)))
            write_json(output/'history.json',dict(history=history))
            update(phase='epoch_complete',best_validation_esr=best,
                   eta_seconds=round(np.mean([h['epoch_seconds'] for h in history])*(epochs-epoch-1),1),
                   eta_scope='remaining training and per-epoch validation; excludes final report build')
            print(json.dumps(history[-1]),flush=True)
        update(phase='final_common_validation',validation_intervals_seen=0,eta_seconds=None)
        saved=torch.load(output/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'])
        result=validate(model,validation,originals,device,batch_size,
            lambda seen,total:update(validation_intervals_seen=seen),full_metrics=True)
        np.testing.assert_allclose(result['macro_mean']['esr'],best,rtol=1e-6,atol=1e-10)
        write_json(output/'final-validation.json',dict(selected_epoch=saved['epoch'],**result))
        add_common_comparison(output,result,validation,saved['epoch'],manifest['signature'])
        update(phase='building_page')
        from pinn.build_comparison_page import build
        build(ROOT/'output/cl1b-comparison.html')
        update(phase='complete',selected_epoch=saved['epoch'],macro_mean=result['macro_mean'],pooled=result['pooled'])
        print(json.dumps(state),flush=True)
    except Exception as exc:
        update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN)
    parser.add_argument('--wait-pids',type=int,nargs='*',default=[])
    parser.add_argument('--start-queued',action='store_true',help='Start an explicitly stopped, untrained queue immediately')
    args=parser.parse_args()
    run(args.output,wait_pids=args.wait_pids,start_queued=args.start_queued)
