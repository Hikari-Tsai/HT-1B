"""Exhaustive lazy CUDA PINN training over every training sample of schema 2.

One epoch is a full pass, not a fixed count of randomly sampled time points.
The coordinate network is a training aid; deployment still uses the circuit.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from physicsnemo.models.mlp.fully_connected import FullyConnected
from ht1b.config import Controls,read_config,write_config
from ht1b.corpus import load_corpus,blocks,batch_bounds,read_block
from ht1b.physicsnemo_model import CircuitPDE,LearnedCircuit,residuals,waveform


def write_json(path,payload):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


class FullTrajectory(nn.Module):
    """Record embeddings avoid allocating a huge one-hot matrix for full batches.

    Absolute recording time and the same embedding continue across I/O blocks.
    No mid-recording reset, no window-specific zero initial states. Coordinates in
    held-out spans are never trained; unknown middle states are latent estimates.
    """
    def __init__(self,records,width=64,layers=3):
        super().__init__()
        self.embedding=nn.Embedding(records,16)
        self.register_buffer('frequencies',2.**torch.arange(17,dtype=torch.float64))
        self.net=FullyConnected(in_features=51,out_features=3,layer_size=width,num_layers=layers,
                               activation_fn='tanh',weight_norm=False)
    def forward(self,t,index,duration):
        phase=2*torch.pi*t/duration*self.frequencies
        ident=self.embedding.weight[index].expand(len(t),-1)
        raw=self.net(torch.cat([t/duration,torch.sin(phase),torch.cos(phase),ident],dim=1))
        bounded=torch.cat([30*torch.sigmoid(raw[:,:1]-6),torch.sigmoid(raw[:,1:]-3)],dim=1)
        return bounded*t/(t+.001)


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    corpus,root=load_corpus(args.manifest,args.dataset)
    circuit,_=read_config(args.config)
    plan=blocks(corpus,seconds=args.block_seconds)
    total_frames=sum(b['stop']-b['start'] for b in plan)
    if total_frames!=corpus['frames_by_split']['train']:raise ValueError('Training plan omitted samples')
    if args.epochs<1 or args.batch_size<1:raise ValueError('Invalid training budget')
    out=args.output;out.mkdir(parents=True,exist_ok=args.resume)
    options=dict(batch_size=args.batch_size,block_seconds=args.block_seconds,seed=args.seed,epochs=args.epochs,
                 network_lr=args.network_lr,physical_lr=args.physical_lr,width=64,layers=3)
    net=FullTrajectory(len(corpus['records'])).to(device='cuda',dtype=torch.float64)
    physical=LearnedCircuit(circuit).to(device='cuda',dtype=torch.float64)
    optimizer=torch.optim.Adam([{'params':net.parameters(),'lr':args.network_lr},
                                {'params':physical.parameters(),'lr':args.physical_lr}])
    prior=torch.stack([v.detach().clone() for v in physical.raw.values()])
    step=0;epoch_start=0;block_cursor=0;covered=0;record_coverage={};group_coverage={}
    if args.resume:
        saved=torch.load(out/'last.pt',map_location='cuda',weights_only=True)
        if saved['manifest_signature']!=corpus['signature'] or saved['options']!=options or saved['circuit']!=asdict(circuit):
            raise ValueError('Resume signature/options mismatch')
        net.load_state_dict(saved['network']);physical.load_state_dict(saved['physical']);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];epoch_start=saved['epoch'];block_cursor=saved['block_cursor'];covered=saved['covered']
        record_coverage=saved['record_coverage'];group_coverage=saved['group_coverage']
    equations={};controls=[Controls(**r['controls']) for r in corpus['records']]
    started=time.perf_counter();session_start_covered=covered;last_save=started;last_values={}
    epoch=epoch_start;cursor=block_cursor
    def status(phase,**extra):
        elapsed=time.perf_counter()-started;processed=covered-session_start_covered
        speed=processed/max(elapsed,1e-6)
        payload=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(),
                     cuda=True,epoch=epoch+1,epochs=args.epochs,step=step,completed_blocks=cursor,blocks_per_epoch=len(plan),
                     frames_covered=covered,frames_planned=total_frames*args.epochs,
                     coverage_percent=100*covered/(total_frames*args.epochs),files_visited=len(record_coverage),
                     files_total=len(corpus['records']),seconds=elapsed,samples_per_second=speed,
                     estimated_remaining_seconds=(total_frames*args.epochs-covered)/speed if speed>0 else None,
                     latest_loss=last_values,peak_cuda_bytes=torch.cuda.max_memory_allocated(),**extra)
        write_json(out/'status.json',payload);print(json.dumps(payload),flush=True)
    def save():
        state=dict(format_version=2,manifest_signature=corpus['signature'],options=options,circuit=asdict(circuit),
                   network=net.state_dict(),physical=physical.state_dict(),optimizer=optimizer.state_dict(),
                   step=step,epoch=epoch,block_cursor=cursor,covered=covered,
                   record_coverage=record_coverage,group_coverage=group_coverage)
        temp=out/'checkpoint.tmp';torch.save(state,temp);temp.replace(out/'last.pt')
        write_config(out/'fitted.json',physical.export(),Controls(),dict(step=step,source='full-corpus inverse PINN',
                     complete_training_pass=covered>=total_frames,manifest_signature=corpus['signature']))
        write_json(out/'coverage.json',dict(frames_covered=covered,frames_per_epoch=total_frames,
                   per_record=record_coverage,per_group=group_coverage,holdout_wet_training_samples=0))
    write_json(out/'run-config.json',dict(options=options,manifest=str(args.manifest.resolve()),
        manifest_signature=corpus['signature'],gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        cuda_version=torch.version.cuda,training_record_count=len(corpus['records']),blocks_per_epoch=len(plan),
        frames_per_epoch=total_frames,hours_per_epoch=total_frames/48000/3600,smoke_blocks=args.smoke_blocks,
        trajectory='Per-record embedding, absolute time; no block reset. Intermediate states are inferred latent states, not measured history.',
        objective='Physics residual + block-energy-normalized waveform MSE + physical prior; all train samples, contiguous microbatches.',
        independent_test_scope=corpus['independence']))
    completed_this_session=0
    try:
        status('training')
        with (out/'history.jsonl').open('a',encoding='utf-8') as log:
            for epoch in range(epoch_start,args.epochs):
                order=np.random.default_rng(args.seed+epoch).permutation(len(plan))
                cursor=block_cursor if epoch==epoch_start else 0
                for position in range(cursor,len(plan)):
                    block=plan[int(order[position])];index=block['record'];row=corpus['records'][index]
                    if index not in equations:equations[index]=CircuitPDE(circuit,controls[index])
                    samples=read_block(root,row,block)
                    tensor=torch.from_numpy(samples).to('cuda')
                    energy=max(float(np.mean(samples[:,1]**2)),1e-8)
                    for a,b in batch_bounds(block,args.batch_size):
                        sl=slice(a-block['start'],b-block['start'])
                        t=(torch.arange(a+1,b+1,device='cuda',dtype=torch.float64)/48000).reshape(-1,1)
                        states=net(t,index,row['frames']/48000);prev=net(t-1/48000,index,row['frames']/48000)
                        params=physical.values();u=tensor[sl,:1]*circuit.input_volts_per_fs
                        residual=residuals(equations[index],states,(states-prev)*48000,u,params)
                        pl=sum(v.square().mean() for v in residual.values())/len(residual)
                        dl=(waveform(states,u,params,controls[index])-tensor[sl,1:2]).square().mean()/energy
                        reg=(torch.stack(list(physical.raw.values()))-prior).square().mean()
                        loss=pl+dl+1e-4*reg
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(physical.parameters()),10.,error_if_nonfinite=True)
                        optimizer.step();step+=1
                        if step%50==0:
                            last_values=dict(total=float(loss.detach()),physics=float(pl.detach()),data=float(dl.detach()))
                            log.write(json.dumps(dict(step=step,record=row['id'],sample_start=a,sample_stop=b,**last_values))+'\n');log.flush()
                    amount=block['stop']-block['start'];covered+=amount
                    record_coverage[row['id']]=record_coverage.get(row['id'],0)+amount
                    group_coverage[str(block['group'])]=group_coverage.get(str(block['group']),0)+amount
                    cursor=position+1;completed_this_session+=1
                    if time.perf_counter()-last_save>=60 or cursor==len(plan) or (args.smoke_blocks and completed_this_session>=args.smoke_blocks):
                        save();status('training');last_save=time.perf_counter()
                    if args.smoke_blocks and completed_this_session>=args.smoke_blocks:
                        status('smoke_complete',note='Infrastructure check only; not a full training pass.');return
                if covered!=(epoch+1)*total_frames:raise RuntimeError('Incomplete epoch coverage')
                save();status('epoch_complete')
        status('training_complete',evaluation_pending=True)
    except BaseException as exc:
        # Save only completed-block checkpoints; partially updated blocks are replayed
        # from the last atomic checkpoint on resume.
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/default.json'))
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=16384)
    p.add_argument('--block-seconds',type=float,default=30);p.add_argument('--network-lr',type=float,default=.001)
    p.add_argument('--physical-lr',type=float,default=.0001);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--smoke-blocks',type=int,default=0);p.add_argument('--resume',action='store_true')
    main(p.parse_args())
