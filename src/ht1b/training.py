"""Inverse PINN training from paired audio; exports physical model parameters."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from .audio import load_manifest, metrics
from .config import Circuit, Controls, write_config
from .physicsnemo_model import CircuitPDE, LearnedCircuit, TrajectoryPINN, residuals, waveform


def _signature(records,circuit):
    data={'circuit':asdict(circuit),'records':[dict(id=r.id,hash=r.metadata['audio_sha256'],
        sr=r.sample_rate,controls=asdict(r.controls),initial=r.initial_state.tolist(),split=r.split)
        for r in records]}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def _checkpoint(path,network,physical,optimizer,step,signature,options,records,rng):
    state={'format_version':1,'step':step,'signature':signature,'options':options,
           'circuit':asdict(physical.base),'network':network.state_dict(),'physical':physical.state_dict(),
           'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
           'numpy_rng':rng.bit_generator.state,
           'record_ids':[r.id for r in records]}
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def export_checkpoint(checkpoint,output):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('format_version') != 1: raise ValueError('Unsupported checkpoint version')
    physical=LearnedCircuit(Circuit(**saved['circuit']))
    physical.load_state_dict(saved['physical'])
    circuit=physical.export()
    write_config(output,circuit,Controls(),{'source_checkpoint':str(checkpoint),'step':saved['step'],
        'model':'reduced circuit; fitted to supplied recordings, not a full CL-1B netlist',
        'controls_note':'Default controls below are not inferred recording settings. Use per-record controls.'})
    return circuit


def train(manifest,output,circuit,steps=500,batch_size=256,width=64,layers=3,
          learning_rate=.001,seed=7,device='cpu',resume=None,physics_weight=1.,data_weight=1.,
          checkpoint_every=100):
    if steps<1 or batch_size<1 or width<1 or layers<1 or checkpoint_every<1:
        raise ValueError('steps, batch_size, width, layers and checkpoint_every must be positive')
    if not np.isfinite([learning_rate,physics_weight,data_weight]).all() or learning_rate<=0 or physics_weight<=0 or data_weight<0:
        raise ValueError('Invalid optimizer/loss settings')
    if device not in ('cpu','cuda'):
        raise ValueError('Use cpu or cuda; this stiff inverse model uses float64 (MPS lacks float64)')
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    all_records=load_manifest(manifest)
    records=[r for r in all_records if r.split=='train']
    if not records: raise ValueError('No train recordings')
    if data_weight>0 and not any(r.wet is not None for r in records):
        raise ValueError('No output data; explicitly set data_weight=0 for physics-only training')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    signature=_signature(all_records,circuit)
    options=dict(width=width,layers=layers,learning_rate=learning_rate,seed=seed,batch_size=batch_size,
                 physics_weight=physics_weight,data_weight=data_weight)
    saved=None
    if resume:
        saved=torch.load(resume,map_location=device,weights_only=True)
        if saved.get('signature') != signature: raise ValueError('Resume data/config signature mismatch')
        if saved.get('options') != options: raise ValueError('Resume training options mismatch')
    elif (output/'last.pt').exists():
        raise ValueError('Output already has a checkpoint; use --resume or another directory')
    network=TrajectoryPINN(len(records),width,layers).to(device=device,dtype=torch.float64)
    physical=LearnedCircuit(circuit).to(device=device,dtype=torch.float64)
    optimizer=torch.optim.Adam(list(network.parameters())+list(physical.parameters()),lr=learning_rate)
    start=0
    if saved:
        network.load_state_dict(saved['network'])
        physical.load_state_dict(saved['physical'])
        optimizer.load_state_dict(saved['optimizer'])
        start=saved['step']
        torch.set_rng_state(saved['torch_rng'].cpu())
        rng.bit_generator.state=saved['numpy_rng']
    equations=[CircuitPDE(circuit,r.controls) for r in records]
    # Anchor to the original config, including after resume.
    prior=torch.stack([v.detach() for v in LearnedCircuit(circuit).raw.values()]).to(device)
    wet_energy=[max(float(np.mean(r.wet**2)),1e-8) if r.wet is not None else 1. for r in records]

    def loss_at(record_index,indices):
        r=records[record_index]
        dt=1/r.sample_rate
        duration=len(r.dry)/r.sample_rate
        t=torch.tensor((indices+1)*dt,dtype=torch.float64,device=device).reshape(-1,1)
        initial=torch.tensor(r.initial_state,dtype=torch.float64,device=device).reshape(1,3)
        states=network(t,record_index,duration,initial)
        prev=network(t-dt,record_index,duration,initial)
        params=physical.values()
        u=torch.tensor(r.dry[indices],dtype=torch.float64,device=device).reshape(-1,1)*circuit.input_volts_per_fs
        residual=residuals(equations[record_index],states,(states-prev)/dt,u,params)
        physics_loss=sum(v.square().mean() for v in residual.values())/len(residual)
        prediction=waveform(states,u,params,r.controls)
        data_loss=torch.zeros((),dtype=torch.float64,device=device)
        if r.wet is not None:
            wet=torch.tensor(r.wet[indices],dtype=torch.float64,device=device).reshape(-1,1)
            # One scale per recording; never normalize dry and wet separately.
            data_loss=(prediction-wet).square().mean()/wet_energy[record_index]
        regularization=(torch.stack(list(physical.raw.values()))-prior).square().mean()
        loss=physics_weight*physics_loss+data_weight*data_loss+1e-4*regularization
        return loss,physics_loss,data_loss

    fixed_indices=[np.linspace(0,len(r.dry)-1,min(512,len(r.dry)),dtype=int) for r in records]
    def audit_loss():
        with torch.no_grad():
            return float(torch.stack([loss_at(j,ids)[0] for j,ids in enumerate(fixed_indices)]).mean())
    first_loss=audit_loss()
    started=time.perf_counter()
    log_path=output/'history.jsonl'
    with log_path.open('a' if saved else 'w') as log:
        for step in range(start+1,start+steps+1):
            j=int(rng.integers(len(records)))
            indices=rng.integers(0,len(records[j].dry),size=batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss,pl,dl=loss_at(j,indices)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(network.parameters())+list(physical.parameters()),10.,error_if_nonfinite=True)
            optimizer.step()
            row=dict(step=step,loss=float(loss.detach()),physics=float(pl.detach()),data=float(dl.detach()))
            log.write(json.dumps(row)+'\n')
            if step==start+1 or step%25==0 or step==start+steps:
                print(json.dumps(row),flush=True)
            if step%checkpoint_every==0 or step==start+steps:
                _checkpoint(output/'last.pt',network,physical,optimizer,step,signature,options,records,rng)
    last_loss=audit_loss()
    fitted=export_checkpoint(output/'last.pt',output/'fitted.json')
    reconstruction={}
    with torch.no_grad():
        for j,r in enumerate(records):
            if r.wet is None: continue
            ids=np.linspace(0,len(r.dry)-1,min(4096,len(r.dry)),dtype=int)
            t=torch.tensor((ids+1)/r.sample_rate,dtype=torch.float64,device=device).reshape(-1,1)
            initial=torch.tensor(r.initial_state,dtype=torch.float64,device=device).reshape(1,3)
            states=network(t,j,len(r.dry)/r.sample_rate,initial)
            u=torch.tensor(r.dry[ids],dtype=torch.float64,device=device).reshape(-1,1)*circuit.input_volts_per_fs
            pred=waveform(states,u,physical.values(),r.controls).cpu().numpy().ravel()
            reconstruction[r.id]=metrics(pred,r.wet[ids])
    report=dict(step=start+steps,first_loss=first_loss,last_loss=last_loss,
                seconds=time.perf_counter()-started,device=device,physicsnemo_version='2.2.2',
                train_records=[r.id for r in records],
                holdout_records=[r.id for r in all_records if r.split!='train'],
                trajectory_reconstruction=reconstruction,
                fitted_parameters=asdict(fitted),
                records_metadata={r.id:r.metadata for r in all_records},
                validation_note='Coordinate PINN reconstruction is not deployment accuracy. Run evaluate with fitted.json on held-out records.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
