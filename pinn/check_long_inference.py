"""Numerical parity and runtime probes; no checkpoint updates."""
import copy,json,time
from pathlib import Path
import torch
from s6.layers import SelectiveSSM
from s6.models import AudioModel
from s6.pinn_model import S6PINN
from s6.inference_scan import accelerate

ROOT=Path(__file__).resolve().parents[1]

def check():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(31)
    report={}
    with torch.inference_mode():
        layer=SelectiveSSM(16,16).cuda().eval()
        fast=accelerate(copy.deepcopy(layer))
        u=torch.randn(3,513,16,device='cuda')*.2
        state=torch.randn(3,16,16,device='cuda')*.01
        expected,es=layer(u,state);actual,as_=fast(u,state)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(as_,es,atol=2e-6,rtol=2e-4)
        first,ss=fast(u[:,:256],state);last,ls=fast(u[:,256:],ss)
        torch.testing.assert_close(torch.cat((first,last),1),actual,atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(ls,as_,atol=2e-6,rtol=2e-4)
        report['selective_scan_max_abs']=float((expected-actual).abs().max())
        for key in ('s6','s6_pinn','s4'):
            if key=='s6_pinn':
                saved=torch.load(ROOT/'runs/s6-tfilm-pinn-gpu-20260929/best.pt',map_location='cpu',weights_only=True)
                model=S6PINN(**saved['config']['model']).cuda().eval();model.load_state_dict(saved['model'])
            else:
                model=AudioModel.from_checkpoint(ROOT/f'runs/s4-s6-representative-gpu-20260925/{key}_tfilm-best.mdlus').cuda().eval()
            dry=torch.randn(2,2048,device='cuda')*.03;controls=torch.rand(2,4,device='cuda')
            call=lambda m,x,s=None:m.trajectory(x,controls,s) if key=='s6_pinn' else m(x,controls,s)
            expected,es=call(model,dry)
            fast=accelerate(copy.deepcopy(model)) if key!='s4' else model
            actual,fs=call(fast,dry)
            torch.testing.assert_close(actual,expected,atol=5e-6,rtol=3e-4)
            first,ss=call(fast,dry[:,:1024]);last,_=call(fast,dry[:,1024:],ss)
            torch.testing.assert_close(torch.cat((first,last),1),actual,atol=2e-5,rtol=5e-4)
            report[key]={'checkpoint_max_abs':float((actual-expected).abs().max())}
            for batch,length in [(16,4096),(32,8192)]:
                x=torch.randn(batch,length,device='cuda')*.03;controls=torch.rand(batch,4,device='cuda')
                _,state=call(fast,x);torch.cuda.synchronize()
                begun=time.perf_counter()
                for _ in range(5):_,state=call(fast,x,state)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begun
                report[key][f'{batch}x{length}']=dict(seconds=elapsed/5,samples_per_second=batch*length*5/elapsed)
            del model,fast
            torch.cuda.empty_cache()
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':check()
