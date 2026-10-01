"""Inference-only fused selective recurrence; unchanged checkpoint parameters.

The time recurrence and initial/final states match SelectiveSSM.forward.
No training calls are redirected. Validate output/state parity before evaluation.
"""
import types
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from .layers import SelectiveSSM


@triton.jit
def _scan(U,B,C,DT,A,D,S,Y,END,T:tl.constexpr,H:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
    bh=tl.program_id(0)
    batch=bh//H;h=bh%H
    ns=tl.arange(0,K)
    a=tl.load(A+h*N+ns,ns<N,0.)
    skip=tl.load(D+h)
    state=tl.load(S+bh*N+ns,ns<N,0.)
    for t in range(T):
        u=tl.load(U+(batch*T+t)*H+h)
        dt=tl.load(DT+(batch*T+t)*H+h)
        b=tl.load(B+(batch*T+t)*2*N+ns,ns<N,0.)
        c=tl.load(C+(batch*T+t)*2*N+ns,ns<N,0.)
        # Preserve separate multiply/add operations, as in the reference scan.
        state=tl.exp(dt*a)*state+dt*b*u
        y=tl.sum(state*c,0)+skip*u
        tl.store(Y+(batch*T+t)*H+h,y)
    tl.store(END+bh*N+ns,state,ns<N)


def fused_forward(self,u,state=None):
    if torch.is_grad_enabled() or u.device.type!='cuda' or u.dtype!=torch.float32:
        raise RuntimeError('Fused scan is for CUDA float32 inference only')
    u=u.contiguous()
    b,c=self.bc(u).chunk(2,dim=-1)
    dt=F.softplus(self.dt(u)).contiguous()
    a=-self.log_a.exp()
    batch,length,width=u.shape
    if state is None:state=u.new_zeros(batch,width,self.state_dim)
    state=state.contiguous()
    y=torch.empty_like(u);end=torch.empty_like(state)
    _scan[(batch*width,)](u,b,c,dt,a,self.skip,state,y,end,length,width,self.state_dim,
        triton.next_power_of_2(self.state_dim),num_warps=1,enable_fp_fusion=False)
    return y,end


def accelerate(model):
    for layer in model.modules():
        if isinstance(layer,SelectiveSSM):
            layer.forward=types.MethodType(fused_forward,layer)
    return model
