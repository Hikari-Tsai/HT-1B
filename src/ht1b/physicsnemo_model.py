"""PhysicsNeMo 2.x symbolic circuit residuals and inverse-parameter PINN.

The coordinate network fits observed trajectories. Export circuit parameters,
not this recording-specific network, for processing unseen audio.
"""
from dataclasses import asdict, replace
from types import SimpleNamespace
import math
import sympy as sp
import torch
from torch import nn
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from .equations import evaluate


BOUNDS = {
    'gre_span': (1e-6,.02), 'gre_half': (.001,3.),
    'gre_attack_fast': (.0001,.05), 'gre_attack_slow': (.001,.5),
    'gre_release_fast': (.002,2.), 'gre_release_slow': (.02,20.),
    'amp_gain': (.25,12.),
}


class TorchOps:
    @staticmethod
    def maximum(a,b):
        if not torch.is_tensor(a): a=torch.as_tensor(a)
        return torch.maximum(a,torch.as_tensor(b,dtype=a.dtype,device=a.device))
    @staticmethod
    def minimum(a,b):
        if not torch.is_tensor(a): a=torch.as_tensor(a)
        return torch.minimum(a,torch.as_tensor(b,dtype=a.dtype,device=a.device))
    absolute=staticmethod(torch.abs)
    tanh=staticmethod(torch.tanh)
    @staticmethod
    def where(condition,a,b):
        a=torch.as_tensor(a,device=condition.device)
        b=torch.as_tensor(b,device=condition.device)
        return torch.where(condition,a,b)


class SymbolicOps:
    maximum=staticmethod(sp.Max)
    minimum=staticmethod(sp.Min)
    absolute=staticmethod(sp.Abs)
    tanh=staticmethod(sp.tanh)
    @staticmethod
    def where(condition,a,b):
        # PhysicsNeMo 2.2.2 rewrites Piecewise as a free symbol. Heaviside is
        # supported and preserves the strict > branch (release at equality).
        h=sp.Heaviside(condition.lhs-condition.rhs,0)
        return b+(a-b)*h


class CircuitPDE(PDE):
    """Continuous ODE residuals; time derivatives supplied explicitly by caller.

    e: C3 voltage; f/s: GRE occupancies; u: input volts.
    No spatial derivatives exist. PhysicsNeMo make_computations compiles SymPy
    to real differentiable Torch computations (not a local substitute).
    """
    name='HT1BReducedCircuit'
    def __init__(self,circuit,controls):
        self.dim=1
        t=sp.Symbol('t')
        e,f,s=[sp.Function(name)(t) for name in ['e','f','s']]
        u=sp.Symbol('u')
        params=asdict(circuit)
        params.update({name:sp.Symbol(name,positive=True) for name in BOUNDS})
        p=SimpleNamespace(**params)
        result=evaluate((e,f,s),u,p,controls,SymbolicOps)
        self.equations={
            'c3_kcl':(circuit.c3*e.diff(t)-result['ia']+result['ir'])*result['ra']/circuit.opamp_rail,
            'gre_fast':result['tau_fast']*f.diff(t)+f-result['target'],
            'gre_slow':result['tau_slow']*s.diff(t)+s-result['target'],
        }
        self.computations=self.make_computations()


def residuals(pde,states,derivatives,input_volts,params):
    inputs={'u':input_volts}
    for i,name in enumerate(('e','f','s')):
        inputs[name]=states[:,i:i+1]
        inputs[name+'__t']=derivatives[:,i:i+1]
    inputs.update({name:getattr(params,name) for name in BOUNDS})
    outputs={}
    for computation in pde.computations:
        outputs.update(computation.evaluate(inputs))
    return outputs


class LearnedCircuit(nn.Module):
    def __init__(self,circuit):
        super().__init__()
        self.base=circuit
        raw={}
        for name,(lo,hi) in BOUNDS.items():
            value=getattr(circuit,name)
            if not lo < value < hi:
                raise ValueError(f'{name} must be strictly inside training bounds ({lo}, {hi})')
            unit=(value-lo)/(hi-lo)
            raw[name]=nn.Parameter(torch.tensor(math.log(unit/(1-unit)),dtype=torch.float64))
        self.raw=nn.ParameterDict(raw)

    def values(self):
        values=asdict(self.base)
        for name,(lo,hi) in BOUNDS.items():
            values[name]=lo+(hi-lo)*torch.sigmoid(self.raw[name])
        return SimpleNamespace(**values)

    def export(self):
        p=self.values()
        return replace(self.base,**{name:float(getattr(p,name).detach().cpu()) for name in BOUNDS})


class TrajectoryPINN(nn.Module):
    def __init__(self,n_records,width=64,layers=3):
        super().__init__()
        if n_records<1: raise ValueError('n_records must be positive')
        self.n_records=n_records
        self.register_buffer('frequencies',torch.tensor([1,2,4,8,16,32,64],dtype=torch.float64))
        self.net=FullyConnected(in_features=15+n_records,out_features=3,layer_size=width,
                               num_layers=layers,activation_fn='tanh',weight_norm=False)

    def forward(self,t,record_index,duration,initial):
        # t=0 is before the first audio sample; sample i is at (i+1)/fs.
        phase=2*torch.pi*t/duration*self.frequencies
        identity=torch.zeros((len(t),self.n_records),dtype=t.dtype,device=t.device)
        identity[:,record_index]=1
        raw=self.net(torch.cat([t/duration,torch.sin(phase),torch.cos(phase),identity],dim=1))
        bounded=torch.cat([30*torch.sigmoid(raw[:,:1]-6),torch.sigmoid(raw[:,1:]-3)],dim=1)
        gate=t/(t+.001)
        return initial+(bounded-initial)*gate


def waveform(states,input_volts,params,controls):
    r=evaluate(tuple(states[:,i:i+1] for i in range(3)),input_volts,params,controls,TorchOps)
    return r['output']/params.output_volts_per_fs
