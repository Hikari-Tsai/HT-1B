"""Optional compiled manual-mode solver, equivalent to coupled backward Euler.

The GRE implicit updates can be eliminated for a candidate C3 voltage. This
leaves a monotone scalar equation, solved with safeguarded analytic Newton.
Numba is optional; the original SciPy solver remains the reference implementation.
"""
from dataclasses import asdict
import numpy as np
from numba import njit

FIELDS=('front_series','threshold_pot','ratio_pot','gain_pot','c3','r11','attack_pot',
        'r9','release_pot','r10','release_trim','r17','control_pot','supply','opamp_rail',
        'opamp_gain','diode_drop','rectifier_gain','rectifier_offset','pot_taper',
        'control_trim_fraction','gre_dark','gre_span','gre_half','gre_mix',
        'gre_attack_fast','gre_attack_slow','gre_release_fast','gre_release_slow',
        'amp_gain','amp_headroom','input_volts_per_fs','output_volts_per_fs')


@njit(cache=True)
def _candidate(e,old,vin,p,c,dt):
    k=p[12]/(p[11]+p[12])*p[20]
    target=e*k/(e*k+p[23]);dtarget=k*p[23]/(e*k+p[23])**2
    af=dt/(p[25] if target>old[1] else p[27]);a_s=dt/(p[26] if target>old[2] else p[28])
    fast=(old[1]+af*target)/(1+af);slow=(old[2]+a_s*target)/(1+a_s)
    df=af/(1+af)*dtarget;ds=a_s/(1+a_s)*dtarget
    load=1/p[3]+p[21]+p[22]*(p[24]*fast+(1-p[24])*slow)
    dload=p[22]*(p[24]*df+(1-p[24])*ds)
    rr=p[2]*c[1];den=1/p[0]+1/p[1]+load/(1+rr*load)
    fa=(vin/p[0])/den;dfa=-(vin/p[0])/den**2*dload/(1+rr*load)**2
    taper=(10**(p[19]*c[0])-1)/(10**p[19]-1)
    det=p[17]*abs(fa)*taper-p[18]
    ddet=p[17]*(1 if fa>=0 else -1)*dfa*taper if det>0 else 0.
    det=max(det,0.)
    u0=p[15]*(det-e);u=min(max(u0,-p[14]),p[14])
    du=p[15]*(ddet-1) if -p[14]<u0<p[14] else 0.
    ra=p[5]+p[6]*(10**(p[19]*c[2])-1)/(10**p[19]-1)
    release=p[7]+p[8]*c[3];bias=p[9]+p[10];rth=release*bias/(release+bias)
    vth=(u*bias+p[13]*release)/(release+bias)
    ia=max(u-e-p[16],0.)/ra;ir=max(e-vth-p[16],0.)/rth
    dia=(du-1)/ra if u-e-p[16]>0 else 0.
    dir_=(1-du*bias/(release+bias))/rth if e-vth-p[16]>0 else 0.
    f=e-old[0]-dt*(ia-ir)/p[4];derivative=1-dt*(dia-dir_)/p[4]
    output=p[30]*np.tanh(p[29]*10**(c[4]/20)*fa/(1+rr*load)/p[30])/p[32]
    return f,derivative,fast,slow,output


@njit(cache=True)
def _process(samples,p,c,sr,state,substeps):
    y=np.empty(len(samples));dt=1/(sr*substeps);old=state.copy()
    for i in range(len(samples)):
        for _ in range(substeps):
            lo=0.;hi=2*p[13];e=old[0]
            converged=False
            for iteration in range(60):
                f,derivative,fast,slow,output=_candidate(e,old,samples[i]*p[31],p,c,dt)
                if abs(f)<1e-10:converged=True;break
                if f>0:hi=e
                else:lo=e
                candidate=e-f/derivative
                e=candidate if lo<candidate<hi else (lo+hi)/2
            if not converged:raise ValueError('Compiled implicit solve failed to converge')
            old[0]=e;old[1]=fast;old[2]=slow;y[i]=output
    return y,old


class FastCircuitSolver:
    def __init__(self,circuit,controls,sample_rate,substeps=1,initial_state=None):
        if controls.mode!='manual':raise ValueError('Compiled solver currently supports manual mode only')
        if int(sample_rate)!=sample_rate or sample_rate<1 or int(substeps)!=substeps or substeps<1:raise ValueError('Invalid rate/substeps')
        d=asdict(circuit);self.params=np.array([d[k] for k in FIELDS],dtype=float)
        self.controls=np.array([controls.threshold,controls.ratio,controls.attack,controls.release,controls.makeup_db])
        self.rate=sample_rate;self.substeps=substeps
        self.state=np.array([0.,0.,0.] if initial_state is None else initial_state,dtype=float)
        if self.state.shape!=(3,) or not np.isfinite(self.state).all() or np.any(self.state<0) or np.any(self.state>np.array([2*circuit.supply,1,1])):raise ValueError('Invalid state')
    def process(self,samples):
        samples=np.asarray(samples,dtype=float)
        if samples.ndim!=1 or not np.isfinite(samples).all():raise ValueError('Expected finite mono samples')
        y,self.state=_process(samples,self.params,self.controls,self.rate,self.state,self.substeps)
        return y
