"""Differentiable reduced-circuit integrator for S6-DiffCircuit (hard constraint).

Same manual-mode equations as ht1b.equations and ht1b.fast_solver, integrated
with backward Euler; e/f/s are computed by the circuit, never predicted.
Design: s6/docs/plans/2026-11-10-circuit-integrated-hybrid-design.md.

Approximations relative to fast_solver (validated by tools/validate_integrator.py):
- The control path (e, f, s) is updated once every `divisor` audio samples.
  The detector uses the block peak of |Dry| through the front network at the
  conductance of the previous control step (a one-step lag; fast_solver is
  fully implicit per sample).
- The audio path uses the GRE conductance interpolated linearly across each
  block, ending at the new state as fast_solver's per-sample output does.

Gradients of the implicit C3 update use one Newton step from the converged
root (implicit-function theorem); the bracketing iterations run without autograd.
"""
from dataclasses import asdict
import math
from types import SimpleNamespace

import torch
from torch import nn

from ht1b.config import Circuit
from ht1b.physicsnemo_model import BOUNDS

# Physical parameters learned with a prior; the threshold pot law is unknown hardware.
LEARNABLE = {**BOUNDS, 'pot_taper': (.5, 4.), 'rectifier_gain': (.5, 8.), 'rectifier_offset': (.001, .2)}
POTS = ('threshold', 'ratio', 'attack', 'release', 'makeup_db')


def taper(position, decades):
    """Assumed logarithmic pot law, as ht1b.equations.taper."""
    return (torch.pow(10., decades*position)-1)/(torch.pow(10., decades)-1)


def c3_residual(e, e0, det, ra, rth, rel, bias, gain, rail, supply, drop, c3, dt):
    """Backward-Euler residual of C3·de/dt = iA − iR and its derivative in e (fast_solver._candidate)."""
    u0 = gain*(det-e)
    u = u0.clamp(-rail, rail)
    du = torch.where(u0.abs() < rail, -gain, 0.)
    vth = (u*bias+supply*rel)/(rel+bias)
    charge, discharge = u-e-drop, e-vth-drop
    ia, ir = charge.clamp_min(0)/ra, discharge.clamp_min(0)/rth
    dia = torch.where(charge > 0, (du-1)/ra, 0.)
    dir_ = torch.where(discharge > 0, (1-du*bias/(rel+bias))/rth, 0.)
    return e-e0-dt*(ia-ir)/c3, 1-dt*(dia-dir_)/c3


def _newton_loop(e0, det, ra, rth, rel, bias, gain, rail, supply, drop, c3, dt, iterations: int):
    lo, hi = torch.zeros_like(e0), torch.full_like(e0, 2*supply)
    e = e0.clamp(0, 2*supply)
    for _ in range(iterations):
        g, dg = c3_residual(e, e0, det, ra, rth, rel, bias, gain, rail, supply, drop, c3, dt)
        lo, hi = torch.where(g < 0, e, lo), torch.where(g > 0, e, hi)
        step = e-g/dg
        e = torch.where((step > lo) & (step < hi), step, (lo+hi)/2)
    return e


def control_step(e, f, s, peak, cdet, catt, crel, gain_thr, rr, ra0, rel0, offset, span, half,
                 attack_fast, attack_slow, release_fast, release_slow, fixed: tuple, iterations: int):
    """One control step: detector -> backward-Euler C3 -> backward-Euler GRE. Returns e, f, s, conductance."""
    gain_pot, front_series, threshold_pot, dark, mix, bias, trim, gain, rail, supply, drop, c3, dt = fixed
    load = 1/gain_pot+dark+span*(mix*f+(1-mix)*s)
    a = (1/front_series)/(1/front_series+1/threshold_pot+load/(1+rr*load))
    det = (gain_thr*torch.exp(cdet)*peak*a-offset).clamp_min(0)
    ra, rel = ra0*torch.exp(catt), rel0*torch.exp(crel)
    rth = rel*bias/(rel+bias)
    k = (gain, rail, supply, drop, c3, dt)
    with torch.no_grad():
        root = _newton_loop(e.detach(), det.detach(), ra.detach(), rth.detach(), rel.detach(), bias, *k, iterations)
    g, dg = c3_residual(root, e, det, ra, rth, rel, bias, *k)
    e = root-g/dg.detach()  # implicit-function gradient through the converged root
    drive = e.clamp_min(0)*trim
    target = drive/(drive+half)
    af = dt/torch.where(target > f, attack_fast, release_fast)
    a_s = dt/torch.where(target > s, attack_slow, release_slow)
    f, s = (f+af*target)/(1+af), (s+a_s*target)/(1+a_s)
    return e, f, s, dark+span*(mix*f+(1-mix)*s)


UNROLL = 8  # control steps fused into one compiled call


def control_block(e, f, s, peak, cdet, catt, crel, *rest):
    """UNROLL consecutive control steps; per-step inputs are [B, n] columns."""
    out = []
    for i in range(peak.shape[1]):
        e, f, s, g = control_step(e, f, s, peak[:, i], cdet[:, i], catt[:, i], crel[:, i], *rest)
        out.append(torch.stack((e, f, s, g), -1))
    return torch.stack(out, 1)


_compiled = None


def _block(*args):
    """Fused (forward and backward) by torch.compile on CUDA; S6_COMPILE_CIRCUIT=0 disables."""
    global _compiled
    import os
    if args[0].is_cuda and os.environ.get('S6_COMPILE_CIRCUIT', '1') != '0':
        if _compiled is None:
            _compiled = torch.compile(control_block)
        return _compiled(*args)
    return control_block(*args)


class CircuitScan(nn.Module):
    def __init__(self, circuit=None, divisor=16, sample_rate=48000, iterations=24, learnable=True):
        super().__init__()
        self.base = circuit or Circuit()
        self.divisor, self.rate, self.iterations = int(divisor), int(sample_rate), int(iterations)
        raw = {}
        for name, (lo, hi) in LEARNABLE.items():
            unit = (getattr(self.base, name)-lo)/(hi-lo)
            if not 0 < unit < 1:
                raise ValueError(f'{name} must lie strictly inside its bounds {lo, hi}')
            raw[name] = nn.Parameter(torch.tensor(math.log(unit/(1-unit)), dtype=torch.float64),
                                     requires_grad=learnable)
        self.raw = nn.ParameterDict(raw)
        self.register_buffer('prior', torch.stack([p.detach().clone() for p in raw.values()]))

    def values(self):
        v = asdict(self.base)
        for name, (lo, hi) in LEARNABLE.items():
            v[name] = lo+(hi-lo)*torch.sigmoid(self.raw[name])
        return SimpleNamespace(**v)

    def prior_loss(self):
        return (torch.stack(list(self.raw.values()))-self.prior).square().mean()

    @staticmethod
    def rest(batch, device):
        z = torch.zeros(batch, device=device, dtype=torch.float64)
        return dict(e=z, f=z.clone(), s=z.clone())

    def _front(self, p, conductance, rr):
        """Return (A, B): front_a = A·u and front_b = B·u for input volts u."""
        load = 1/p.gain_pot+conductance
        den = 1/p.front_series+1/p.threshold_pot+load/(1+rr*load)
        a = (1/p.front_series)/den
        return a, a/(1+rr*load)

    def forward(self, dry, pots, state=None, corrections=None):
        """dry [B,T] full-scale audio, T a multiple of divisor; pots [B,5] physical positions.

        corrections (optional, all [B,K] with K=T/divisor, already bounded):
        'detector', 'attack', 'release' are log-scale factors on the detector
        gain, the charge resistance and the discharge resistance.
        Returns wet [B,T] (float64, full scale), states [B,K,3], next state.
        """
        b, t = dry.shape
        r = self.divisor
        if t % r:
            raise ValueError(f'Chunk length {t} must be a multiple of {r}')
        k = t//r
        p, dt = self.values(), r/self.rate
        corrections = corrections or {}
        state = self.rest(b, dry.device) if state is None else state
        u = dry.double()*p.input_volts_per_fs
        peak = u.abs().reshape(b, k, r).amax(-1)
        thr, ratio, attack, release, makeup = (pots[:, i].double() for i in range(5))
        rr = p.ratio_pot*ratio
        gain_thr = p.rectifier_gain*taper(thr, p.pot_taper)
        ra0 = p.r11+p.attack_pot*taper(attack, p.pot_taper)
        rel0 = p.r9+p.release_pot*release
        bias = p.r10+p.release_trim
        trim = p.control_pot/(p.r17+p.control_pot)*p.control_trim_fraction
        fixed = (p.gain_pot, p.front_series, p.threshold_pot, p.gre_dark, p.gre_mix, bias, trim,
                 p.opamp_gain, p.opamp_rail, p.supply, p.diode_drop, p.c3, dt)
        zero = peak.new_zeros(b, k)
        cdet, catt, crel = (corrections.get(name, zero) for name in ('detector', 'attack', 'release'))
        e, f, s = state['e'], state['f'], state['s']
        conductance = [p.gre_dark+p.gre_span*(p.gre_mix*f+(1-p.gre_mix)*s)]
        blocks = []
        for i in range(0, k, UNROLL):
            n = slice(i, i+UNROLL)
            out = _block(e, f, s, peak[:, n], cdet[:, n], catt[:, n], crel[:, n], gain_thr, rr, ra0, rel0,
                         p.rectifier_offset, p.gre_span, p.gre_half, p.gre_attack_fast, p.gre_attack_slow,
                         p.gre_release_fast, p.gre_release_slow, fixed, self.iterations)
            e, f, s = out[:, -1, 0], out[:, -1, 1], out[:, -1, 2]
            blocks.append(out)
        out = torch.cat(blocks, 1)
        states = out[..., :3]
        conductance = torch.cat((conductance[0][:, None], out[..., 3]), 1)
        g = conductance  # [B,K+1] at control-step boundaries
        w = torch.arange(1, r+1, device=dry.device, dtype=torch.float64)/r
        per_sample = (g[:, :-1, None]*(1-w)+g[:, 1:, None]*w).reshape(b, t)
        _, front_b = self._front(p, per_sample, rr[:, None])
        amp = p.amp_gain*torch.pow(10., makeup[:, None]/20)*front_b*u
        wet = p.amp_headroom*torch.tanh(amp/p.amp_headroom)/p.output_volts_per_fs
        return wet, states, dict(e=e, f=f, s=s)
