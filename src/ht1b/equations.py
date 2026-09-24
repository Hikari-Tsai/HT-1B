"""Single source of equations: scalar/array NumPy, Torch and SymPy backends.

Topology is reduced, not a full CL-1B netlist. See docs/EQUATIONS.md.
"""
import numpy as np


class NumpyOps:
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    absolute = staticmethod(np.abs)
    tanh = staticmethod(np.tanh)
    where = staticmethod(np.where)


def taper(position, decades):
    """Assumed logarithmic pot law. Exact hardware taper is not known."""
    return (10**(decades*position)-1)/(10**decades-1)


def evaluate(state, input_volts, p, c, ops):
    """Return rates [V/s, 1/s, 1/s], output volts and diagnostic nodes."""
    e, fast, slow = state
    conductance = p.gre_dark + p.gre_span*(p.gre_mix*fast+(1-p.gre_mix)*slow)
    load = 1/p.gain_pot + conductance
    rr = p.ratio_pot*c.ratio
    # Exact elimination of the two-node passive front network, ideal input source.
    front_a = (input_volts/p.front_series) / (1/p.front_series + 1/p.threshold_pot + load/(1+rr*load))
    front_b = front_a/(1+rr*load)
    detector = ops.maximum(p.rectifier_gain*ops.absolute(front_a)*taper(c.threshold, p.pot_taper)-p.rectifier_offset, 0.)
    u = ops.maximum(ops.minimum(p.opamp_gain*(detector-e), p.opamp_rail), -p.opamp_rail)
    ra = p.r11 + p.attack_pot*taper(c.attack, p.pot_taper)
    release_r = p.r9 + p.release_pot*c.release
    bias_r = p.r10+p.release_trim
    # Thevenin equivalent at RELEASE-1; D4 points from C3 to RELEASE-1.
    vth = (u*bias_r+p.supply*release_r)/(release_r+bias_r)
    rth = release_r*bias_r/(release_r+bias_r)
    ia = ops.maximum(u-e-p.diode_drop, 0.)/ra
    ir = ops.maximum(e-vth-p.diode_drop, 0.)/rth
    de = (ia-ir)/p.c3
    if c.mode == 'fixed': control = detector
    elif c.mode == 'manual': control = ops.maximum(e, 0.)
    else: control = ops.maximum(detector, e)
    # C8 is eliminated quasi-statically; its 16.7 us pole is omitted.
    drive = control*p.control_pot/(p.r17+p.control_pot)*p.control_trim_fraction
    target = drive/(drive+p.gre_half)
    tf = ops.where(target > fast, p.gre_attack_fast, p.gre_release_fast)
    ts = ops.where(target > slow, p.gre_attack_slow, p.gre_release_slow)
    df = (target-fast)/tf
    ds = (target-slow)/ts
    amp_input = p.amp_gain*10**(c.makeup_db/20)*front_b
    output = p.amp_headroom*ops.tanh(amp_input/p.amp_headroom)
    return dict(rhs=(de, df, ds), output=output, front_a=front_a, front_b=front_b,
                front_kcl_a=(front_a-input_volts)/p.front_series+front_a/p.threshold_pot+front_b*load,
                front_kcl_b=front_a-front_b-rr*front_b*load,
                detector=detector, opamp=u, ia=ia, ir=ir, control=control,
                drive=drive, target=target, tau_fast=tf, tau_slow=ts, ra=ra,
                conductance=conductance)
