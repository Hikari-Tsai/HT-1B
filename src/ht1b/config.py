"""SI circuit parameters and normalized physical potentiometer positions."""
from dataclasses import dataclass, asdict, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Controls:
    threshold: float = .7
    ratio: float = .5
    attack: float = .15
    release: float = .3
    makeup_db: float = 0.
    mode: str = 'manual'

    def __post_init__(self):
        for key in ('threshold', 'ratio', 'attack', 'release'):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must be a normalized potentiometer position in [0, 1]')
        if self.mode not in ('fixed', 'manual', 'fix-man'):
            raise ValueError('mode must be fixed, manual or fix-man')
        if not math.isfinite(self.makeup_db) or not -60 <= self.makeup_db <= 30:
            raise ValueError('makeup_db must be finite and in [-60, 30]')


@dataclass(frozen=True)
class Circuit:
    # Verified resistor/capacitor values, 1993 rev 1.0 schematic.
    front_series: float = 100_000.
    threshold_pot: float = 100_000.
    ratio_pot: float = 10_000.
    gain_pot: float = 100_000.
    c3: float = 10e-6
    r11: float = 274.
    attack_pot: float = 500_000.
    r9: float = 47_500.
    release_pot: float = 500_000.
    r10: float = 274_000.
    release_trim: float = 235_000.  # Unknown hardware setting: assumed midpoint.
    r17: float = 20_000.
    control_pot: float = 100_000.
    # Explicit approximations/calibration values.
    supply: float = 15.
    opamp_rail: float = 13.5
    opamp_gain: float = 1_000.
    diode_drop: float = .55
    rectifier_gain: float = 2.
    rectifier_offset: float = .03
    pot_taper: float = 2.
    control_trim_fraction: float = .5
    gre_dark: float = 1e-8
    gre_span: float = .001
    gre_half: float = .12
    gre_mix: float = .8
    gre_attack_fast: float = .001
    gre_attack_slow: float = .01
    gre_release_fast: float = .05
    gre_release_slow: float = .5
    amp_gain: float = 2.05
    amp_headroom: float = 24.
    input_volts_per_fs: float = 10.
    output_volts_per_fs: float = 10.

    def __post_init__(self):
        nonnegative = {'release_trim', 'diode_drop', 'rectifier_offset', 'gre_dark'}
        fractions = {'control_trim_fraction', 'gre_mix'}
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f'{field.name} must be finite')
            if field.name in fractions:
                if not 0 <= value <= 1: raise ValueError(f'{field.name} must be in [0, 1]')
            elif field.name in nonnegative:
                if value < 0: raise ValueError(f'{field.name} must be nonnegative')
            elif value <= 0:
                raise ValueError(f'{field.name} must be positive')
        if self.opamp_rail >= self.supply:
            raise ValueError('opamp_rail must be smaller than supply')


def read_config(path=None):
    raw = json.loads(Path(path).read_text()) if path else {}
    if set(raw) - {'circuit', 'controls', 'metadata'}:
        raise ValueError('Unknown top-level config key')
    return Circuit(**raw.get('circuit', {})), Controls(**raw.get('controls', {}))


def write_config(path, circuit, controls, metadata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({'circuit': asdict(circuit), 'controls': asdict(controls),
                                    'metadata': metadata or {}}, indent=2)+'\n')
