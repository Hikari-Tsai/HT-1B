"""Adapter for the five-setting CL-1B recordings described by the user.

Filename values are metadata. Mapping them to this reduced circuit is explicitly
approximate; they are not claimed to be measured potentiometer resistances.
"""
from dataclasses import asdict
from pathlib import Path
import json
import math
import re
import soundfile as sf
from .config import Circuit, Controls


PATTERN=re.compile(r'^TubeTech_a_(\d+)_r_(\d+)_r_(\d+)_t_(-?\d+(?:\.\d+)?)_g_(-?\d+(?:\.\d+)?)\.wav$',re.I)


def parse_filename(path):
    match=PATTERN.match(Path(path).name)
    if not match: raise ValueError('Expected TubeTech_a_<0..4>_r_<0..4>_r_<ratio>_t_<threshold>_g_<gain>.wav')
    a,r,ratio,t,g=map(float,match.groups())
    if a not in range(5) or r not in range(5) or ratio not in (2,4,6,8,10) or t not in (0,-10,-20,-30,-40):
        raise ValueError('Filename is outside the documented five-setting dataset')
    return dict(attack_index=int(a),release_index=int(r),ratio=int(ratio),threshold_db=t,gain_db=g)


def dataset_controls(labels,circuit,mode='manual'):
    # Assumption: 40 dB of threshold labels maps to 40 dB of wiper sensitivity.
    # Absolute calibration (volts/FS, transformers, GRE) still needs identification.
    fraction=10**((-40-labels['threshold_db'])/20)
    position=math.log10(1+(10**circuit.pot_taper-1)*fraction)/circuit.pot_taper
    c=Controls(threshold=position,ratio=(labels['ratio']-2)/8,
               attack=labels['attack_index']/4,release=labels['release_index']/4,
               makeup_db=labels['gain_db'],mode=mode)
    notes=[
        'Attack/release labels 0..4 map to pot positions index/4; actual resistances are unmeasured.',
        'Ratio 2..10 maps linearly to the 0..10 kohm ratio pot; actual ratio response is not calibrated.',
        'Threshold 0..-40 maps to wiper fraction 0.01..1 (40 dB sensitivity span); absolute dBu calibration is unknown.',
        f'Mode {mode} is supplied to this adapter; dataset description does not specify mode.',
        'Gain label is treated as dB makeup; analog input/output volts-per-FS use the selected config.',
    ]
    return c,notes


def prepare_manifest(paths,output,circuit=None,mode='manual',seconds=None,delay_samples=0):
    p=circuit or Circuit()
    if seconds is not None and (not math.isfinite(seconds) or seconds<=0):
        raise ValueError('seconds must be positive and finite')
    rows=[]
    for path in paths:
        path=Path(path).resolve()
        labels=parse_filename(path)
        info=sf.info(path)
        if info.channels!=2: raise ValueError('Dataset recording must have left dry/right wet channels')
        if delay_samples<0: raise ValueError('Negative delay requires a manually prepared initial-state manifest')
        available=info.frames-delay_samples
        stop=available if seconds is None else min(available,round(seconds*info.samplerate))
        if stop<2: raise ValueError('Insufficient audio after alignment/duration selection')
        controls,notes=dataset_controls(labels,p,mode)
        rows.append(dict(id=path.stem,stereo_pair=str(path),split='train',controls=asdict(controls),
                         delay_samples=delay_samples,start_sample=0,stop_sample=stop,initial_state=[0,0,0],
                         metadata={'source':'user-supplied CL-1B dataset','filename_parameters':labels,
                                   'mapping_assumptions':notes,'original_frames':info.frames,
                                   'used_frames':stop,'initial_state_note':'Assumed fully released at recording start.'}))
    if not rows: raise ValueError('No input recordings')
    destination=Path(output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise ValueError('Manifest already exists; choose a new output path')
    destination.write_text(json.dumps({'schema_version':1,'records':rows},indent=2)+'\n')
    return destination
