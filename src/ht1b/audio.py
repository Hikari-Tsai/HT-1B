"""Explicit dry/wet channel mapping. Never normalize channels independently."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf
from .config import Controls


@dataclass
class Recording:
    id: str
    dry: np.ndarray
    wet: np.ndarray | None
    sample_rate: int
    controls: Controls
    initial_state: np.ndarray
    split: str
    metadata: dict


def read_audio(path):
    audio, rate = sf.read(path, dtype='float64', always_2d=True)
    if len(audio) < 2 or not np.isfinite(audio).all():
        raise ValueError(f'{path}: need at least two finite samples')
    return audio, rate


def write_audio(path, data, rate):
    data = np.asarray(data)
    if not data.size or not np.isfinite(data).all():
        raise ValueError('Refusing to write empty/nonfinite audio')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, rate, subtype='FLOAT')
    return {'peak':float(np.max(np.abs(data))), 'above_full_scale':bool(np.max(np.abs(data))>1)}


def load_manifest(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    if raw.get('schema_version') != 1 or not isinstance(raw.get('records'), list):
        raise ValueError('Expected schema_version: 1 and a records list')
    records, seen = [], set()
    for row in raw['records']:
        name = row['id']
        if not isinstance(name,str) or not name or name in seen:
            raise ValueError('Recording IDs must be nonempty and unique')
        seen.add(name)
        if row.get('split','train') not in ('train','validation','test'):
            raise ValueError('split must be train, validation or test')
        if 'stereo_pair' in row:
            if 'input' in row or 'output' in row:
                raise ValueError('Use stereo_pair OR input/output')
            data,rate = read_audio(path.parent/row['stereo_pair'])
            if data.shape[1] != 2:
                raise ValueError('stereo_pair requires exactly two channels: left=dry, right=wet')
            dry,wet = data[:,0],data[:,1]
        else:
            data,rate = read_audio(path.parent/row['input'])
            if data.shape[1] != 1: raise ValueError('input must be mono; use stereo_pair for L/R')
            dry,wet = data[:,0],None
            if row.get('output'):
                out,wr = read_audio(path.parent/row['output'])
                if wr != rate: raise ValueError('Paired sample rate mismatch')
                if len(out) != len(dry): raise ValueError('Paired length mismatch')
                if out.shape[1] != 1: raise ValueError('output must be mono')
                wet = out[:,0]
        delay = row.get('delay_samples',0)
        if isinstance(delay,bool) or not isinstance(delay,int):
            raise ValueError('delay_samples must be an integer')
        if abs(delay) >= len(dry)-1: raise ValueError('delay leaves insufficient samples')
        if delay and wet is None: raise ValueError('Delay needs a wet recording')
        if delay < 0 and 'initial_state' not in row:
            raise ValueError('Negative delay crops dry history; specify initial_state explicitly')
        if delay > 0: dry,wet = dry[:-delay],wet[delay:]
        elif delay < 0: dry,wet = dry[-delay:],wet[:delay]
        start = row.get('start_sample',0)
        stop = row.get('stop_sample',len(dry))
        if not isinstance(start,int) or not isinstance(stop,int) or not 0 <= start < stop <= len(dry):
            raise ValueError('Invalid start_sample/stop_sample interval')
        if start and 'initial_state' not in row:
            raise ValueError('Cropping input history requires explicit initial_state')
        dry = np.ascontiguousarray(dry[start:stop])
        wet = None if wet is None else np.ascontiguousarray(wet[start:stop])
        if len(dry) < 2: raise ValueError('Recording needs at least two samples')
        initial = np.asarray(row.get('initial_state',[0,0,0]),dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all() or initial[0]<0 or np.any(initial[1:]<0) or np.any(initial[1:]>1):
            raise ValueError('Invalid initial_state')
        metadata = dict(row.get('metadata',{}))
        digest=hashlib.sha256(dry.tobytes())
        if wet is not None: digest.update(wet.tobytes())
        metadata['audio_sha256']=digest.hexdigest()
        metadata['delay_samples']=delay
        metadata['assumed_controls']=sorted(set(Controls.__dataclass_fields__)-set(row.get('controls',{})))
        records.append(Recording(name,dry,wet,rate,Controls(**row.get('controls',{})),initial,
                                 row.get('split','train'),metadata))
    if not records: raise ValueError('Manifest contains no recordings')
    return records


def metrics(prediction, target):
    prediction,target=np.asarray(prediction),np.asarray(target)
    if prediction.shape != target.shape or not prediction.size:
        raise ValueError('Metric inputs must have matching, nonempty shapes')
    error=prediction-target
    mse=float(np.mean(error**2))
    energy=float(np.mean(target**2))
    return {'mse':mse,'mae':float(np.mean(np.abs(error))),
            'esr':mse/max(energy,1e-12),'target_rms':float(np.sqrt(energy)),
            'peak':float(np.max(np.abs(prediction)))}
