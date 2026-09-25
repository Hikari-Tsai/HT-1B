"""Lazy, aligned stereo reads. No resampling, normalization or random chunks."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from ht1b.dataset import parse_filename


@dataclass(frozen=True)
class Record:
    id: str
    path: Path
    split: str
    start: int
    stop: int
    delay: int
    controls: tuple[float, ...]
    gain_db: float
    sample_rate: int
    source_id: str | None = None


@dataclass(frozen=True)
class Chunk:
    dry: np.ndarray
    wet: np.ndarray
    offset: int  # relative to the beginning of this record


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    return value


def encode_labels(labels):
    """Order: threshold, ratio, attack, release. These are NOT circuit pots."""
    try:
        t, r, a, rel, g = (float(labels[key]) for key in
                           ('threshold_db', 'ratio', 'attack_index', 'release_index', 'gain_db'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('parameters need threshold_db, ratio, attack_index, release_index, gain_db') from exc
    if not all(math.isfinite(v) for v in (t, r, a, rel, g)):
        raise ValueError('parameters must be finite')
    if not (-40 <= t <= 0 and 2 <= r <= 10 and 0 <= a <= 4 and 0 <= rel <= 4 and -60 <= g <= 60):
        raise ValueError('parameters outside supported CL-1B ranges (gain -60..60 dB)')
    return (-t/40, (r-2)/8, a/4, rel/4), g


def load_records(manifest, sample_rate):
    manifest = Path(manifest).resolve()
    doc = json.loads(manifest.read_text())
    if doc.get('schema_version') != 1 or not isinstance(doc.get('records'), list):
        raise ValueError('Expected schema_version=1 and a records list')
    records, ids = [], set()
    for row in doc['records']:
        path = (manifest.parent / row['stereo_pair']).resolve()
        info = sf.info(path)
        if info.channels != 2:
            raise ValueError(f'{path.name}: expected stereo WAV (left dry, right wet)')
        if info.samplerate != sample_rate:
            raise ValueError(f'{path.name}: sample rate {info.samplerate}, expected {sample_rate}; no automatic resampling')
        if not info.format.startswith('WAV'):
            raise ValueError(f'{path.name}: expected a WAV file')
        name, split = str(row['id']), row.get('split', 'train')
        if name in ids or not name:
            raise ValueError('Record ids must be unique and nonempty')
        if split not in ('train', 'validation', 'test'):
            raise ValueError(f'Unsupported split: {split}')
        ids.add(name)
        delay = _integer(row.get('delay_samples', 0), 'delay_samples')
        start = _integer(row.get('start_sample', max(0, -delay)), 'start_sample')
        stop = _integer(row.get('stop_sample', min(info.frames, info.frames-delay)), 'stop_sample')
        if not (0 <= start < stop <= info.frames and 0 <= start+delay < stop+delay <= info.frames):
            raise ValueError(f'{name}: sample interval or alignment outside recording')
        labels = row.get('parameters') or row.get('metadata', {}).get('filename_parameters')
        if labels is None:
            labels = parse_filename(path)
        controls, gain = encode_labels(labels)
        # Existing PINN controls encode approximate resistances, so never reuse them.
        source = row.get('source_id')
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError('source_id must be a nonempty string when supplied')
        records.append(Record(name, path, split, start, stop, delay, controls, gain,
                              sample_rate, source))
    if not records:
        raise ValueError('No recordings')
    for i, a in enumerate(records):
        for b in records[i+1:]:
            if a.split != b.split and a.source_id is not None and a.source_id == b.source_id:
                raise ValueError(f'Source {a.source_id} occurs across splits; group source material together')
            if a.path == b.path:
                # Guard both original channel timelines, including the alignment shift.
                alo, ahi = a.start+min(0, a.delay), a.stop+max(0, a.delay)
                blo, bhi = b.start+min(0, b.delay), b.stop+max(0, b.delay)
                if max(alo, blo) < min(ahi, bhi):
                    raise ValueError(f'Record intervals overlap: {a.id}, {b.id}')
    return records


def iter_chunks(record, chunk_samples):
    if chunk_samples < 1:
        raise ValueError('chunk_samples must be positive')
    with sf.SoundFile(record.path) as dry_file, sf.SoundFile(record.path) as wet_file:
        dry_file.seek(record.start)
        wet_file.seek(record.start + record.delay)
        for start in range(record.start, record.stop, chunk_samples):
            count = min(chunk_samples, record.stop-start)
            dry = dry_file.read(count, dtype='float32', always_2d=True)[:, 0].copy()
            wet = wet_file.read(count, dtype='float32', always_2d=True)[:, 1].copy()
            if len(dry) != count or len(wet) != count:
                raise ValueError(f'{record.id}: recording changed or ended early')
            if not (np.isfinite(dry).all() and np.isfinite(wet).all()):
                raise ValueError(f'{record.id}: audio must be finite')
            yield Chunk(dry, wet, start-record.start)


def prepare_manifest(files, output, *, sample_rate=48000, validation_files=None,
                     validation_fraction=.2, gap_samples=48000, delay_samples=0):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Manifest already exists; choose a new output path')
    if not files or not 0 < validation_fraction < 1 or gap_samples < 0:
        raise ValueError('Need input files, fraction in (0,1), and a nonnegative gap')
    rows = []
    explicit = bool(validation_files)
    for split, paths in [('train', files), ('validation', validation_files or [])]:
        for path in paths:
            path = Path(path).resolve()
            info = sf.info(path)
            first, last = max(0, -delay_samples), min(info.frames, info.frames-delay_samples)
            base = dict(stereo_pair=str(path), parameters=parse_filename(path),
                        delay_samples=delay_samples)
            if explicit:
                rows.append(dict(base, id=f'{split}-{len(rows)}-{path.stem}', split=split,
                                 start_sample=first, stop_sample=last))
            else:
                boundary = first + int((last-first)*(1-validation_fraction))
                # At least enough guard for the wet/dry alignment footprints.
                validation_start = boundary + max(gap_samples, abs(delay_samples))
                if validation_start >= last or boundary <= first:
                    raise ValueError('Recording too short for requested split and guard interval')
                rows.extend([
                    dict(base, id=f'train-{len(rows)}-{path.stem}', split='train',
                         start_sample=first, stop_sample=boundary),
                    dict(base, id=f'validation-{len(rows)}-{path.stem}', split='validation',
                         start_sample=validation_start, stop_sample=last)])
    doc = dict(schema_version=1, split_strategy='held-out-files' if explicit else 'chronological-with-guard',
               note='Temporal splitting is not independent-source or unseen-setting validation.', records=rows)
    # Validate through the same loader before exposing the final manifest.
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=output.parent, delete=False) as temp:
        json.dump(doc, temp, indent=2)
        temporary = Path(temp.name)
    try:
        load_records(temporary, sample_rate)
        with output.open('x') as stream:
            stream.write(json.dumps(doc, indent=2)+'\n')
    finally:
        temporary.unlink(missing_ok=True)
    return output


def data_signature(manifest, records):
    """Content hashes, not only names or mtimes, protect epoch-boundary resume."""
    digest = hashlib.sha256(Path(manifest).read_bytes())
    for path in sorted({r.path for r in records}):
        digest.update(str(path).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()
