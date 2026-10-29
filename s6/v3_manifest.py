"""Convert the v3 full-corpus manifest (content-grouped intervals) to an s6 manifest.

Usage: python -m s6.v3_manifest <v3 manifest.json> <dataset root> <output.json>

Each v3 interval becomes one s6 record with the same split. Content groups
become source_id, so the s6 loader also rejects any cross-split content leak.
"""
import json
import sys
from pathlib import Path


def convert(source, dataset, output):
    source, dataset, output = (Path(p).resolve() for p in (source, dataset, output))
    if output.exists():
        raise ValueError(f'{output} exists; choose a new path')
    doc = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    for record in doc['records']:
        wav = dataset / record['path']
        for index, interval in enumerate(record['intervals']):
            rows.append(dict(
                id=f"{record['id']}-i{index:02d}",
                stereo_pair=str(wav),
                split=interval['split'],
                source_id=f"v3-group-{interval['group']}",
                start_sample=interval['start_sample'],
                stop_sample=interval['stop_sample'],
                delay_samples=0,
                parameters=record['labels']))
    out = dict(schema_version=1, split_strategy='v3-content-grouped-intervals',
               source_manifest=str(source), source_signature=doc.get('signature'),
               note='Intervals from the v3 full-corpus manifest; each interval streams from its own '
                    'start with warmup and reset state. Content isolation is acoustic-grouping based.',
               records=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(out, stream, indent=1)
    counts = {}
    for row in rows:
        c = counts.setdefault(row['split'], [0, 0])
        c[0] += 1
        c[1] += row['stop_sample'] - row['start_sample']
    return {k: dict(records=n, hours=s/48000/3600) for k, (n, s) in counts.items()}


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(convert(*sys.argv[1:]))
