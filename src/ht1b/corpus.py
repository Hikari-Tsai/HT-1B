"""Lazy, exhaustive traversal of sample-disjoint full-corpus partitions."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf


def load_corpus(path, dataset_root=None):
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    if raw.get('schema_version')!=2:raise ValueError('Expected full-corpus schema 2')
    root=Path(dataset_root or raw['dataset_root'])
    seen=set();group_splits={}
    for row in raw['records']:
        if row['id'] in seen:raise ValueError('Duplicate record ID')
        seen.add(row['id']);cursor=0
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe corpus path')
        for interval in row['intervals']:
            a,b=interval['start_sample'],interval['stop_sample'];split=interval['split']
            if split not in ('train','validation','test') or a!=cursor or b<=a or b>row['frames']:
                raise ValueError('Partition has a gap, overlap or invalid bounds')
            old=group_splits.setdefault(interval['group'],split)
            if old!=split:raise ValueError('Acoustic group leaked across splits')
            cursor=b
        if cursor!=row['frames']:raise ValueError('Unassigned recording tail')
        info=sf.info(root/rel)
        if info.frames!=row['frames'] or info.samplerate!=48000 or info.channels!=2:raise ValueError('Audio changed since preparation')
    return raw,root


def blocks(corpus, split='train', seconds=30):
    if seconds<=0:raise ValueError('Block duration must be positive')
    result=[]
    for index,row in enumerate(corpus['records']):
        size=max(1,round(seconds*row['sample_rate']))
        for interval in row['intervals']:
            if interval['split']!=split:continue
            for a in range(interval['start_sample'],interval['stop_sample'],size):
                result.append(dict(record=index,start=a,stop=min(a+size,interval['stop_sample']),group=interval['group']))
    return result


def batch_bounds(block,batch_size):
    if batch_size<1:raise ValueError('Invalid batch size')
    for start in range(block['start'],block['stop'],batch_size):yield start,min(start+batch_size,block['stop'])


def read_block(root,row,block):
    with sf.SoundFile(root/row['path']) as handle:
        handle.seek(block['start']); audio=handle.read(block['stop']-block['start'],dtype='float64',always_2d=True)
    if len(audio)!=block['stop']-block['start'] or not np.isfinite(audio).all():raise ValueError('Truncated/nonfinite audio')
    return audio
