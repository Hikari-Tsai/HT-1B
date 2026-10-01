"""Partition full recordings by shared dry-audio content, never by knob/file alone.

Automatic acoustic boundaries are approximate; they are not verified song IDs.
Only dry audio participates in segmentation/grouping. Every sample is assigned.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import correlate, find_peaks, resample_poly, stft
from ht1b.config import Circuit
from ht1b.dataset import dataset_controls, parse_filename


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def dry_features(path):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != 48000 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f'Unexpected audio format: {path}')
    dry = resample_poly(audio[:, 0], 1, 8)
    _, times, spectrum = stft(dry, 6000, window='hann', nperseg=1024,
                              noverlap=424, boundary='zeros', padded=True)
    frequencies = np.fft.rfftfreq(1024, 1/6000)
    hz = 700 * (10**(np.linspace(0, 2595*np.log10(1+3000/700), 34)/2595)-1)
    filters = np.maximum(0, np.minimum((frequencies[None]-hz[:-2,None])/np.maximum(hz[1:-1,None]-hz[:-2,None],1e-9),
                                     (hz[2:,None]-frequencies[None])/np.maximum(hz[2:,None]-hz[1:-1,None],1e-9)))
    mel = np.maximum(filters @ (np.abs(spectrum)**2), 1e-12).T
    # Exclude cepstral c0 for robustness to analog gain differences.
    mfcc = dct(np.log(mel), type=2, norm='ortho', axis=1)[:,1:14]
    n = len(audio)//4800
    rms = np.sqrt(np.mean(audio[:n*4800,0].reshape(n,4800)**2, axis=1))
    return times, mfcc, np.log10(np.maximum(rms,1e-6))*20


def cosine(a,b):
    return float(np.dot(a,b)/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)))


def align_envelopes(reference_path, path):
    """Find a constant content offset on a 10-ms dry envelope, only for outliers."""
    def envelope(file):
        x,_=sf.read(file,dtype='float32',always_2d=True)
        n=len(x)//480
        return 20*np.log10(np.maximum(np.sqrt(np.mean(x[:n*480,0].reshape(n,480)**2,axis=1)),1e-6))
    a,b=envelope(reference_path),envelope(path)
    scores=[]
    for lag in range(-500,501):
        start=max(0,-lag);stop=min(len(a),len(b)-lag)
        x=a[start:stop];y=b[start+lag:stop+lag]
        scores.append(cosine(x-x.mean(),y-y.mean()))
    best=int(np.argmax(scores));lag=best-500
    return lag/100,scores[best]


def repeated_content(path, threshold=.90):
    """Search gain/polarity-invariant 2-second waveform matches away from self.

    Timbre averages alone miss repeated performances. These normalized inner
    products conservatively join every segment touched by a detected match.
    """
    samples,_=sf.read(path,dtype='float32',always_2d=True)
    dry=resample_poly(samples[:,0],1,16);length=6000
    cumulative=np.r_[0.,np.cumsum(dry.astype(float)**2)]
    energies=cumulative[length:]-cumulative[:-length]
    matches=[]
    for start in range(0,len(dry)-length+1,3000):
        query=dry[start:start+length];energy=float(np.sum(query*query))
        if energy/length<1e-7:continue
        scores=np.abs(correlate(dry,query,mode='valid',method='fft'))/np.sqrt(np.maximum(energies*energy,1e-15))
        scores[max(0,start-15000):min(len(scores),start+15001)]=0
        stop=int(np.argmax(scores))
        if scores[stop]>=threshold:
            matches.append(dict(source_start=start/3000,match_start=stop/3000,seconds=2.,similarity=float(scores[stop])))
    return matches


def partition_intervals(edges, group_ids, splits, frames, sr=48000):
    rows=[]
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        start=min(frames,round(a*sr)); stop=min(frames,round(b*sr))
        if start<stop: rows.append(dict(start_sample=start,stop_sample=stop,group=group_ids[i],split=splits[group_ids[i]]))
    if not rows or rows[0]['start_sample']!=0 or rows[-1]['stop_sample']!=frames:
        raise ValueError('Incomplete partition')
    if any(a['stop_sample']!=b['start_sample'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Gap/overlap in partition')
    return rows


def prepare(dataset: Path, output: Path):
    output.mkdir(parents=True,exist_ok=False)
    files=sorted(dataset.rglob('*.wav'))
    if not files: raise ValueError('No WAV recordings')
    inventory=[]
    for file in files:
        info=sf.info(file); labels=parse_filename(file)
        if info.samplerate!=48000 or info.channels!=2: raise ValueError(f'Unexpected format: {file}')
        controls,_=dataset_controls(labels,Circuit())
        inventory.append(dict(id=file.stem,path=file.relative_to(dataset).as_posix(),frames=info.frames,
                              sample_rate=info.samplerate,labels=labels,controls=asdict(controls)))
    # One representative per release-index / duration family; full dry-only analysis.
    representatives={}
    for i,row in enumerate(inventory): representatives.setdefault((row['labels']['release_index'],round(row['frames']/48000)),i)
    features=[]
    for i in representatives.values():
        print(f'Analyzing full dry reference {inventory[i]["path"]}',flush=True)
        features.append(dry_features(files[i]))
    duration=max(r['frames']/48000 for r in inventory)
    bins=round(duration*10)+1
    stack=np.full((len(features),bins,13),np.nan)
    for i,(times,mfcc,_) in enumerate(features): stack[i,:min(bins,len(mfcc))]=mfcc[:bins]
    consensus=np.nanmedian(stack,axis=0)
    scale=np.maximum(np.nanstd(consensus,axis=0),.5)
    normalized=(consensus-np.nanmean(consensus,axis=0))/scale
    novelty=np.zeros(bins)
    for i in range(20,bins-20):
        left=normalized[i-20:i].mean(axis=0);right=normalized[i:i+20].mean(axis=0)
        novelty[i]=np.linalg.norm(left-right)
    peaks,_=find_peaks(novelty,distance=80,prominence=.65)
    # Reject weak boundaries; divide very long regions but retain common group IDs below.
    selected=[int(x) for x in peaks if novelty[x]>=np.quantile(novelty[20:-20],.65)]
    edges=[0.]+[x/10 for x in selected if 8<=x/10<=duration-8]+[duration]
    edges=sorted(set(edges))
    original_edges=edges[:]
    # Segment descriptors pool MFCC trajectories; conservative acoustic grouping.
    descriptors=[]
    for a,b in zip(edges[:-1],edges[1:]):
        chunk=consensus[round(a*10):round(b*10)]
        descriptors.append(np.r_[chunk.mean(axis=0),np.std(chunk,axis=0)])
    # Whiten using the reference statistics; mean cepstra dominate timbre matching.
    descriptors=np.asarray(descriptors)
    descriptor_scale=np.r_[scale,scale]
    descriptors=descriptors/descriptor_scale
    parent=list(range(len(descriptors)))
    def root(i):
        while parent[i]!=i: i=parent[i]
        return i
    pairs=[]
    for i,j in itertools.combinations(range(len(descriptors)),2):
        sim=cosine(descriptors[i],descriptors[j])
        if sim>=.975:
            parent[root(j)]=root(i);pairs.append(dict(a=i,b=j,similarity=sim))
    longest=max(representatives.values(),key=lambda i:inventory[i]['frames'])
    repeats=repeated_content(files[longest])
    for match in repeats:
        touched=set()
        for start in (match['source_start'],match['match_start']):
            touched.update(i for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])) if a<start+2 and b>start)
        touched=sorted(touched)
        for other in touched[1:]:parent[root(other)]=root(touched[0])
    print(json.dumps(dict(repeated_content_matches=len(repeats))),flush=True)
    groups=[root(i) for i in range(len(descriptors))]
    durations={g:sum(b-a for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])) if groups[i]==g) for g in set(groups)}
    if len(durations)<3: raise ValueError('Fewer than three acoustic groups; cannot construct honest holdouts')
    # Hold out entire groups. Prefer sufficient duration and keep most data in training.
    possibilities=[]
    for val,test in itertools.permutations(durations,2):
        if min(durations[val],durations[test])<8: continue
        remaining=duration-durations[val]-durations[test]
        if remaining<duration*.6:continue
        cost=abs(durations[val]/duration-.1)+abs(durations[test]/duration-.1)
        possibilities.append((cost,val,test))
    if not possibilities:raise ValueError('Automatic groups do not allow an adequate training/validation/test partition')
    _,val,test=min(possibilities)
    assignments={g:'validation' if g==val else 'test' if g==test else 'train' for g in durations}
    # Audit every dry track, not only representative files. This also detects shifted layouts.
    ref_rms=features[0][2]
    audit=[]
    started=time.perf_counter()
    for i,(file,row) in enumerate(zip(files,inventory)):
        # Streaming 1-second reads: the full recording is scanned with bounded RAM.
        sums=[]
        with sf.SoundFile(file) as handle:
            while True:
                block=handle.read(48000,dtype='float32',always_2d=True)
                if not len(block):break
                if not np.isfinite(block).all():raise ValueError(f'Nonfinite data: {file}')
                count=len(block)//4800
                sums.extend(np.sqrt(np.mean(block[:count*4800,0].reshape(count,4800)**2,axis=1)).tolist())
        db=20*np.log10(np.maximum(sums,1e-6));n=min(len(db),len(ref_rms))
        corr=cosine(db[:n]-np.mean(db[:n]),ref_rms[:n]-np.mean(ref_rms[:n]))
        offset=0.;original_corr=corr
        if corr<.8:
            offset,corr=align_envelopes(files[next(iter(representatives.values()))],file)
            print(json.dumps(dict(alignment_correction=row['id'],offset_seconds=offset,similarity=corr)),flush=True)
        row['content_offset_seconds']=offset
        mapped=[0.]+[max(0.,min(row['frames']/48000,x+offset)) for x in edges[1:-1]]+[row['frames']/48000]
        audit.append(dict(id=row['id'],dry_envelope_similarity=corr,unshifted_similarity=original_corr,content_offset_seconds=offset))
        row['intervals']=partition_intervals(mapped,groups,assignments,row['frames'])
        if i%25==0 or i==len(files)-1:
            print(json.dumps(dict(scanned=i+1,total=len(files),elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
            write_json(output/'status.json',dict(phase='scanning',scanned=i+1,total=len(files)))
    low=[r for r in audit if r['dry_envelope_similarity']<.8]
    # Fail closed if content ordering cannot be supported by the full-file audit.
    if low:
        write_json(output/'alignment-failures.json',low)
        raise ValueError(f'{len(low)} files have uncertain content alignment; no train manifest will be published')
    totals=Counter()
    for row in inventory:
        for interval in row['intervals']:totals[interval['split']]+=interval['stop_sample']-interval['start_sample']
    timeline=[dict(index=i,start_seconds=a,stop_seconds=b,group=groups[i],split=assignments[groups[i]])
              for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]
    payload=dict(schema_version=2,dataset_root=str(dataset.resolve()),sample_rate=48000,records=inventory,
                 timeline=timeline,partition_method='dry MFCC novelty + cosine acoustic grouping, shared across settings',
                 grouping_threshold=.975,representatives=[inventory[i]['id'] for i in representatives.values()],
                 candidate_boundaries_seconds=original_edges,group_links=pairs,repeated_content_matches=repeats,dry_audit=audit,
                 frames_by_split=dict(totals),hours_by_split={k:v/48000/3600 for k,v in totals.items()},
                 coverage='Every original sample belongs to exactly one split; no tail or silence removed.',
                 independence='Acoustic-content-group holdout inferred from dry audio; not verified song/performer independence.',
                 evaluation_history='Read dry history from recording start; never read held-out wet for training. Validation/test wet only used by their evaluators.',
                 limitations=['Acoustic segmentation may split a performance or group similar instruments.','Whole-file envelope audit supports common ordering, not exact sample alignment.','Identical musical sources cannot be guaranteed absent without original source labels.'])
    payload['signature']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    write_json(output/'manifest.json',payload)
    np.savez_compressed(output/'dry-analysis.npz',times=np.arange(bins)/10,mfcc=consensus,novelty=novelty)
    write_json(output/'status.json',dict(phase='prepared',files=len(files),hours_by_split=payload['hours_by_split']))
    print(json.dumps(dict(phase='prepared',segments=len(timeline),groups=len(durations),hours=payload['hours_by_split'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('data/tubetech-cl-1b-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.dataset,args.output)
