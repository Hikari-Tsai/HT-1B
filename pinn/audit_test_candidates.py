"""Dry-only content audit for discussion; never trains or scores compressor models.

Requires the separate content-audit environment (transformers, torch, scipy,
soundfile). CLAP labels are hypotheses, not human listening or source identities.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/test-candidates'
LABELS={
 'acoustic_guitar':'The sound of a solo acoustic guitar being strummed and plucked.',
 'electric_guitar':'The sound of a solo electric guitar playing a riff.',
 'distorted_guitar':'The sound of a distorted electric guitar playing rock music.',
 'bass':'The sound of a solo electric bass guitar playing low plucked notes.',
 'drum_kit':'The sound of a solo drum kit playing a rhythmic drum beat.',
 'kick':'The sound of isolated bass drum kick hits.',
 'snare':'The sound of isolated snare drum hits.',
 'cymbals':'The sound of hi-hat and cymbal hits.',
 'percussion':'The sound of hand percussion playing a rhythm.',
 'piano':'The sound of a solo piano playing music.',
 'strings':'The sound of bowed string instruments playing music.',
 'synthesizer':'The sound of an electronic synthesizer playing musical notes.',
 'male_singing':'The sound of a man singing without instrumental accompaniment.',
 'female_singing':'The sound of a woman singing without instrumental accompaniment.',
 'choir':'The sound of several people singing together.',
 'male_speech':'The sound of a man speaking.',
 'female_speech':'The sound of a woman speaking.',
 'mixed_music':'The sound of a full band playing music with several instruments together.',
 'sine_sweep':'The sound of a sine wave test tone sweeping from low to high frequency.',
 'white_noise':'The sound of continuous white noise or static hiss.',
 'clicks':'The sound of short electronic clicks or impulses.',
 'silence':'The sound of silence with very faint background noise.'}

def features(x):
    x=np.asarray(x,dtype=np.float64)
    frames=x[:len(x)//4800*4800].reshape(-1,4800)
    db=10*np.log10(np.maximum(np.mean(frames**2,axis=1),1e-20))
    return dict(rms_dbfs=float(10*np.log10(max(np.mean(x*x),1e-20))),
                peak_dbfs=float(20*np.log10(max(np.max(np.abs(x)),1e-10))),
                quiet_fraction_minus65=float(np.mean(db<-65)),
                frame_rms_dbfs_quantiles=np.quantile(db,[.1,.5,.9]).tolist())

def main(verify=False):
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model_id='laion/clap-htsat-unfused'
    print('Loading CPU content classifier',flush=True)
    processor=ClapProcessor.from_pretrained(model_id)
    model=ClapModel.from_pretrained(model_id).cpu().eval()
    with torch.inference_mode():
        inputs=processor(text=list(LABELS.values()),padding=True,return_tensors='pt')
        text=model.get_text_features(**inputs)
        text=text/text.norm(dim=1,keepdim=True)
    manifest=json.loads((ROOT/'runs/full-corpus-gpu-20260925/manifest.json').read_text())
    if verify:
        checks=[]
        for rid in manifest['representatives']:
            ref=next(r for r in manifest['records'] if r['id']==rid)
            dry,sr=sf.read(ROOT/'data/tubetech-cl-1b-v1'/ref['path'],dtype='float32',always_2d=True)
            for start,stop in [(35,40),(60,65),(65,70),(70,75),(75,80),(90,95),(95,100),(100,105),(150,155),(155,160),(200,205),(205,210),(210,215),(215,219.8)]:
                if stop*sr>len(dry):continue
                x=dry[round(start*sr):round(stop*sr),0]
                with torch.inference_mode():
                    inp=processor(audio=x*(.5/max(float(np.max(np.abs(x))),1e-8)),sampling_rate=sr,return_tensors='pt')
                    embedding=model.get_audio_features(**inp)
                    embedding=embedding/embedding.norm(dim=1,keepdim=True)
                    sims=(embedding@text.T)[0].numpy()
                top=np.argsort(sims)[::-1][:3]
                checks.append(dict(reference=ref['path'],start=start,stop=stop,features=features(x),candidates=[dict(label=list(LABELS)[i],cosine=float(sims[i])) for i in top]))
            print('Verified content family',rid,flush=True)
        (OUT/'content-crosscheck.json').write_text(json.dumps(dict(model_id=model_id,revision=model.config._commit_hash,checks=checks),indent=2),encoding='utf-8')
        return
    ref=next(r for r in manifest['records'] if r['id']==manifest['representatives'][1])
    dry,sr=sf.read(ROOT/'data/tubetech-cl-1b-v1'/ref['path'],dtype='float32',always_2d=True)
    assert sr==48000
    dry=dry[:,0]
    observations=[]
    for start in range(0,len(dry)//sr,5):
        stop=min(start+5,len(dry)/sr)
        x=dry[start*sr:round(stop*sr)]
        item=dict(start=start,stop=stop,**features(x))
        # Content recognition only; no normalized waveform enters model evaluation.
        normalized=x*(.5/max(float(np.max(np.abs(x))),1e-8))
        with torch.inference_mode():
            audio_inputs=processor(audio=normalized,sampling_rate=sr,return_tensors='pt')
            embedding=model.get_audio_features(**audio_inputs)
            embedding=embedding/embedding.norm(dim=1,keepdim=True)
            similarities=(embedding@text.T)[0].numpy()
        best=np.argsort(similarities)[::-1][:5]
        item['candidates']=[dict(label=list(LABELS)[i],cosine=float(similarities[i])) for i in best]
        observations.append(item)
        print(start,stop,item['candidates'][:3],flush=True)
    payload=dict(status='draft_content_hypotheses',reference=ref['path'],sample_rate=sr,
                 model_id=model_id,revision=model.config._commit_hash,device='cpu',
                 label_definitions=LABELS,observations=observations,
                 source_manifest_signature=manifest['signature'],
                 limitations=['No human listening confirmation or original source names.',
                              'Cosine similarities are not calibrated probabilities.',
                              'Classification uses peak-normalized Dry only; candidates do not alter historical split or metrics.'])
    (OUT/'content-audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    main(parser.parse_args().verify)
