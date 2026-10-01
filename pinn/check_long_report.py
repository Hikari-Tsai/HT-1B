"""Validate completed long evaluation, frozen artifacts, and offline page payload."""
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/long-clip-evaluation-20260929'
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def check():
    comparison=read(RUN/'comparison.json')
    protocol=read(RUN/'protocol.json')
    old=read(ROOT/'runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json')
    data=read(ROOT/'output/cl1b-comparison.metrics.json')
    html=(ROOT/'output/cl1b-comparison.html').read_text(encoding='utf-8')
    embedded=json.loads(re.search(r'<script id="benchmark-data" type="application/json">(.*?)</script>',html,re.S)[1])
    assert embedded==data and data['long_form']==comparison and data['five_models']==old
    assert comparison['protocol']==protocol and not protocol['smoke']
    assert comparison['manifest_signature']==old['manifest_signature']
    signature=hashlib.sha256(json.dumps(protocol['windows'],sort_keys=True).encode()).hexdigest()
    assert signature==comparison['window_signature']==protocol['window_signature']
    frozen=read(RUN/'frozen-proposal.json')
    assert len(frozen['windows'])==4835
    assert frozen['source_manifest_signature']==protocol['manifest_signature']
    assert frozen['windows']==[w for w in protocol['windows'] if w['candidate'] in set('ABCDEFGH')]
    assert set(comparison['studies'])==set('ABCDEFGH')|{'A1','A2','A3'}
    assert set(comparison['sources'])==set(old['sources']) and len(old['sources'])==7
    for method,source in comparison['sources'].items():
        assert source['sha256']==old['sources'][method]['sha256']
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
        assert len(list((RUN/'metrics'/method).glob('*.json')))==620
    for cid,study in comparison['studies'].items():
        expected={w['record']:w for w in protocol['windows'] if w['candidate']==cid}
        assert set(r['id'] for r in study['records'])==set(expected)
        assert len(study['records'])==study['files']==(495 if cid=='G' else 620)
        for row in study['records']:
            w=expected[row['id']]
            assert row['samples']==w['score_stop_sample']-w['score_start_sample']
            assert set(row['metrics'])==set(old['sources'])
        assert study['samples_scored']==sum(r['samples'] for r in study['records'])
    result=dict(verified=True,window_signature=signature,methods=7,scopes=11,
        old_short_scores_unchanged=True,html_matches_completed_comparison=True,
        results={cid:comparison['studies'][cid]['macro_mean'] for cid in ('B','A3')})
    (RUN/'report-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':check()
