import numpy as np
import soundfile as sf
from pinn.prepare_full_corpus import partition_intervals,repeated_content


def test_shifted_boundaries_preserve_every_sample_including_short_tail():
    rows=partition_intervals([0,9.25,19.25,30],[0,1,2],{0:'train',1:'validation',2:'test'},25*48000)
    assert rows[0]['start_sample']==0
    assert rows[-1]['stop_sample']==25*48000
    assert sum(r['stop_sample']-r['start_sample'] for r in rows)==25*48000
    assert all(a['stop_sample']==b['start_sample'] for a,b in zip(rows,rows[1:]))


def test_inner_product_finds_gain_and_polarity_changed_repeat(tmp_path):
    rng=np.random.default_rng(19)
    dry=rng.normal(0,.1,48000*14).astype('float32')
    dry[8*48000:10*48000]=-.4*dry[:2*48000]
    path=tmp_path/'repeated.wav';sf.write(path,np.column_stack([dry,np.zeros_like(dry)]),48000,subtype='FLOAT')
    matches=repeated_content(path)
    assert any(abs(m['source_start'])<.1 and abs(m['match_start']-8)<.1 and m['similarity']>.99 for m in matches)
