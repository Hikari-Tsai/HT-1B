import json
import numpy as np
import pytest
import soundfile as sf


def test_stereo_lr_preserves_amplitude_and_alignment(tmp_path):
    from ht1b.audio import load_manifest
    left = np.array([.1,.2,.3,.4,.5])
    right = np.array([0,.05,.1,.15,.2])
    sf.write(tmp_path/'pair.wav', np.column_stack([left,right]), 8000, subtype='FLOAT')
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'schema_version':1,'records':[{'id':'pair','stereo_pair':'pair.wav',
              'delay_samples':1,'controls':{},'split':'train'}]}))
    r = load_manifest(path)[0]
    np.testing.assert_allclose(r.dry,[.1,.2,.3,.4],atol=1e-7)
    np.testing.assert_allclose(r.wet,[.05,.1,.15,.2],atol=1e-7)
    assert r.sample_rate == 8000


def test_bad_pairs_and_duplicate_ids_are_rejected(tmp_path):
    from ht1b.audio import load_manifest
    sf.write(tmp_path/'mono.wav',np.ones(5)*.2,8000)
    path=tmp_path/'manifest.json'
    row={'id':'a','stereo_pair':'mono.wav','controls':{},'split':'train'}
    path.write_text(json.dumps({'schema_version':1,'records':[row]}))
    with pytest.raises(ValueError,match='two channels'): load_manifest(path)
    sf.write(tmp_path/'stereo.wav',np.zeros((5,2)),8000)
    row['stereo_pair']='stereo.wav'
    path.write_text(json.dumps({'schema_version':1,'records':[row,row]}))
    with pytest.raises(ValueError,match='unique'): load_manifest(path)


def test_sample_rate_and_length_mismatch_rejected(tmp_path):
    from ht1b.audio import load_manifest
    sf.write(tmp_path/'x.wav', np.zeros(6),8000)
    sf.write(tmp_path/'y.wav', np.zeros(6),16000)
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'schema_version':1,'records':[{'id':'r','input':'x.wav','output':'y.wav',
                                                        'controls':{},'split':'train'}]}))
    with pytest.raises(ValueError,match='sample rate'): load_manifest(path)
    sf.write(tmp_path/'y.wav', np.zeros(4),8000)
    with pytest.raises(ValueError,match='length'): load_manifest(path)


def test_negative_delay_does_not_silently_reset_history(tmp_path):
    from ht1b.audio import load_manifest
    sf.write(tmp_path/'p.wav',np.zeros((5,2)),8000)
    p=tmp_path/'m.json'
    p.write_text(json.dumps({'schema_version':1,'records':[{'id':'x','stereo_pair':'p.wav','delay_samples':-1}]}))
    with pytest.raises(ValueError,match='history'): load_manifest(p)
