import json
import numpy as np
import pytest
import soundfile as sf
from ht1b.config import Circuit


def test_filename_two_r_fields_and_pot_mapping():
    from ht1b.dataset import parse_filename, dataset_controls
    labels=parse_filename('TubeTech_a_0_r_0_r_10_t_0_g_0.wav')
    assert labels==dict(attack_index=0,release_index=0,ratio=10,threshold_db=0,gain_db=0)
    c,notes=dataset_controls(labels,Circuit())
    assert c.attack==0 and c.release==0 and c.ratio==1
    assert c.makeup_db==0 and c.mode=='manual'
    assert notes
    assert 0<c.threshold<1
    slow=dataset_controls(dict(attack_index=4,release_index=4,ratio=2,threshold_db=-40,gain_db=0),Circuit())[0]
    assert slow.attack==1 and slow.release==1 and slow.ratio==0 and slow.threshold==1
    with pytest.raises(ValueError): parse_filename('TubeTech_a_7_r_0_r_10_t_0_g_0.wav')


@pytest.mark.parametrize('threshold', [0, 10, 20, 30, 40])
@pytest.mark.parametrize('gain', [-6, 0, 6])
def test_kaggle_unsigned_threshold_matches_signed_controls(threshold, gain):
    from ht1b.dataset import parse_filename, dataset_controls
    unsigned=parse_filename(f'TubeTech_a_2_r_3_r_6_t_{threshold}_g_{gain}.wav')
    signed=parse_filename(f'TubeTech_a_2_r_3_r_6_t_{-threshold}_g_{gain}.wav')
    assert unsigned==signed
    assert unsigned['threshold_db']==-threshold
    assert unsigned['gain_db']==gain
    controls,_=dataset_controls(unsigned,Circuit())
    assert controls==dataset_controls(signed,Circuit())[0]
    assert 0<=controls.threshold<=1
    assert controls.makeup_db==gain


@pytest.mark.parametrize('threshold', ['5', '-5', '15', '-15', '50', '-50', '10.5', '-10.5'])
def test_filename_rejects_threshold_outside_five_settings(threshold):
    from ht1b.dataset import parse_filename
    with pytest.raises(ValueError,match='outside the documented five-setting dataset'):
        parse_filename(f'TubeTech_a_0_r_0_r_10_t_{threshold}_g_0.wav')


def test_prepare_preserves_unsigned_source_and_normalizes_threshold(tmp_path):
    from ht1b.audio import load_manifest
    from ht1b.dataset import prepare_manifest
    audio=np.column_stack([np.linspace(-.5,.5,32),np.linspace(-.1,.1,32)])
    sources=[tmp_path/f'TubeTech_a_0_r_0_r_10_t_{label}_g_0.wav' for label in (20,-20)]
    for source in sources:
        sf.write(source,audio,48000,subtype='FLOAT')
    original_bytes=[source.read_bytes() for source in sources]
    manifest=prepare_manifest(sources,tmp_path/'manifest.json')
    rows=json.loads(manifest.read_text())['records']
    assert rows[0]['controls']==rows[1]['controls']
    for source,row in zip(sources,rows):
        assert row['stereo_pair']==str(source.resolve())
        assert row['metadata']['original_filename']==source.name
        assert row['metadata']['filename_parameters']['threshold_db']==-20
    records=load_manifest(manifest)
    assert len(records)==2
    for record in records:
        np.testing.assert_allclose(record.dry,audio[:,0],atol=1e-7)
        np.testing.assert_allclose(record.wet,audio[:,1],atol=1e-7)
    assert [source.read_bytes() for source in sources]==original_bytes
