import pytest
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
