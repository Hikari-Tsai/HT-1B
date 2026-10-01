import numpy as np
import soundfile as sf
import json
import pytest
from ht1b.corpus import blocks,batch_bounds,load_corpus,read_block


def test_exhaustive_train_blocks_do_not_read_holdouts_or_drop_tail(tmp_path):
    wav=np.arange(200,dtype=float).reshape(100,2)/200
    sf.write(tmp_path/'pair.wav',wav,48000,subtype='FLOAT')
    row={'id':'a','path':'pair.wav','frames':100,'sample_rate':48000,'intervals':[
        dict(start_sample=0,stop_sample=31,split='train',group=0),
        dict(start_sample=31,stop_sample=80,split='test',group=1),
        dict(start_sample=80,stop_sample=100,split='train',group=2)]}
    raw={'schema_version':2,'dataset_root':str(tmp_path),'records':[row]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(raw))
    corpus,root=load_corpus(path)
    seen=[]
    for block in blocks(corpus,seconds=.0004):
        x=read_block(root,row,block)
        np.testing.assert_allclose(x,wav[block['start']:block['stop']],atol=1e-7)
        for a,b in batch_bounds(block,7):seen.extend(range(a,b))
    assert seen==list(range(31))+list(range(80,100))
    row['intervals'][2]['start_sample']=81
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='gap'):load_corpus(path)


def test_group_split_leakage_is_rejected(tmp_path):
    sf.write(tmp_path/'a.wav',np.zeros((10,2)),48000)
    row=dict(id='a',path='a.wav',frames=10,sample_rate=48000,intervals=[
        dict(start_sample=0,stop_sample=5,split='train',group=0),
        dict(start_sample=5,stop_sample=10,split='test',group=0)])
    path=tmp_path/'m.json';path.write_text(json.dumps(dict(schema_version=2,dataset_root=str(tmp_path),records=[row])))
    with pytest.raises(ValueError,match='leaked'):load_corpus(path)
