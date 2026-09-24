import json
import numpy as np
import pytest
import soundfile as sf
pytest.importorskip('physicsnemo')
from ht1b.config import Circuit, Controls
from ht1b.solver import CircuitSolver


def test_training_checkpoint_resume_and_export(tmp_path):
    from ht1b.training import train, export_checkpoint
    sr=2000
    x=.1*np.sin(np.arange(80)*.24)
    y=CircuitSolver(Circuit(),Controls(),sr).process(x)
    sf.write(tmp_path/'pair.wav',np.column_stack([x,y]),sr,subtype='FLOAT')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':1,'records':[{
        'id':'train','stereo_pair':'pair.wav','controls':{},'split':'train'}]}))
    options=dict(steps=12,batch_size=48,width=16,layers=2,learning_rate=.003,seed=42,device='cpu')
    report=train(manifest,tmp_path/'run',Circuit(),**options)
    assert np.isfinite(report['last_loss'])
    assert report['last_loss'] < report['first_loss']
    checkpoint=tmp_path/'run'/'last.pt'
    assert checkpoint.exists()
    exported=export_checkpoint(checkpoint,tmp_path/'fit.json')
    assert exported.gre_span>0
    more=train(manifest,tmp_path/'run',Circuit(),resume=checkpoint,**{**options,'steps':2})
    assert more['step']==14
    prediction=CircuitSolver(exported,Controls(),sr).process(x)
    assert np.isfinite(prediction).all()


def test_resume_matches_uninterrupted_optimization(tmp_path):
    from ht1b.training import train
    import torch
    x=.1*np.sin(np.arange(32)*.3)
    sf.write(tmp_path/'p.wav',np.column_stack([x,.7*x]),2000,subtype='FLOAT')
    manifest=tmp_path/'m.json'
    manifest.write_text(json.dumps({'schema_version':1,'records':[{'id':'r','stereo_pair':'p.wav','controls':{}}]}))
    options=dict(batch_size=16,width=8,layers=1,learning_rate=.01,seed=1)
    train(manifest,tmp_path/'whole',Circuit(),steps=4,**options)
    train(manifest,tmp_path/'split',Circuit(),steps=2,**options)
    train(manifest,tmp_path/'split',Circuit(),steps=2,resume=tmp_path/'split'/'last.pt',**options)
    a=torch.load(tmp_path/'whole'/'last.pt',weights_only=True)
    b=torch.load(tmp_path/'split'/'last.pt',weights_only=True)
    for key in a['physical']:
        torch.testing.assert_close(a['physical'][key],b['physical'][key],atol=1e-13,rtol=1e-13)
