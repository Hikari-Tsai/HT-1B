import json
import subprocess
import sys
import soundfile as sf


def run(*args):
    return subprocess.run([sys.executable,'-m','ht1b',*map(str,args)],capture_output=True,text=True,check=True)


def test_demo_simulate_evaluate(tmp_path):
    run('demo-data',tmp_path/'demo','--sample-rate',2000,'--seconds',.04)
    manifest=tmp_path/'demo'/'manifest.json'
    assert manifest.exists()
    x,sr=sf.read(tmp_path/'demo'/'train.wav')
    assert x.shape==(80,2)
    out=tmp_path/'processed.wav'
    run('simulate',tmp_path/'demo'/'validation.wav',out,'--channel','left')
    y,yr=sf.read(out)
    assert yr==sr and y.shape==(80,)
    report=tmp_path/'metrics.json'
    run('evaluate',manifest,'--output',report,'--split','validation')
    data=json.loads(report.read_text())
    assert list(data['records'])==['validation']
    assert data['records']['validation']['esr']<1e-9
