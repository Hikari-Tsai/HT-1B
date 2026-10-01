"""Run one complete CUDA pass, select on validation, then open the test set once."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        payload=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**extra)
        path=out/'experiment-status.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(payload,indent=2));temp.replace(path)
        print(json.dumps(payload),flush=True)
    def execute(script,flags):
        subprocess.run([sys.executable,'-u',f'pinn/{script}',*map(str,flags)],check=True)
    common=['--manifest',args.manifest,'--dataset',args.dataset]
    shutil.copy2(args.manifest,out/'manifest.json')
    shutil.copy2('configs/default.json',out/'initial-config.json')
    try:
        status('training',note='One exhaustive pass through every train sample; no fixed 5000-step cutoff.')
        execute('train_full_corpus.py',[*common,'--output',out/'training','--config',out/'initial-config.json','--epochs',1])
        coverage=json.loads((out/'training/coverage.json').read_text())
        if coverage['frames_covered']!=coverage['frames_per_epoch']:raise RuntimeError('Training coverage not complete')
        status('validation')
        for name,config in [('baseline',out/'initial-config.json'),('fitted',out/'training/fitted.json')]:
            execute('evaluate_full_corpus.py',[*common,'--config',config,'--split','validation','--output',out/f'{name}-validation.json'])
        baseline=json.loads((out/'baseline-validation.json').read_text())
        fitted=json.loads((out/'fitted-validation.json').read_text())
        selected='fitted' if fitted['macro_mean']['esr']<baseline['macro_mean']['esr'] else 'baseline'
        config=out/'training/fitted.json' if selected=='fitted' else out/'initial-config.json'
        shutil.copy2(config,out/'selected-config.json')
        selection=dict(selected=selected,criterion='validation macro mean ESR',baseline=baseline['macro_mean'],fitted=fitted['macro_mean'],test_not_used=True)
        (out/'selection.json').write_text(json.dumps(selection,indent=2))
        status('test',selection=selection)
        # The selection is locked before any test score is computed.
        execute('evaluate_full_corpus.py',[*common,'--config',out/'selected-config.json','--split','test','--output',out/'selected-test.json'])
        result=json.loads((out/'selected-test.json').read_text())
        status('complete',selection=selection,test_macro_mean=result['macro_mean'],test_samples=result['samples_scored'])
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
