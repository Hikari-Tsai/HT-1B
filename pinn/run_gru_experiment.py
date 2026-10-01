"""Run exhaustive causal GRU + PhysicsNeMo training and matched evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main(args: argparse.Namespace) -> None:
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    out = args.output
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if ((out / 'manifest.json').read_bytes() != args.manifest.read_bytes() or
                (out / 'initial-config.json').read_bytes() != Path('configs/default.json').read_bytes()):
            raise ValueError('Resume manifest or initial circuit changed')
    else:
        shutil.copy2(args.manifest, out / 'manifest.json')
        shutil.copy2('configs/default.json', out / 'initial-config.json')

    def status(phase: str, **extra: object) -> None:
        value = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = out / 'experiment-status.tmp'
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(out / 'experiment-status.json')
        print(json.dumps(value), flush=True)

    def execute(script: str, *flags: object) -> None:
        subprocess.run([sys.executable, '-u', f'pinn/{script}', *map(str, flags)], check=True)

    common = ('--manifest', out / 'manifest.json', '--dataset', args.dataset)
    try:
        status('training', note='One exhaustive CUDA pass; GRU uses Dry history across all partitions.')
        prior_status = out / 'training/status.json'
        done = (prior_status.exists() and
                json.loads(prior_status.read_text(encoding='utf-8'))['phase'] == 'training_complete')
        if not done:
            train_flags = ('--resume',) if args.resume and (out / 'training/last.pt').exists() else ()
            execute('train_gru_pinn.py', *common, '--config', out / 'initial-config.json',
                    '--output', out / 'training', *train_flags)
        coverage = json.loads((out / 'training/coverage.json').read_text(encoding='utf-8'))
        if coverage['frames_covered'] != coverage['frames_per_epoch'] or coverage['holdout_wet_training_samples'] != 0:
            raise RuntimeError('GRU training coverage or holdout integrity check failed')
        status('validation')
        if not (out / 'gru-validation.json').exists():
            execute('evaluate_full_corpus.py', *common, '--config', out / 'training/fitted.json',
                    '--split', 'validation', '--output', out / 'gru-validation.json')
        gru = json.loads((out / 'gru-validation.json').read_text(encoding='utf-8'))
        previous_run = Path('runs/full-corpus-gpu-20260925')
        mlp = json.loads((previous_run / 'fitted-validation.json').read_text(encoding='utf-8'))
        baseline = json.loads((previous_run / 'baseline-validation.json').read_text(encoding='utf-8'))
        if len({gru['manifest_signature'], mlp['manifest_signature'], baseline['manifest_signature']}) != 1:
            raise RuntimeError('GRU and MLP evaluations use different corpora')
        comparison = dict(criterion='validation macro mean ESR',
            gru=gru['macro_mean'], mlp=mlp['macro_mean'], baseline=baseline['macro_mean'],
            validation_winner=min(('gru', 'mlp', 'baseline'),
                key=lambda name: {'gru':gru, 'mlp':mlp, 'baseline':baseline}[name]['macro_mean']['esr']),
            test_scope='Previously opened holdout; exploratory architecture comparison, not a fresh independent test.')
        (out / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        status('test', validation_comparison=comparison)
        if not (out / 'gru-test.json').exists():
            execute('evaluate_full_corpus.py', *common, '--config', out / 'training/fitted.json',
                    '--split', 'test', '--output', out / 'gru-test.json')
        test = json.loads((out / 'gru-test.json').read_text(encoding='utf-8'))
        status('complete', validation_comparison=comparison,
               gru_test_macro_mean=test['macro_mean'], test_samples=test['samples_scored'])
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
