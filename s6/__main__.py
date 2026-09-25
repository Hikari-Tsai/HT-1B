"""Separate sequence-model CLI; never trains on import or with `check`."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='HT-1B PhysicsNeMo S4/S6 stereo dry-wet training')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Create an aligned stereo manifest with held-out data')
    prepare.add_argument('files', nargs='+')
    prepare.add_argument('--validation-files', nargs='+')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--sample-rate', type=int, default=48000)
    prepare.add_argument('--validation-fraction', type=float, default=.2)
    prepare.add_argument('--gap-seconds', type=float, default=1.)
    prepare.add_argument('--delay-samples', type=int, default=0)
    for command in ('check', 'train'):
        p = sub.add_parser(command, help='Read-only forward check; no optimizer' if command == 'check'
                           else 'Explicitly start chronological TBPTT training')
        p.add_argument('manifest')
        p.add_argument('--config', required=True)
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        if command == 'train':
            p.add_argument('--output', required=True)
            p.add_argument('--resume', help='Latest completed epoch directory')
    evaluate = sub.add_parser('evaluate', help='Score a completed checkpoint without training')
    evaluate.add_argument('manifest')
    evaluate.add_argument('--checkpoint', required=True, help='Epoch directory containing model.mdlus and training.pt')
    evaluate.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    evaluate.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    evaluate.add_argument('--output', help='Optional new JSON report; otherwise print only')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            import math
            from .data import prepare_manifest
            if args.sample_rate < 1 or not math.isfinite(args.gap_seconds) or args.gap_seconds < 0:
                raise ValueError('Use a positive sample rate and finite nonnegative gap')
            result = prepare_manifest(args.files, args.output, sample_rate=args.sample_rate,
                                      validation_files=args.validation_files,
                                      validation_fraction=args.validation_fraction,
                                      gap_samples=round(args.gap_seconds*args.sample_rate),
                                      delay_samples=args.delay_samples)
            print(result)
        elif args.command == 'check':
            from .training import check
            print(json.dumps(check(args.manifest, args.config, args.device), indent=2))
        elif args.command == 'train':
            from .training import train
            train(args.manifest, args.config, args.output, device=args.device, resume=args.resume)
        else:
            from .training import evaluate_checkpoint
            if args.output and Path(args.output).exists():
                raise ValueError('Report already exists; choose a new output path')
            report = evaluate_checkpoint(args.manifest, args.checkpoint, split=args.split, device=args.device)
            text = json.dumps(report, indent=2, allow_nan=False)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as stream:
                    stream.write(text+'\n')
            print(text)
    except ImportError as error:
        parser.exit(2, f's6: {error}; install the train extra: uv sync --locked --extra train --extra test\n')
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f's6: {error}\n')


if __name__ == '__main__':
    main()
