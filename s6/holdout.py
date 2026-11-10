"""Knob-combination holdout manifests for the S6+PINN generalization study.

Every CL-1B file is one unique (attack, release, ratio, threshold) cell. A
holdout removes all train and validation intervals of its held-out cells, so
neither optimisation nor checkpoint selection sees them; test intervals are
kept and the analysis scores held-out and in-distribution cells separately.

Schemes (s6/docs/plans/2026-11-06-pinn-generalization-experiments.md):
  K1-K3   three disjoint 20% folds, stratified by threshold x ratio
  T40     threshold -40 dB layer          A0   attack 0 layer
  K1-F10/F25/F50  K1 held out, and training keeps only a nested 10/25/50% of all cells

Usage: python -m s6.holdout <v3 manifest.json> <output dir> [seed]
"""
from collections import defaultdict
import json
import random
import sys
from pathlib import Path

SEED = 20261106
FOLDS = 5
FRACTIONS = (10, 25, 50)


def cell(parameters):
    return (parameters['attack_index'], parameters['release_index'], parameters['ratio'],
            float(parameters['threshold_db']))


def interpolation_folds(cells, seed=SEED, folds=FOLDS):
    """Disjoint folds; each threshold x ratio stratum is spread evenly across folds."""
    strata = defaultdict(list)
    for c in sorted(cells):
        strata[(c[3], c[2])].append(c)
    rng = random.Random(seed)
    result, offset = [set() for _ in range(folds)], 0
    for key in sorted(strata):
        members = strata[key]
        rng.shuffle(members)
        for i, c in enumerate(members):
            result[(offset+i) % folds].add(c)
        offset += len(members)  # rotate so fold sizes stay balanced
    return result


def training_order(cells, seed=SEED):
    """Stratified permutation; any prefix is a balanced subset, so prefixes are nested."""
    folds = interpolation_folds(cells, seed+1, folds=20)
    rng = random.Random(seed+2)
    order = []
    for fold in folds:
        members = sorted(fold)
        rng.shuffle(members)
        order.extend(members)
    return order


def schemes(cells, seed=SEED):
    """name -> (held-out cells, allowed training cells or None for all others)."""
    cells = set(cells)
    folds = interpolation_folds(cells, seed)
    result = {f'K{i+1}': (folds[i], None) for i in range(3)}
    result['T40'] = ({c for c in cells if c[3] == -40.}, None)
    result['A0'] = ({c for c in cells if c[0] == 0}, None)
    order = training_order(cells, seed)
    for percent in FRACTIONS:
        kept = set(order[:round(len(order)*percent/100)])
        result[f'K1-F{percent}'] = (folds[0], kept-folds[0])
    return result


def holdout_manifest(doc, name, heldout, allowed, seed):
    rows = []
    for row in doc['records']:
        c = cell(row['parameters'])
        if row['split'] == 'test' or (c not in heldout and (allowed is None or c in allowed)):
            rows.append(row)
    train = {cell(r['parameters']) for r in rows if r['split'] == 'train'}
    if train & heldout or {cell(r['parameters']) for r in rows if r['split'] == 'validation'} & heldout:
        raise AssertionError('held-out cell leaked into train or validation')
    return dict(doc, split_strategy=doc['split_strategy']+f'+holdout-{name}',
                holdout=dict(scheme=name, seed=seed, heldout_cells=sorted(map(list, heldout)),
                             training_cells=len(train)),
                records=rows)


def write(source, output, seed=SEED):
    source, output = Path(source).resolve(), Path(output).resolve()
    doc = json.loads(source.read_text(encoding='utf-8'))
    cells = {cell(r['parameters']) for r in doc['records']}
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (heldout, allowed) in schemes(cells, seed).items():
        manifest = holdout_manifest(doc, name, heldout, allowed, seed)
        with (output/f'holdout-{name}.json').open('x') as stream:
            json.dump(manifest, stream, indent=1)
        summary[name] = dict(heldout=len(heldout), training_cells=manifest['holdout']['training_cells'],
                             records=len(manifest['records']))
    return summary


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    print(json.dumps(write(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else SEED), indent=1))
