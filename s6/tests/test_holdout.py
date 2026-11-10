"""Knob-combination holdout splits: disjoint, balanced, leak-free and nested."""
from itertools import product
import json

import pytest

from s6.holdout import interpolation_folds, schemes, holdout_manifest, training_order, cell, write

GRID = sorted(set(product(range(5), range(5), (2, 4, 6, 8, 10), (0., -10., -20., -30., -40.))) -
              {(0, 4, 6, -40.), (2, 1, 2, -30.)})


def test_folds_are_disjoint_balanced_and_stratified():
    folds = interpolation_folds(GRID)
    assert set().union(*folds) == set(GRID)
    assert sum(map(len, folds)) == len(GRID)
    assert max(map(len, folds))-min(map(len, folds)) <= 1
    for fold in folds:
        thresholds = [sum(c[3] == t for c in fold) for t in (0., -10., -20., -30., -40.)]
        assert max(thresholds)-min(thresholds) <= 2


def test_schemes_are_deterministic_and_select_the_named_layers():
    a, b = schemes(GRID), schemes(GRID)
    assert a == b
    assert a['T40'][0] == {c for c in GRID if c[3] == -40.}
    assert a['A0'][0] == {c for c in GRID if c[0] == 0}
    assert not a['K1'][0] & a['K2'][0] and not a['K2'][0] & a['K3'][0]


def test_training_fractions_are_nested_and_exclude_k1():
    s = schemes(GRID)
    f10, f25, f50 = (s[f'K1-F{p}'][1] for p in (10, 25, 50))
    assert f10 < f25 < f50
    assert not f50 & s['K1'][0]
    order = training_order(GRID)
    assert sorted(order) == GRID


def manifest():
    rows = []
    for c in GRID[:40]:
        params = dict(attack_index=c[0], release_index=c[1], ratio=c[2], threshold_db=c[3], gain_db=0.)
        for split in ('train', 'validation', 'test'):
            rows.append(dict(id=f'{c}-{split}', split=split, parameters=params))
    return dict(schema_version=1, split_strategy='v3', records=rows)


def test_manifest_drops_heldout_train_and_validation_but_keeps_all_test():
    doc = manifest()
    heldout = {cell(doc['records'][0]['parameters']), cell(doc['records'][3]['parameters'])}
    out = holdout_manifest(doc, 'X', heldout, None, 1)
    assert sum(r['split'] == 'test' for r in out['records']) == 40
    assert not {cell(r['parameters']) for r in out['records'] if r['split'] != 'test'} & heldout
    assert out['holdout']['training_cells'] == 38


def test_write_refuses_to_overwrite(tmp_path):
    source = tmp_path/'v3.json'
    source.write_text(json.dumps(manifest()))
    summary = write(source, tmp_path/'out')
    assert set(summary) == {'K1', 'K2', 'K3', 'T40', 'A0', 'K1-F10', 'K1-F25', 'K1-F50'}
    with pytest.raises(FileExistsError):
        write(source, tmp_path/'out')
