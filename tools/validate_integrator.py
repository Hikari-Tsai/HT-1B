"""Gate G1 (s6/docs/plans/2026-11-10-circuit-integrated-hybrid-design.md).

The differentiable integrator (s6.circuit_scan) vs. the implicit reference
solver (ht1b.fast_solver), original circuit parameters, on real test intervals.
Both start at rest at the interval start and see the same Dry. Scored like
the full test (from the first 16-aligned sample >= 49152 after the start).
Pass: integrator-vs-reference ESR < 0.01 on every file, and C3 state error
< 2% of the reference's peak (median over files).

Usage: python tools/validate_integrator.py [output.json]
"""
import json
import sys
import time

import numpy as np
import torch

from ht1b.config import Circuit, Controls
from ht1b.fast_solver import FastCircuitSolver
from pinn.evaluate_full_test import MANIFEST, read_intervals, span
from s6.circuit_scan import CircuitScan, POTS

DIVISORS = (16, 8, 4)  # fastest first
rows = sorted(json.loads(MANIFEST.read_text())['records'], key=lambda r: r['id'])  # every file has one test interval
cells = lambda r: (abs(r['labels']['threshold_db']), r['labels']['attack_index'])
# 12 hard cases (threshold -40, attack 0/1), 8 spread over the rest, fixed choice.
hard = [r for r in rows if cells(r)[0] == 40 and cells(r)[1] <= 1][::5][:12]
rest = [r for r in rows if r not in hard][::70][:8]
chosen = hard+rest
spans, dry, wet = read_intervals(chosen)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def esr(a, b):
    return float(np.sum((a-b)**2)/max(np.sum(b**2), 1e-12))


reference, ref_states = [], []
for i, r in enumerate(chosen):
    solver = FastCircuitSolver(Circuit(), Controls(**r['controls']), 48000)
    n = spans[i][2]-spans[i][0]
    y, st = np.empty(n), np.empty(n)
    for o in range(0, n, 4800):  # sample the C3 state every 100 ms block end
        y[o:o+4800] = solver.process(dry[i, o:o+4800].astype(np.float64))
        st[o:o+4800] = solver.state[0]
    reference.append(y); ref_states.append(st)
report = dict(files=[r['id'] for r in chosen], device=device, divisors={})
pots = torch.tensor([[r['controls'][k] for k in POTS] for r in chosen], dtype=torch.float64, device=device)
for divisor in DIVISORS:
    started = time.monotonic()
    scan = CircuitScan(divisor=divisor).to(device)
    x = torch.from_numpy(dry).to(device)
    usable = x.shape[1]//divisor*divisor
    state, outs, es = None, [], []
    with torch.no_grad():
        for o in range(0, usable, 1024 if 1024 % divisor == 0 else divisor*64):
            y, st, state = scan(x[:, o:o+1024], pots, state)
            outs.append(y.cpu()); es.append(st[..., 0].repeat_interleave(divisor, 1).cpu())
    out, e = torch.cat(outs, 1).numpy(), torch.cat(es, 1).numpy()
    files = []
    for i, (r, (start, first, stop)) in enumerate(zip(chosen, spans)):
        a, b = first-start, min(stop-start, usable)
        blocks = np.arange(4799, b, 4800)  # the reference stores its state at these 100 ms block ends
        blocks = blocks[blocks >= a]
        files.append(dict(id=r['id'], threshold_db=-cells(r)[0], attack=cells(r)[1],
                          integrator_vs_reference=esr(out[i, a:b], reference[i][a:b]),
                          reference_vs_wet=esr(reference[i][a:b], wet[i, a:b].astype(np.float64)),
                          integrator_vs_wet=esr(out[i, a:b], wet[i, a:b].astype(np.float64)),
                          reference_c3_peak=float(ref_states[i][blocks].max()),
                          c3_error=float(np.max(np.abs(e[i, blocks]-ref_states[i][blocks]))/max(ref_states[i][blocks].max(), 1e-3))))
    worst = max(f['integrator_vs_reference'] for f in files)
    c3 = float(np.median([f['c3_error'] for f in files]))
    report['divisors'][divisor] = dict(seconds=round(time.monotonic()-started, 1), worst_esr=worst, median_c3_error=c3,
                                       passed=bool(worst < .01 and c3 < .02), files=files)
    print(f'divisor {divisor}: worst ESR {worst:.2e}  median C3 error {c3:.2%}  '
          f'{"PASS" if report["divisors"][divisor]["passed"] else "FAIL"}  ({report["divisors"][divisor]["seconds"]} s)', flush=True)
if len(sys.argv) > 1:
    json.dump(report, open(sys.argv[1], 'w'), indent=1)
