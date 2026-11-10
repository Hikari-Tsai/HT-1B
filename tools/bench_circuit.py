"""Gate G2: time one training step (forward + backward) of S6Circuit vs. S6.

Batch and chunk as the generalization configs (128 lanes x 1024 samples),
model size from s6/configs/generalization/s6-seed7.json.
Usage: python tools/bench_circuit.py [batch] [steps]
"""
import json
import os
import sys
import time

import torch

from s6.batched import _accelerate
from s6.hybrid_model import S6Circuit
from s6.models import AudioModel

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
model_options = json.load(open('s6/configs/generalization/s6-seed7.json'))['model']
device = torch.device('cuda')
torch.manual_seed(0)
x = torch.randn(batch, 1024, device=device)*.2
controls = torch.rand(batch, 4, device=device)
pots = torch.rand(batch, 5, device=device, dtype=torch.float64)
pots[:, 4] = 0


def bench(name, make, run):
    model = _accelerate(make().to(device), device)
    state = None
    for i in range(steps+2):  # two warm-up steps (compilation, autotuning)
        if i == 2:
            torch.cuda.synchronize(); started = time.perf_counter()
        y, state = run(model, state)
        y.square().mean().backward()
        state = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()} if isinstance(state, dict) else state
    torch.cuda.synchronize()
    seconds = (time.perf_counter()-started)/steps
    print(f'{name:28s} {seconds*1e3:8.1f} ms/step  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
    torch.cuda.reset_peak_memory_stats()
    return seconds


def detach(state):
    from s6.layers import detach_state
    return detach_state(state)


s6 = bench('S6 (bounded gain)', lambda: AudioModel(**model_options),
           lambda m, s: (lambda y, st: (y, detach(st)))(*m(x, controls, s)))
results = dict(s6=s6)
for compiled in ('0', '1'):
    os.environ['S6_COMPILE_CIRCUIT'] = compiled
    options = {**model_options, 'architecture': 's6_tfilm'}
    options.pop('output_gain', None)
    results[f'circuit_compile{compiled}'] = bench(
        f'S6Circuit, compile={compiled}', lambda: S6Circuit('audio', **options),
        lambda m, s: (lambda y, _, st: (y, detach(st)))(*m.stream(x, controls, pots, s)))
print(json.dumps({k: round(v/results['s6'], 2) for k, v in results.items()}))
