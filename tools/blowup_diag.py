"""Why do T40-held-out S6+PINN checkpoints blow up on threshold -40 dB files?

Files: every held-out file where either PINN seed has ESR > 1, plus the same
(attack, release, ratio) at threshold -30 dB (trained). Models are streamed
with the full-test protocol. Per file and model: signed output level error,
error concentration in time, onset coincidence, e/f/s reach, ODE residuals.
Also the learned circuit parameters of each PINN checkpoint vs. their prior.
"""
import json
import re
import numpy as np
import torch
from pinn.evaluate_full_test import MANIFEST, read_intervals, stream, CHUNK
from s6.data import encode_labels
from s6.fused_scan import accelerate_training
from s6.models import AudioModel
from s6.pinn_experiment import pots_for
from s6.pinn_model import S6PINN
from ht1b.physicsnemo_model import BOUNDS

torch.backends.cuda.matmul.allow_tf32 = False
RUNS = {'pinn_s7': 'runs/t40-pinn-s7-20261106-1436/checkpoints/epoch-000013',
        'pinn_s8': 'runs/t40-pinn-s8-20261106-1436/checkpoints/epoch-000016',
        'struct_s7': 'runs/t40-struct-s7-20261106-1436/checkpoints/epoch-000013',
        's6_s8': 'runs/t40-s6-s8-20261106-1436/checkpoints/epoch-000011',
        'pinn_full': 'runs/s6-pinn-ramp-20261104-1442/checkpoints/epoch-000020'}
test = json.load(open('runs/full-test-t40-pinn-struct-20261110/report.json'))['records']
bad = sorted({f for m in ('t40_pinn_s7', 't40_pinn_s8') for f, v in test[m].items() if '_t_40_' in f and v['esr'] > 1})
pairs = [f.replace('_t_40_', '_t_30_') for f in bad]
files = bad+pairs
originals = {r['id']: r for r in json.load(open(MANIFEST))['records']}
FRAME = 480  # 10 ms


def frames(x):
    n = len(x)//FRAME
    return np.sqrt(np.mean(x[:n*FRAME].reshape(n, FRAME).astype(np.float64)**2, 1)+1e-12)


def db(x):
    return 20*np.log10(np.maximum(x, 1e-6))


out = dict(files=files, heldout=bad, models=RUNS, records={}, parameters={})
for name, path in RUNS.items():
    model = accelerate_training(AudioModel.from_checkpoint(path+'/model.mdlus').cuda().eval())
    pinn = isinstance(model, S6PINN)
    if pinn:
        p = model.physical.values()
        prior = model.physical_prior
        out['parameters'][name] = {k: dict(value=float(getattr(p, k)),
                                           initial=float(BOUNDS[k][0]+(BOUNDS[k][1]-BOUNDS[k][0])*torch.sigmoid(prior[i])))
                                   for i, k in enumerate(model.physical.raw.keys())}
        captured = []
        original = model.trajectory

        def recording(*a, **k):
            s, r = original(*a, **k)
            captured.append(s.float().cpu())
            return s, r
        model.trajectory = recording
    for i in range(0, len(files), 8):
        recs = [originals[k] for k in files[i:i+8]]
        spans, dry, wet = read_intervals(recs)
        labels = [encode_labels(r['labels']) for r in recs]
        controls = torch.tensor([c for c, _ in labels], dtype=torch.float32, device='cuda')
        gain = torch.tensor([g for _, g in labels], dtype=torch.float32, device='cuda')
        x = torch.from_numpy(dry).cuda()
        if pinn:
            captured.clear()
            pots = pots_for(recs, originals, 'cuda')
            state, ys, res = None, [], []
            with torch.no_grad():
                for o in range(0, x.shape[1], CHUNK):
                    y, r, state = model.stream(x[:, o:o+CHUNK], controls, pots, state)
                    ys.append(y.float().cpu()); res.append({k: v.float().cpu() for k, v in r.items()})
            pred = torch.cat(ys, 1).numpy()
            states = torch.cat(captured, 1).numpy()
            resid = {k: torch.cat([r[k] for r in res], 1).numpy() for k in res[0]}
        else:
            pred = stream(model, x, controls, gain)
        for j, (rec, (start, first, stop)) in enumerate(zip(recs, spans)):
            a, b = first-start, stop-start
            p_, t_, d_ = pred[j, a:b].astype(np.float64), wet[j, a:b].astype(np.float64), dry[j, a:b].astype(np.float64)
            err = p_-t_
            fe, ft, fp, fd = frames(err), frames(t_), frames(p_), frames(d_)
            e2 = fe**2
            worst = np.argsort(e2)[::-1][:max(1, len(e2)//100)]
            # onset: dry frame >= 20 dB above the max of the preceding 200 ms
            prev = np.array([fd[max(0, k-20):k].max() if k else fd[0] for k in range(len(fd))])
            onset = db(fd)-db(prev) >= 20
            near_onset = np.zeros_like(onset)
            for k in np.flatnonzero(onset):
                near_onset[k:k+50] = True  # 500 ms after an onset
            loud = db(fd) > -50
            row = dict(esr=float(e2.sum()/max(ft.__pow__(2).sum(), 1e-12)),
                       level_error_db=float(db(np.sqrt(np.mean(p_**2)))-db(np.sqrt(np.mean(t_**2)))),
                       gain_db_pred=float(np.median((db(fp)-db(fd))[loud])) if loud.any() else None,
                       gain_db_target=float(np.median((db(ft)-db(fd))[loud])) if loud.any() else None,
                       worst1pct_error_share=float(e2[worst].sum()/e2.sum()),
                       worst_near_onset=float(near_onset[worst].mean()), frames_near_onset=float(near_onset.mean()),
                       worst_frame_overshoot_db=float(np.median(db(fp[worst])-db(ft[worst]))))
            if pinn:
                st = states[j, a:b]
                row.update(e_max=float(st[:, 0].max()), e_p50=float(np.median(st[:, 0])),
                           f_max=float(st[:, 1].max()), f_p50=float(np.median(st[:, 1])), s_p50=float(np.median(st[:, 2])),
                           residual={k: float(np.mean(v[j, a:b]**2)) for k, v in resid.items()})
                # states inside the worst frames vs elsewhere
                idx = np.concatenate([np.arange(w*FRAME, (w+1)*FRAME) for w in worst])
                row.update(f_worst=float(np.median(st[idx, 1])), e_worst=float(np.median(st[idx, 0])))
            out['records'].setdefault(rec['id'], {})[name] = row
    print(name, 'done', flush=True)
json.dump(out, open('runs/blowup-diagnosis-20261110.json', 'w'), indent=1)
print('wrote runs/blowup-diagnosis-20261110.json', len(files), 'files')
