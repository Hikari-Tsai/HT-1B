"""Score every method on the fixed common windows, then once on the test split.

Validation re-scores the 1240 common windows and must reproduce the stored
seven-method table before the test windows (one per test interval, same
4096-sample Dry warmup and 1024 scored samples) are touched. The batched v3
S4/S6 checkpoints were chosen on validation ESR only; test never selects.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import time

import numpy as np
import torch

from ht1b.config import Controls, read_config
from pinn.compare_pretrained import AuthorCL1B
from pinn.evaluate_five_models import ROOT, SEQUENCE, METRICS, score, summarize, sha
from s6.models import AudioModel
from s6.pinn_experiment import pots_for
from s6.pinn_model import S6PINN
from s6.representative_experiment import windows, load_batch, predict, write_json, utc, WARMUP, SCORE

OUT = ROOT / "runs/common-test-20261031"
BASE = ROOT / "runs/six-model-common-validation-20260925"
PINN = ROOT / "runs/s6-tfilm-pinn-gpu-20260929/best.pt"
RICCARDO = ROOT / "data/external/optical-compressor/Fully_Conditioned_Black_Box_Model_for_Compression/Models/CL1BModel/Checkpoints/best/weights.h5"
CIRCUITS = {"baseline": "runs/full-corpus-gpu-20260925/initial-config.json",
            "mlp": "runs/full-corpus-gpu-20260925/training/fitted.json",
            "gru": "runs/gru-pinn-gpu-20260925/training/fitted.json"}
SEQUENCES = {"s4": SEQUENCE / "s4_tfilm-best.mdlus", "s6": SEQUENCE / "s6_tfilm-best.mdlus"}
BATCHED = {"s4_v3": ROOT / "runs/s4-lr3e-4-20261029-0840", "s6_v3": ROOT / "runs/s6-lr3e-4-20261029-0840"}
KEYS = (*CIRCUITS, *SEQUENCES, "s6_pinn", "riccardovib", *BATCHED)


def circuit(path, rows, dry, originals):
    from ht1b.fast_solver import FastCircuitSolver  # numba; only circuit scoring needs it
    config, _ = read_config(ROOT / path)
    output = np.empty((len(rows), SCORE))
    for i, row in enumerate(rows):
        solver = FastCircuitSolver(config, Controls(**originals[row["id"]]["controls"]), 48000)
        output[i] = solver.process(dry[i].numpy().astype(np.float64))[WARMUP:]
    return output


@torch.no_grad()
def sequence(path, dry, controls, gain):
    # Match the original S4/S6 evaluator's cuDNN mode for checkpoint parity.
    torch.backends.cudnn.allow_tf32 = True
    model = AudioModel.from_checkpoint(path).cuda().eval()
    output = [predict(model, dry[s].cuda(), controls[s].cuda(), gain[s].cuda()).cpu().numpy()
              for s in (slice(o, o+256) for o in range(0, len(dry), 256))]
    return np.concatenate(output).astype(np.float64)


@torch.no_grad()
def s6_pinn(rows, dry, controls, originals):
    saved = torch.load(PINN, map_location="cuda", weights_only=False)
    model = S6PINN(**saved["config"]["model"]).cuda().eval()
    model.load_state_dict(saved["model"])
    output = []
    for offset in range(0, len(rows), 256):
        s = slice(offset, offset+256)
        predicted, _ = model.predict_window(dry[s].cuda(), controls[s].cuda(), pots_for(rows[s], originals, "cuda"))
        output.append(predicted.cpu().numpy())
    return np.concatenate(output).astype(np.float64)


@torch.inference_mode()
def riccardovib(rows, dry, originals):
    torch.backends.cudnn.allow_tf32 = False
    calibration = json.loads((ROOT / "runs/pretrained-comparison/conditioning-calibration.json").read_text())["selected"]
    if sha(RICCARDO) != json.loads((ROOT / "runs/pretrained-comparison/comparison.json").read_text())["weights_sha256"]:
        raise ValueError("RiccardoVib weights changed")
    author = AuthorCL1B(RICCARDO).float().cuda().eval()
    output = []
    for offset in range(0, len(rows), 64):
        batch = rows[offset:offset+64]
        # Every 16-sample output block has precisely its native 16 Dry history samples.
        x = dry[offset:offset+len(batch), WARMUP-16:].unfold(1, 32, 16).reshape(-1, 32).cuda()
        conditions = []
        for row in batch:
            label = originals[row["id"]]["labels"]
            c = np.array([label["attack_index"]/4, label["release_index"]/4,
                          (label["ratio"]-2)/8, abs(label["threshold_db"])/40], dtype=np.float32)
            if calibration["threshold_flipped"]:
                c[3] = 1-c[3]
            conditions.append(c[calibration["permutation"]])
        c = torch.tensor(np.asarray(conditions), device="cuda").repeat_interleave(SCORE//16, dim=0)
        output.append(author(x, c).reshape(len(batch), SCORE).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def evaluate(split, manifest, status):
    originals = {r["id"]: r for r in manifest["records"]}
    rows = windows(manifest, ROOT / "data/tubetech-cl-1b-v1", split, 0, 1)
    for row in rows:
        if (row["start"]+WARMUP) % 16 or row["stop"] % 16:
            raise ValueError(f"RiccardoVib output grid mismatch: {row['id']}")
    index = [dict(id=r["id"], interval=r["interval"], warmup_start=r["start"],
                  start_sample=r["start"]+WARMUP, stop_sample=r["stop"]) for r in rows]
    signature = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    dry, wet, controls, gain = load_batch(rows, torch.device("cpu"))
    target = wet[:, WARMUP:].numpy()
    predictions, sources = {}, {}
    for key in KEYS:
        status(split=split, model=key, windows=len(rows))
        if key in CIRCUITS:
            predictions[key] = circuit(CIRCUITS[key], rows, dry, originals)
            sources[key] = dict(path=CIRCUITS[key], sha256=sha(ROOT/CIRCUITS[key]))
        elif key in SEQUENCES:
            predictions[key] = sequence(SEQUENCES[key], dry, controls, gain)
            sources[key] = dict(path=str(SEQUENCES[key].relative_to(ROOT)), sha256=sha(SEQUENCES[key]))
        elif key in BATCHED:
            best = json.loads((BATCHED[key] / "best.json").read_text())
            path = BATCHED[key] / best["checkpoint"] / "model.mdlus"
            predictions[key] = sequence(path, dry, controls, gain)
            sources[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), selected_epoch=best["epoch"],
                                selection="sample-weighted ESR over full v3 validation intervals")
        elif key == "s6_pinn":
            predictions[key] = s6_pinn(rows, dry, controls, originals)
            sources[key] = dict(path=str(PINN.relative_to(ROOT)), sha256=sha(PINN))
        else:
            predictions[key] = riccardovib(rows, dry, originals)
            sources[key] = dict(path=str(RICCARDO.relative_to(ROOT)), sha256=sha(RICCARDO))
        torch.cuda.empty_cache()
    report = dict(split=split, manifest_signature=manifest["signature"], window_signature=signature,
                  files=len({r["id"] for r in rows}), windows=len(rows), warmup_samples=WARMUP,
                  score_samples=SCORE, gpu=torch.cuda.get_device_name(0), sources=sources,
                  macro_mean={}, pooled={}, median_esr={}, records={})
    for key in KEYS:
        scores = [score(predictions[key][i], target[i]) for i in range(len(rows))]
        grouped = defaultdict(list)
        for row, values in zip(rows, scores):
            grouped[row["id"]].append(values)
        records = {name: summarize(values) for name, values in grouped.items()}
        report["records"][key] = records
        report["macro_mean"][key] = {m: float(np.mean([r[m] for r in records.values()])) for m in METRICS}
        report["pooled"][key] = summarize(scores)
        report["median_esr"][key] = float(np.median([r["esr"] for r in records.values()]))
    return report


def run():
    if (OUT / "test.json").exists():
        raise FileExistsError("Test scores already exist; use the fixed results")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()

    def status(**more):
        value = dict(utc=utc(), seconds=round(time.monotonic()-started, 1), **more)
        write_json(OUT / "status.json", value)
        print(json.dumps(value), flush=True)

    manifest = json.loads((ROOT / "runs/full-corpus-gpu-20260925/manifest.json").read_text())
    stored = json.loads((BASE / "comparison.json").read_text())
    validation = evaluate("validation", manifest, status)
    if validation["window_signature"] != stored["window_signature"]:
        raise ValueError("Validation windows differ from the common comparison")
    for key, values in stored["macro_mean"].items():
        for metric in ("esr", "mse", "mae"):
            np.testing.assert_allclose(validation["macro_mean"][key][metric], values[metric], rtol=2e-4, atol=1e-10,
                                       err_msg=f"{key} {metric} does not reproduce")
    write_json(OUT / "validation.json", validation | dict(completed_utc=utc()))
    test = evaluate("test", manifest, status)
    write_json(OUT / "test.json", test | dict(completed_utc=utc()))
    status(phase="complete", validation_esr={k: v["esr"] for k, v in validation["macro_mean"].items()},
           test_esr={k: v["esr"] for k, v in test["macro_mean"].items()})


if __name__ == "__main__":
    run()
