"""Fixed-checkpoint comparison on identical short validation windows for five models.

Run from the repository: python -m pinn.evaluate_five_models
No optimizer, checkpoint selection, gain fitting or test-set access occurs here.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ht1b.audio_metrics import frame_analysis, region_metrics, stft_magnitude
from ht1b.config import Controls, read_config
from ht1b.fast_solver import FastCircuitSolver
from pinn.compare_pretrained import AuthorCL1B
from s6.models import AudioModel
from s6.representative_experiment import windows, load_batch, predict, write_json, utc, WARMUP, SCORE, WINDOW

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/five-model-common-validation-20260925"
SEQUENCE = ROOT / "runs/s4-s6-representative-gpu-20260925"
KEYS = ("mlp", "gru", "s4", "s6", "riccardovib")
METRICS = ("esr", "mse", "mae", "mrstft", "rms_db_mae")
FFT_SIZES = (256, 512, 1024)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(prediction, target):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != (SCORE,) or t.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Invalid common scoring window")
    error = p - t
    spectral = []
    for size in FFT_SIZES:
        ps, ts = stft_magnitude(p, size), stft_magnitude(t, size)
        spectral.append(float(np.linalg.norm(ps-ts) / np.linalg.norm(ts) + np.mean(np.abs(np.log(ps)-np.log(ts)))))
    frame = frame_analysis({"reference_wet": t, "prediction": p})
    energy, squared, absolute = float(t @ t), float(error @ error), float(np.abs(error).sum())
    return dict(squared_error=squared, absolute_error=absolute, target_energy=energy,
                esr=squared/max(energy, SCORE*1e-8), mse=squared/SCORE, mae=absolute/SCORE,
                mrstft=float(np.mean(spectral)), rms_db_mae=region_metrics(frame, "prediction")["rms_db_mae"])


def summarize(rows):
    samples = len(rows)*SCORE
    squared = sum(r["squared_error"] for r in rows)
    energy = sum(r["target_energy"] for r in rows)
    return dict(esr=squared/max(energy, samples*1e-8), mse=squared/samples,
                mae=sum(r["absolute_error"] for r in rows)/samples,
                mrstft=float(np.mean([r["mrstft"] for r in rows])),
                rms_db_mae=float(np.mean([r["rms_db_mae"] for r in rows])))


def run():
    if (OUT / "comparison.json").exists():
        raise FileExistsError("Common comparison already exists; use its fixed results")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the original S4/S6 evaluator's cuDNN mode for checkpoint parity.
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()

    def status(phase, **more):
        value = dict(phase=phase, utc=utc(), seconds=round(time.monotonic()-started, 2), **more)
        write_json(OUT / "status.json", value)
        print(json.dumps(value), flush=True)

    try:
        manifest = json.loads((ROOT / "runs/full-corpus-gpu-20260925/manifest.json").read_text())
        protocol = json.loads((SEQUENCE / "protocol.json").read_text())
        if manifest["signature"] != protocol["manifest_signature"]:
            raise ValueError("Manifest mismatch")
        source_rows = {r["id"]: r for r in manifest["records"]}
        rows = windows(manifest, ROOT / "data/tubetech-cl-1b-v1", "validation", 0, protocol["epochs"])
        if len(rows) != 1240 or len({r["id"] for r in rows}) != 620:
            raise ValueError("Incomplete common validation windows")
        for row in rows:
            if (row["start"]+WARMUP) % 16 or row["stop"] % 16:
                raise ValueError("RiccardoVib output grid mismatch")
        status("loading", windows=len(rows))
        dry, wet, controls, gain = load_batch(rows, torch.device("cpu"))
        target = wet[:, WARMUP:].numpy()
        predictions = {}
        sources = {}
        for key, rel in (("mlp", "runs/full-corpus-gpu-20260925/training/fitted.json"),
                         ("gru", "runs/gru-pinn-gpu-20260925/training/fitted.json")):
            status("inference", model=key, completed=0, total=len(rows))
            path = ROOT / rel
            circuit, _ = read_config(path)
            output = np.empty_like(target, dtype=np.float64)
            for i, row in enumerate(rows):
                solver = FastCircuitSolver(circuit, Controls(**source_rows[row["id"]]["controls"]), 48000)
                output[i] = solver.process(dry[i].numpy().astype(np.float64))[WARMUP:]
                if (i+1) % 200 == 0:
                    status("inference", model=key, completed=i+1, total=len(rows))
            predictions[key] = output
            sources[key] = dict(path=rel, sha256=sha(path), method="fitted circuit; zero state then common Dry warmup")
        for key, arch in (("s4", "s4_tfilm"), ("s6", "s6_tfilm")):
            path = SEQUENCE / f"{arch}-best.mdlus"
            model = AudioModel.from_checkpoint(path).cuda().eval()
            outputs = []
            with torch.no_grad():
                for offset in range(0, len(rows), 256):
                    sl = slice(offset, offset+256)
                    outputs.append(predict(model, dry[sl].cuda(), controls[sl].cuda(), gain[sl].cuda()).cpu().numpy())
                    status("inference", model=key, completed=min(offset+256,len(rows)), total=len(rows))
            predictions[key] = np.concatenate(outputs).astype(np.float64)
            selected = json.loads((SEQUENCE / f"{arch}-best-validation.json").read_text())
            sources[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                selected_epoch=selected["selected_epoch"], method="fixed best checkpoint; common Dry warmup")
            del model
            torch.cuda.empty_cache()
        weights = ROOT / "data/external/optical-compressor/Fully_Conditioned_Black_Box_Model_for_Compression/Models/CL1BModel/Checkpoints/best/weights.h5"
        calibration = json.loads((ROOT / "runs/pretrained-comparison/conditioning-calibration.json").read_text())["selected"]
        original = json.loads((ROOT / "runs/pretrained-comparison/comparison.json").read_text())
        if sha(weights) != original["weights_sha256"]:
            raise ValueError("RiccardoVib weights changed")
        torch.backends.cudnn.allow_tf32 = False
        author = AuthorCL1B(weights).float().cuda().eval()
        reference = np.load(ROOT / "runs/pretrained-comparison/tensorflow-reference.npz")
        with torch.inference_mode():
            check = author(torch.tensor(reference["windows"], dtype=torch.float32, device="cuda"),
                           torch.tensor(reference["conditioning"], dtype=torch.float32, device="cuda")).cpu().numpy()
        np.testing.assert_allclose(check, reference["predictions"], rtol=1e-4, atol=1e-5)
        output = []
        with torch.inference_mode():
            for offset in range(0, len(rows), 64):
                batch = rows[offset:offset+64]
                # Every 16-sample output block has precisely its native 16 Dry history samples.
                x = dry[offset:offset+len(batch), WARMUP-16:].unfold(1,32,16).reshape(-1,32).cuda()
                conditions = []
                for row in batch:
                    label = source_rows[row["id"]]["labels"]
                    c = np.array([label["attack_index"]/4,label["release_index"]/4,
                                  (label["ratio"]-2)/8,abs(label["threshold_db"])/40],dtype=np.float32)
                    if calibration["threshold_flipped"]:
                        c[3] = 1-c[3]
                    conditions.append(c[calibration["permutation"]])
                c = torch.tensor(np.asarray(conditions), device="cuda").repeat_interleave(SCORE//16,dim=0)
                output.append(author(x,c).reshape(len(batch), SCORE).cpu().numpy())
                status("inference", model="riccardovib", completed=offset+len(batch), total=len(rows))
        predictions["riccardovib"] = np.concatenate(output).astype(np.float64)
        sources["riccardovib"] = dict(path=str(weights.relative_to(ROOT)), sha256=sha(weights),
                                     conditioning=calibration, method="fixed pretrained model; native 16-sample history per output block")
        del author
        torch.cuda.empty_cache()
        per_model = {}
        target_energies = defaultdict(float)
        for row, signal in zip(rows, target):
            target_energies[row["id"]] += float(np.square(signal.astype(np.float64)).sum())
        for key in KEYS:
            status("metrics", model=key)
            scores = [score(predictions[key][i], target[i]) for i in range(len(rows))]
            grouped = defaultdict(list)
            for row, values in zip(rows, scores):
                grouped[row["id"]].append(values)
            per_model[key] = dict(records={name:summarize(values) for name,values in grouped.items()},
                                  pooled=summarize(scores), intervals=scores)
            per_model[key]["macro_mean"] = {m:float(np.mean([r[m] for r in per_model[key]["records"].values()])) for m in METRICS}
            # Re-inference must reproduce the selected S4/S6 waveform metrics.
            if key in ("s4", "s6"):
                prior = json.loads((SEQUENCE / f"{key}_tfilm-best-validation.json").read_text())
                for metric in ("esr", "mse", "mae"):
                    np.testing.assert_allclose(per_model[key]["macro_mean"][metric], prior["macro_mean"][metric], rtol=2e-4, atol=1e-10)
        index = [dict(id=r["id"], interval=r["interval"], warmup_start=r["start"],
                      start_sample=r["start"]+WARMUP, stop_sample=r["stop"]) for r in rows]
        window_hash = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
        report = dict(version=1, completed_utc=utc(), manifest_signature=manifest["signature"],
                      window_signature=window_hash, split="validation", files=620, windows=1240,
                      samples_scored=1240*SCORE, warmup_samples=WARMUP, score_samples=SCORE,
                      sample_rate=48000, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                      protocol=dict(esr_floor=1e-8, fft_sizes=list(FFT_SIZES),
                                    spectral="mean(SC + log-magnitude L1); periodic Hann; center reflect; hop=n/4; power floor 1e-8",
                                    rms="480-sample frames; final partial frame; -100 dBFS floor",
                                    state="Reset per window; up to 4096 preceding Dry samples; no Wet state fitting",
                                    riccardovib_history="Native 16-sample history per 16-sample output block",
                                    scope="Previously used validation windows; unequal training budgets; RiccardoVib source overlap unknown"),
                      sources=sources, macro_mean={k:per_model[k]["macro_mean"] for k in KEYS},
                      pooled={k:per_model[k]["pooled"] for k in KEYS},
                      records=[dict(id=name, labels=source_rows[name]["labels"], samples=2*SCORE,
                                    target_energy=target_energies[name],
                                    metrics={k:per_model[k]["records"][name] for k in KEYS}) for name in sorted(source_rows)],
                      intervals=[dict(**row, metrics={k:{m:per_model[k]["intervals"][i][m] for m in METRICS} for k in KEYS}) for i,row in enumerate(index)])
        write_json(OUT / "windows.json", dict(window_signature=window_hash, windows=index))
        write_json(OUT / "comparison.json", report)
        status("complete", macro_mean=report["macro_mean"])
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    run()
