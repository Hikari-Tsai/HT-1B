"""Score every method on the complete test intervals over identical samples.

Each test interval is scored from the first 16-aligned sample at least 49152
samples (the v3 TBPTT warmup) after its start, to its last full 16-sample block.
History before that point follows each method's own protocol: circuits run
continuously from file sample 0, sequence models and the S6 PINN start from zero
state at the interval start (Dry only), RiccardoVib sees its native 16 samples.
Silent frames are reported separately: on a near-silent target ESR only
measures the output noise floor.
"""
from __future__ import annotations
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from ht1b.audio import metrics
from ht1b.audio_metrics import mrstft, frame_analysis, region_metrics, METRIC_PROTOCOL
from ht1b.config import Circuit, Controls, read_config
from pinn.compare_pretrained import AuthorCL1B
from s6.data import encode_labels
from s6.fused_scan import accelerate_training
from s6.models import AudioModel
from s6.pinn_experiment import pots_for
from s6.pinn_model import S6PINN
from s6.representative_experiment import write_json, utc

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs/full-corpus-gpu-20260925/manifest.json"
DATASET = ROOT / "data/tubetech-cl-1b-v1"
WARMUP = 49152
CHUNK = 1024
AUDIBLE_DB = -60.  # 10 ms target frames below this are reported as silent
MIN_AUDIBLE_SECONDS = 1.
METRICS = ("esr", "mse", "mae", "mrstft", "rms_db_mae")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def span(row):
    (interval,) = [i for i in row["intervals"] if i["split"] == "test"]
    start = interval["start_sample"]
    first = -(-(start+WARMUP) // 16) * 16
    stop = first + (interval["stop_sample"]-first) // 16 * 16
    return start, first, stop


def interval_metrics(prediction, target):
    p, t = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prediction")
    values = {k: v for k, v in metrics(p, t).items() if k in ("mse", "mae", "esr", "target_rms")}
    frames = frame_analysis({"reference_wet": t, "prediction": p})
    values.update(mrstft=mrstft(p, t)["total"], rms_db_mae=region_metrics(frames, "prediction")["rms_db_mae"])
    counts = np.asarray(frames["counts"])
    energy = np.asarray(frames["target_energy"])
    error = np.asarray(frames["errors"]["prediction"]["squared_error"])
    audible = 10*np.log10(np.maximum(energy/counts, 1e-20)) > AUDIBLE_DB
    values.update(squared_error=float(error.sum()), target_energy=float(energy.sum()),
                  audible_seconds=float(counts[audible].sum()/48000),
                  audible_squared_error=float(error[audible].sum()),
                  audible_target_energy=float(energy[audible].sum()))
    values["audible_esr"] = (values["audible_squared_error"]/values["audible_target_energy"]
                             if values["audible_seconds"] >= MIN_AUDIBLE_SECONDS else None)
    return values


def circuit_record(task):
    from ht1b.fast_solver import FastCircuitSolver  # numba; only circuit scoring needs it
    key, config, row = task
    start, first, stop = span(row)
    solver = FastCircuitSolver(Circuit(**config), Controls(**row["controls"]), 48000)
    output, position = [], 0
    with sf.SoundFile(DATASET/row["path"]) as source:
        while position < stop:
            block = source.read(min(480000, stop-position), dtype="float64", always_2d=True)
            if not len(block):
                raise ValueError("Unexpected EOF")
            prediction = solver.process(block[:, 0])
            a, b = max(position, first), position+len(block)
            if a < b:
                output.append((prediction[a-position:], block[a-position:, 1]))
            position += len(block)
    p, t = (np.concatenate(x) for x in zip(*output))
    return key, row["id"], interval_metrics(p, t)


def read_intervals(rows):
    """Dry/Wet from each interval start, zero-padded to a common CHUNK multiple."""
    spans = [span(r) for r in rows]
    width = -(-max(stop-start for start, _, stop in spans) // CHUNK) * CHUNK
    dry = np.zeros((len(rows), width), np.float32)
    wet = np.zeros_like(dry)
    for i, (row, (start, _, stop)) in enumerate(zip(rows, spans)):
        with sf.SoundFile(DATASET/row["path"]) as audio:
            audio.seek(start)
            pair = audio.read(stop-start, dtype="float32", always_2d=True)
        if pair.shape != (stop-start, 2):
            raise ValueError(f"Short read: {row['id']}")
        dry[i, :stop-start], wet[i, :stop-start] = pair[:, 0], pair[:, 1]
    return spans, dry, wet


@torch.no_grad()
def stream(model, dry, controls, gain, pots=None):
    state, previous, output = None, None, []
    for offset in range(0, dry.shape[1], CHUNK):
        x = dry[:, offset:offset+CHUNK]
        if pots is None:
            y, state = model(x, controls, state, gain)
        else:
            if previous is None:  # one Dry-only chunk seeds the physical state
                latent, state = model.trajectory(x, controls, None)
                previous = latent[:, -1]
                y = torch.zeros_like(x)
            else:
                y, _, state, previous = model(x, controls, pots, state, previous)
        output.append(y.float().cpu())
    return torch.cat(output, 1).numpy()


@torch.inference_mode()
def riccardovib(author, dry, spans, rows, originals, calibration):
    output = np.zeros(dry.shape, np.float32)
    for i, ((start, first, stop), row) in enumerate(zip(spans, rows)):
        x = torch.from_numpy(dry[i, first-start-16:stop-start]).cuda().unfold(0, 32, 16)
        label = originals[row["id"]]["labels"]
        c = np.array([label["attack_index"]/4, label["release_index"]/4,
                      (label["ratio"]-2)/8, abs(label["threshold_db"])/40], dtype=np.float32)
        if calibration["threshold_flipped"]:
            c[3] = 1-c[3]
        c = torch.tensor(c[calibration["permutation"]], device="cuda").expand(len(x), 4)
        output[i, first-start:stop-start] = torch.cat(
            [author(x[o:o+65536], c[o:o+65536]).reshape(-1) for o in range(0, len(x), 65536)]).cpu().numpy()
    return output


def neural_models():
    from pinn.evaluate_common_test import PINN, SEQUENCES, BATCHED
    for key in (*SEQUENCES, *BATCHED):
        if key in SEQUENCES:
            path = SEQUENCES[key]
            source = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
        else:
            best = json.loads((BATCHED[key]/"best.json").read_text())
            path = BATCHED[key]/best["checkpoint"]/"model.mdlus"
            source = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), selected_epoch=best["epoch"])
        yield key, accelerate_training(AudioModel.from_checkpoint(path).cuda().eval()), source
    saved = torch.load(PINN, map_location="cuda", weights_only=False)
    model = S6PINN(**saved["config"]["model"]).cuda().eval()
    model.load_state_dict(saved["model"])
    yield "s6_pinn", accelerate_training(model), dict(path=str(PINN.relative_to(ROOT)), sha256=sha(PINN))


def summarize(records):
    rows = list(records.values())
    audible = [r["audible_esr"] for r in rows if r["audible_esr"] is not None]
    return dict(macro_mean={m: float(np.mean([r[m] for r in rows])) for m in METRICS},
                median_esr=float(np.median([r["esr"] for r in rows])),
                pooled_esr=sum(r["squared_error"] for r in rows)/sum(r["target_energy"] for r in rows),
                audible=dict(files=len(audible), macro_esr=float(np.mean(audible)),
                             median_esr=float(np.median(audible)),
                             pooled_esr=sum(r["audible_squared_error"] for r in rows) /
                             sum(r["audible_target_energy"] for r in rows),
                             seconds=sum(r["audible_seconds"] for r in rows)))


def run(output, limit=None, batch=32, workers=12):
    # Circuit and comparison imports need numba; the checkpoint-only scorer never loads them.
    from pinn.evaluate_common_test import RICCARDO, CIRCUITS, KEYS
    if (output/"report.json").exists():
        raise FileExistsError("Report exists; scores are fixed once written")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()

    def status(**more):
        value = dict(utc=utc(), seconds=round(time.monotonic()-started, 1), **more)
        write_json(output/"status.json", value)
        print(json.dumps(value), flush=True)

    manifest = json.loads(MANIFEST.read_text())
    rows = sorted(manifest["records"], key=lambda r: r["id"])[:limit]
    originals = {r["id"]: r for r in manifest["records"]}
    records = {key: {} for key in KEYS}
    sources = {key: dict(path=path, sha256=sha(ROOT/path)) for key, path in CIRCUITS.items()}
    configs = {key: asdict(read_config(ROOT/path)[0]) for key, path in CIRCUITS.items()}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(circuit_record, (key, configs[key], row)) for key in CIRCUITS for row in rows]
        status(phase="gpu models", circuits_queued=len(jobs))
        for offset in range(0, len(rows), batch):
            selected = rows[offset:offset+batch]
            spans, dry, wet = read_intervals(selected)
            labels = [encode_labels(r["labels"]) for r in selected]
            controls = torch.tensor([c for c, _ in labels], dtype=torch.float32, device="cuda")
            gain = torch.tensor([g for _, g in labels], dtype=torch.float32, device="cuda")
            x = torch.from_numpy(dry).cuda()
            predictions = {}
            torch.backends.cudnn.allow_tf32 = True  # original S4/S6 evaluator mode
            for key, model, source in neural_models():
                pots = pots_for(selected, originals, "cuda") if key == "s6_pinn" else None
                predictions[key] = stream(model, x, controls, gain, pots)
                sources[key] = source
                del model
            torch.backends.cudnn.allow_tf32 = False
            calibration = json.loads((ROOT/"runs/pretrained-comparison/conditioning-calibration.json").read_text())["selected"]
            author = AuthorCL1B(RICCARDO).float().cuda().eval()
            predictions["riccardovib"] = riccardovib(author, dry, spans, selected, originals, calibration)
            sources["riccardovib"] = dict(path=str(RICCARDO.relative_to(ROOT)), sha256=sha(RICCARDO))
            del author, x
            torch.cuda.empty_cache()
            for key, values in predictions.items():
                for i, (row, (start, first, stop)) in enumerate(zip(selected, spans)):
                    a, b = first-start, stop-start
                    records[key][row["id"]] = interval_metrics(values[i, a:b], wet[i, a:b])
            status(phase="gpu models", files=offset+len(selected), total=len(rows))
        for done, job in enumerate(as_completed(jobs), 1):
            key, name, values = job.result()
            records[key][name] = values
            if done % 100 == 0:
                status(phase="circuits", completed=done, total=len(jobs))
    scored = sum(stop-first for _, first, stop in map(span, rows))
    report = dict(completed_utc=utc(), manifest_signature=manifest["signature"], split="test",
                  files=len(rows), samples_scored=scored, warmup_samples=WARMUP, chunk_samples=CHUNK,
                  audible_threshold_dbfs=AUDIBLE_DB, min_audible_seconds=MIN_AUDIBLE_SECONDS,
                  gpu=torch.cuda.get_device_name(0), metric_protocol=METRIC_PROTOCOL, sources=sources,
                  history=dict(circuits="continuous Dry-only simulation from file sample 0",
                               sequence="zero state at interval start; Dry-only warmup of at least 49152 samples; fused scan",
                               s6_pinn="as sequence; first 1024-sample chunk seeds the physical state",
                               riccardovib="native 16-sample Dry history per 16-sample output block"),
                  summary={key: summarize(records[key]) for key in KEYS}, records=records)
    write_json(output/"report.json", report)
    status(phase="complete", macro_esr={k: v["macro_mean"]["esr"] for k, v in report["summary"].items()},
           audible_macro_esr={k: v["audible"]["macro_esr"] for k, v in report["summary"].items()})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/full-test-20261103")
    parser.add_argument("--limit", type=int, help="Smoke test on the first N files")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(args.output, args.limit, args.batch, args.workers)
