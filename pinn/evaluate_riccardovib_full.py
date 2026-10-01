"""Score fixed RiccardoVib CL1B weights on the full-corpus held-out spans.

The author model consumes 16 Dry history samples and predicts the next 16. Every
scored span is aligned to that global block grid, so only Dry data immediately
preceding each span is needed; held-out Wet is read solely for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from ht1b.audio import metrics
from ht1b.audio_metrics import METRIC_PROTOCOL, frame_analysis, mrstft, region_metrics
from ht1b.corpus import load_corpus
from compare_pretrained import AuthorCL1B

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data/external/optical-compressor/Fully_Conditioned_Black_Box_Model_for_Compression/Models/CL1BModel/Checkpoints/best/weights.h5"
CALIBRATION = ROOT / "runs/pretrained-comparison/conditioning-calibration.json"
METRICS = ("mse", "mae", "esr", "mrstft", "rms_db_mae")


def controls(row: dict, selected: dict) -> np.ndarray:
    labels = row["labels"]
    values = np.array((labels["attack_index"] / 4, labels["release_index"] / 4,
                       (labels["ratio"] - 2) / 8, abs(labels["threshold_db"]) / 40), dtype=np.float32)
    if selected["threshold_flipped"]:
        values[3] = 1 - values[3]
    return values[selected["permutation"]]


def score_span(source: sf.SoundFile, span: dict, condition: torch.Tensor,
               model: AuthorCL1B, chunk_samples: int) -> dict:
    start, stop = span["start_sample"], span["stop_sample"]
    if start < 16 or start % 16 or stop % 16 or (stop - start) % 16:
        raise ValueError(f"Span does not align with the author's 16-sample output grid: {span}")
    predictions, targets = [], []
    for position in range(start, stop, chunk_samples):
        end = min(position + chunk_samples, stop)
        source.seek(position - 16)
        block = source.read(end - position + 16, dtype="float32", always_2d=True)
        if block.shape != (end - position + 16, 2):
            raise ValueError("Unexpected stereo audio length")
        dry = block[:, 0]
        windows = np.lib.stride_tricks.sliding_window_view(dry, 32)[::16]
        if len(windows) * 16 != end - position:
            raise ValueError("Prediction block coverage mismatch")
        with torch.inference_mode():
            x = torch.as_tensor(np.ascontiguousarray(windows), dtype=torch.float32, device="cuda")
            c = condition.expand(len(windows), -1)
            predicted = model(x, c).reshape(-1).cpu().numpy().astype("float64")
        predictions.append(predicted)
        targets.append(block[16:, 1].astype("float64"))
    p, t = np.concatenate(predictions), np.concatenate(targets)
    if len(p) != stop - start or len(t) != stop - start:
        raise ValueError("Scored sample count mismatch")
    values = metrics(p, t)
    values["mrstft"] = mrstft(p, t)["total"]
    values["rms_db_mae"] = region_metrics(frame_analysis({"reference_wet": t, "prediction": p}), "prediction")["rms_db_mae"]
    return dict(**span, samples=len(p), metrics=values)


def score_record(dataset: Path, row: dict, split: str, model: AuthorCL1B,
                 selected: dict, chunk_samples: int) -> dict:
    spans = [span for span in row["intervals"] if span["split"] == split]
    condition = torch.as_tensor(controls(row, selected), dtype=torch.float32, device="cuda")[None, :]
    with sf.SoundFile(dataset / row["path"]) as source:
        if source.samplerate != 48000 or source.channels != 2:
            raise ValueError(f"Expected stereo 48 kHz: {row['id']}")
        results = [score_span(source, span, condition, model, chunk_samples) for span in spans]
    total = sum(span["samples"] for span in results)
    mse = sum(span["samples"] * span["metrics"]["mse"] for span in results) / total
    energy = sum(span["samples"] * span["metrics"]["target_rms"] ** 2 for span in results) / total
    summary = {key: sum(span["samples"] * span["metrics"][key] for span in results) / total
               for key in ("mae", "mrstft", "rms_db_mae")}
    summary.update(mse=mse, esr=mse / max(energy, 1e-12))
    return dict(id=row["id"], split=split, samples=total, metrics=summary, intervals=results)


def evaluate(manifest_path: Path, dataset_path: Path, output: Path,
             split: str, chunk_samples: int, smoke_records: int = 0) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RiccardoVib inference")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if chunk_samples % 16:
        raise ValueError("chunk_samples must be divisible by 16")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpus, dataset = load_corpus(manifest_path, dataset_path)
    selected = json.loads(CALIBRATION.read_text(encoding="utf-8"))["selected"]
    original = json.loads((ROOT / "runs/pretrained-comparison/comparison.json").read_text(encoding="utf-8"))
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != original["weights_sha256"]:
        raise ValueError("RiccardoVib weights differ from the verified five-clip benchmark")
    model = AuthorCL1B(WEIGHTS).float().cuda().eval()
    reference = np.load(ROOT / "runs/pretrained-comparison/tensorflow-reference.npz")
    with torch.inference_mode():
        checked = model(torch.as_tensor(reference["windows"], dtype=torch.float32, device="cuda"),
                        torch.as_tensor(reference["conditioning"], dtype=torch.float32, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(checked, reference["predictions"], rtol=1e-4, atol=1e-5)
    rows = [row for row in corpus["records"] if any(span["split"] == split for span in row["intervals"])]
    if smoke_records:
        rows = rows[:smoke_records]
    started = time.perf_counter()
    records = []
    status_path = output.with_suffix(".status.json")
    for row in rows:
        records.append(score_record(dataset, row, split, model, selected, chunk_samples))
        if len(records) % 10 == 0 or len(records) == len(rows):
            progress = {"phase": "evaluating", "split": split, "completed": len(records), "total": len(rows),
                        "seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name()}
            status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps(progress), flush=True)
    records.sort(key=lambda row: row["id"])
    expected = sum(row["samples"] for row in records) if smoke_records else corpus["frames_by_split"][split]
    if sum(row["samples"] for row in records) != expected:
        raise ValueError("Incomplete evaluation coverage")
    report = {"manifest_signature": corpus["signature"], "split": split, "model": "RiccardoVib CL1B ED",
              "weights": str(WEIGHTS), "conditioning": selected, "records": records,
              "samples_scored": expected, "seconds": time.perf_counter() - started,
              "macro_mean": {key: float(np.mean([row["metrics"][key] for row in records])) for key in METRICS},
              "metric_protocol": METRIC_PROTOCOL,
              "history": "Author model 16-sample Dry lookback; fixed pretrained weights, no output gain or delay correction.",
              "test_scope": "Previously opened test set; exploratory reuse, not a fresh independent test." if split == "test" else "Validation comparison; source training overlap unknown.",
              "smoke_records": smoke_records}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    status_path.write_text(json.dumps({"phase": "complete", "split": split, "completed": len(rows),
                                       "total": len(rows), "seconds": report["seconds"]}, indent=2), encoding="utf-8")
    print(json.dumps({"complete": split, "macro_mean": report["macro_mean"],
                      "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--chunk-samples", type=int, default=240000)
    parser.add_argument("--smoke-records", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.manifest, args.dataset, args.output, args.split, args.chunk_samples, args.smoke_records)
