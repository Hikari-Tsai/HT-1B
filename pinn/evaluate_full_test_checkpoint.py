"""Score extra S4/S6 checkpoints on the complete test intervals.

Same samples, history and metrics as evaluate_full_test.py (zero state at each
interval start, fused scan, scoring from the first 16-aligned sample >= 49152
after the start), for checkpoints trained after that report was fixed. Each
model is given as name=checkpoint-directory; its model.mdlus is scored.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import torch

from pinn.evaluate_full_test import (ROOT, sha, MANIFEST, WARMUP, CHUNK, AUDIBLE_DB, MIN_AUDIBLE_SECONDS, span,
                                     interval_metrics, read_intervals, stream, summarize)
from ht1b.audio_metrics import METRIC_PROTOCOL
from s6.data import encode_labels
from s6.fused_scan import accelerate_training
from s6.models import AudioModel
from s6.representative_experiment import write_json, utc


def run(output, models, limit=None, batch=32):
    if (output/"report.json").exists():
        raise FileExistsError("Report exists; scores are fixed once written")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # as evaluate_full_test for sequence models
    started = time.monotonic()

    def status(**more):
        value = dict(utc=utc(), seconds=round(time.monotonic()-started, 1), **more)
        write_json(output/"status.json", value)
        print(json.dumps(value), flush=True)

    manifest = json.loads(MANIFEST.read_text())
    rows = sorted(manifest["records"], key=lambda r: r["id"])[:limit]
    loaded, sources = {}, {}
    for key, directory in models.items():
        path = directory/"model.mdlus"
        loaded[key] = accelerate_training(AudioModel.from_checkpoint(path).cuda().eval())
        sources[key] = dict(path=str(path.resolve().relative_to(ROOT)), sha256=sha(path),
                            output_gain=loaded[key].output_gain)
    records = {key: {} for key in models}
    for offset in range(0, len(rows), batch):
        selected = rows[offset:offset+batch]
        spans, dry, wet = read_intervals(selected)
        labels = [encode_labels(r["labels"]) for r in selected]
        controls = torch.tensor([c for c, _ in labels], dtype=torch.float32, device="cuda")
        gain = torch.tensor([g for _, g in labels], dtype=torch.float32, device="cuda")
        x = torch.from_numpy(dry).cuda()
        for key, model in loaded.items():
            values = stream(model, x, controls, gain)
            for i, (row, (start, first, stop)) in enumerate(zip(selected, spans)):
                a, b = first-start, stop-start
                records[key][row["id"]] = interval_metrics(values[i, a:b], wet[i, a:b])
        status(files=offset+len(selected), total=len(rows))
    report = dict(completed_utc=utc(), manifest_signature=manifest["signature"], split="test",
                  files=len(rows), samples_scored=sum(stop-first for _, first, stop in map(span, rows)),
                  warmup_samples=WARMUP, chunk_samples=CHUNK, audible_threshold_dbfs=AUDIBLE_DB,
                  min_audible_seconds=MIN_AUDIBLE_SECONDS, gpu=torch.cuda.get_device_name(0),
                  metric_protocol=METRIC_PROTOCOL, sources=sources,
                  history="zero state at interval start; Dry-only warmup of at least 49152 samples; fused scan",
                  summary={key: summarize(records[key]) for key in models}, records=records)
    write_json(output/"report.json", report)
    status(phase="complete", macro_esr={k: v["macro_mean"]["esr"] for k, v in report["summary"].items()})
    return report


def model_argument(text):
    name, _, path = text.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("expected name=checkpoint-directory")
    return name, Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=model_argument, action="append", required=True)
    parser.add_argument("--limit", type=int, help="Smoke test on the first N files")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    run(args.output, dict(args.model), args.limit, args.batch)
