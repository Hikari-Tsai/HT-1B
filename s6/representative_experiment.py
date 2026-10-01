"""Paired GPU S4/S6 study on deterministic windows from every training interval.

This experiment deliberately samples the full-corpus manifest; it does not
claim full-sample training coverage or an untouched independent test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch

from .data import encode_labels
from .layers import detach_state
from .losses import AudioLoss
from .models import AudioModel
from .training import load_config


WINDOW = 5120
WARMUP = 4096
SCORE = 1024
ARCHITECTURES = ("s4_tfilm", "s6_tfilm")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def windows(manifest: dict, dataset: Path, split: str, epoch: int, epochs: int) -> list[dict]:
    rows = []
    for record in manifest["records"]:
        path = dataset / record["path"]
        controls, gain = encode_labels(record["labels"])
        for index, interval in enumerate(record["intervals"]):
            if interval["split"] != split:
                continue
            lo, hi = interval["start_sample"], interval["stop_sample"]
            if hi - lo < WINDOW:
                raise ValueError(f"Interval shorter than {WINDOW}: {record['id']} {index}")
            # Stratified positions cover different times on successive epochs.
            fraction = (epoch + .5) / epochs if split == "train" else .5
            start = lo + int((hi - lo - WINDOW) * fraction)
            rows.append(dict(id=record["id"], path=str(path), interval=index,
                             split=split, start=start, stop=start + WINDOW,
                             controls=controls, gain=gain))
    return rows


def load_batch(rows: list[dict], device: torch.device):
    dry = np.empty((len(rows), WINDOW), dtype=np.float32)
    wet = np.empty_like(dry)
    for i, row in enumerate(rows):
        with sf.SoundFile(row["path"]) as audio:
            if audio.samplerate != 48000 or audio.channels != 2:
                raise ValueError(f"Expected 48 kHz stereo WAV: {row['path']}")
            audio.seek(row["start"])
            pair = audio.read(WINDOW, dtype="float32", always_2d=True)
        if pair.shape != (WINDOW, 2) or not np.isfinite(pair).all():
            raise ValueError(f"Invalid audio window: {row['id']} {row['start']}")
        dry[i], wet[i] = pair[:, 0], pair[:, 1]
    return (torch.from_numpy(dry).to(device), torch.from_numpy(wet).to(device),
            torch.tensor([r["controls"] for r in rows], dtype=torch.float32, device=device),
            torch.tensor([r["gain"] for r in rows], dtype=torch.float32, device=device))


def predict(model: AudioModel, dry, controls, gain):
    state = None
    with torch.no_grad():
        for start in range(0, WARMUP, SCORE):
            _, state = model(dry[:, start:start + SCORE], controls, state, gain)
            state = detach_state(state)
    return model(dry[:, WARMUP:], controls, state, gain)[0]


@torch.no_grad()
def evaluate(model, rows, device, batch_size, progress=None):
    model.eval()
    per_file = defaultdict(lambda: [0., 0., 0., 0])
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        dry, wet, controls, gain = load_batch(batch, device)
        predicted = predict(model, dry, controls, gain)
        target = wet[:, WARMUP:]
        if not torch.isfinite(predicted).all():
            raise ValueError("Nonfinite validation output")
        delta = predicted - target
        square = delta.square().sum(dim=1).double().cpu().tolist()
        absolute = delta.abs().sum(dim=1).double().cpu().tolist()
        energy = target.square().sum(dim=1).double().cpu().tolist()
        for row, sq, ab, en in zip(batch, square, absolute, energy):
            sums = per_file[row["id"]]
            sums[0] += sq
            sums[1] += ab
            sums[2] += en
            sums[3] += SCORE
        if progress:
            progress(min(offset + batch_size, len(rows)), len(rows))
    records = []
    for name, (sq, ab, en, count) in sorted(per_file.items()):
        records.append(dict(id=name, windows=int(count / SCORE), scored_samples=int(count),
                            esr=sq / max(en, count * 1e-8), mse=sq / count, mae=ab / count))
    totals = [sum(v[i] for v in per_file.values()) for i in range(4)]
    macro = {key: float(np.mean([r[key] for r in records])) for key in ("esr", "mse", "mae")}
    pooled = dict(esr=totals[0] / max(totals[2], totals[3] * 1e-8),
                  mse=totals[0] / totals[3], mae=totals[1] / totals[3])
    return dict(windows=len(rows), files=len(records), scored_samples=int(totals[3]),
                macro_mean=macro, pooled=pooled, records=records)


def run(manifest_path: Path, dataset: Path, output: Path, *, epochs: int, batch_size: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("sample_rate") != 48000:
        raise ValueError("Expected 48 kHz full-corpus manifest schema 2")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    manifest_signature = manifest["signature"]
    train_probe = windows(manifest, dataset, "train", 0, epochs)
    validation = windows(manifest, dataset, "validation", 0, epochs)
    if len(train_probe) != 8680 or len(validation) != 1240:
        raise ValueError("Unexpected interval counts; review the sampling protocol")
    if len({r["id"] for r in train_probe}) != 620 or len({r["id"] for r in validation}) != 620:
        raise ValueError("Expected all 620 files in both train and validation")
    for row in train_probe + validation:
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    output.mkdir(parents=True)
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    config_root = Path(__file__).resolve().parent / "configs"
    configs = {arch: load_config(config_root / ("neural-s4-tfilm.json" if arch == "s4_tfilm" else "neural-s6-tfilm.json"))
               for arch in ARCHITECTURES}
    protocol = dict(manifest_signature=manifest_signature, epochs=epochs, batch_size=batch_size,
                    warmup_samples=WARMUP, scored_samples_per_window=SCORE, window_samples=WINDOW,
                    train_intervals=len(train_probe), validation_intervals=len(validation),
                    train_files=620, validation_files=620,
                    train_frames_in_full_manifest=manifest["frames_by_split"]["train"],
                    train_input_samples_per_epoch=len(train_probe) * WINDOW,
                    train_scored_samples_per_epoch=len(train_probe) * SCORE,
                    validation_scored_samples=len(validation) * SCORE,
                    sampling="one stratified window per training interval per epoch; fixed midpoint window per validation interval",
                    gpu=gpu, cuda=torch.version.cuda, split_scope="validation_model_selection_only")
    write_json(output / "protocol.json", protocol)
    status_path = output / "status.json"
    begun = time.monotonic()
    state = dict(phase="starting", utc=utc(), gpu=gpu, cuda=torch.version.cuda,
                 architecture=None, epoch=0, epochs=epochs, train_intervals_seen=0,
                 train_intervals_total=8680, validation_intervals_seen=0,
                 validation_intervals_total=1240, batch_loss=None, eta_seconds=None)

    def update(**changes):
        state.update(changes, utc=utc(), elapsed_seconds=round(time.monotonic() - begun, 1))
        write_json(status_path, state)

    update()
    try:
        for arch in ARCHITECTURES:
            config = configs[arch]
            model = AudioModel(**config["model"]).to(device)
            options = config["training"]
            loss_fn = AudioLoss(options["fft_sizes"], options["l1_weight"],
                                options["esr_weight"], options["spectral_weight"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                           weight_decay=options["weight_decay"])
            history = []
            best_esr = math.inf
            for epoch in range(epochs):
                rows = windows(manifest, dataset, "train", epoch, epochs)
                random.Random(7 + epoch).shuffle(rows)
                model.train()
                loss_sum = 0.
                seen = 0
                epoch_start = time.monotonic()
                update(phase="training", architecture=arch, epoch=epoch + 1,
                       train_intervals_seen=0, validation_intervals_seen=0)
                for offset in range(0, len(rows), batch_size):
                    batch = rows[offset:offset + batch_size]
                    dry, wet, controls, gain = load_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    predicted = predict(model, dry, controls, gain)
                    loss, _ = loss_fn(predicted, wet[:, WARMUP:])
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite {arch} loss at epoch {epoch + 1}, offset {offset}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"])
                    optimizer.step()
                    batch_loss = loss.item()
                    seen += len(batch)
                    loss_sum += batch_loss * len(batch)
                    remaining = (len(rows) - seen) / max(seen, 1) * (time.monotonic() - epoch_start)
                    update(train_intervals_seen=seen, batch_loss=batch_loss,
                           eta_seconds=round(remaining, 1))
                update(phase="validation", validation_intervals_seen=0, eta_seconds=None)
                result = evaluate(model, validation, device, batch_size,
                                  lambda count, total: update(validation_intervals_seen=count))
                current = result["macro_mean"]["esr"]
                if current < best_esr:
                    best_esr = current
                    model.save(output / f"{arch}-best.mdlus")
                    write_json(output / f"{arch}-best-validation.json",
                               dict(architecture=arch, selected_epoch=epoch + 1, **result))
                history.append(dict(epoch=epoch + 1, train_mean_batch_loss=loss_sum / len(rows),
                                    validation_macro_mean=result["macro_mean"],
                                    validation_pooled=result["pooled"], best_esr=best_esr,
                                    epoch_seconds=round(time.monotonic() - epoch_start, 1)))
                write_json(output / f"{arch}-history.json", dict(architecture=arch, history=history))
                update(phase="epoch_complete", best_validation_esr=best_esr,
                       epoch_seconds=history[-1]["epoch_seconds"])
                print(json.dumps(dict(architecture=arch, **history[-1]), ensure_ascii=False), flush=True)
            del model, optimizer
            torch.cuda.empty_cache()
        reports = {arch: json.loads((output / f"{arch}-best-validation.json").read_text())
                   for arch in ARCHITECTURES}
        comparison = dict(protocol=protocol, validation={arch: dict(
            selected_epoch=reports[arch]["selected_epoch"],
            macro_mean=reports[arch]["macro_mean"], pooled=reports[arch]["pooled"],
            files=reports[arch]["files"], windows=reports[arch]["windows"],
            scored_samples=reports[arch]["scored_samples"]) for arch in ARCHITECTURES},
            validation_winner=min(ARCHITECTURES, key=lambda a: reports[a]["macro_mean"]["esr"]),
            scope="sampled_windows_same_data; validation_only; no new independent_test")
        write_json(output / "comparison.json", comparison)
        update(phase="building_page", architecture=None, eta_seconds=None)
        from pinn.build_comparison_page import build
        build(Path(__file__).resolve().parents[1] / "output/cl1b-comparison.html")
        update(phase="complete")
        print(json.dumps(dict(phase="complete", validation_winner=comparison["validation_winner"])), flush=True)
    except Exception as exc:
        update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run(args.manifest, args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size)
