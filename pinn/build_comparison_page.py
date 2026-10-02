"""Build a portable, offline comparison page from actual benchmark artifacts.

Run with the repository environment: python -m pinn.build_comparison_page
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from ht1b.audio_metrics import METRIC_PROTOCOL, frame_analysis, mrstft, region_metrics

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).with_name("comparison-ui")
FULL_RUN = ROOT / "runs/full-corpus-gpu-20260925"
GRU_RUN = ROOT / "runs/gru-pinn-gpu-20260925"
RICCARDO_RUN = ROOT / "runs/riccardovib-full-corpus-20260925"
S4S6_RUN = ROOT / "runs/s4-s6-representative-gpu-20260925"
FIVE_RUN = ROOT / "runs/six-model-common-validation-20260925"
S6_PINN_RUN = ROOT / "runs/s6-tfilm-pinn-gpu-20260929"
ARCH_METRICS = ("esr", "mse", "mae", "mrstft", "rms_db_mae")


def envelope(samples: np.ndarray, bins: int = 480) -> list[list[float]]:
    return [[round(float(x.min()), 7), round(float(x.max()), 7)]
            for x in np.array_split(samples, bins)]


def full_corpus_results() -> dict | None:
    """Add the locked, held-out test only when the experiment is complete."""
    status_path = FULL_RUN / "experiment-status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["phase"] != "complete":
        return None
    manifest = json.loads((FULL_RUN / "manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads((FULL_RUN / "training/coverage.json").read_text(encoding="utf-8"))
    selection = json.loads((FULL_RUN / "selection.json").read_text(encoding="utf-8"))
    baseline = json.loads((FULL_RUN / "baseline-validation.json").read_text(encoding="utf-8"))
    fitted = json.loads((FULL_RUN / "fitted-validation.json").read_text(encoding="utf-8"))
    test = json.loads((FULL_RUN / "selected-test.json").read_text(encoding="utf-8"))
    if coverage["frames_covered"] != coverage["frames_per_epoch"] or coverage["frames_covered"] != manifest["frames_by_split"]["train"]:
        raise ValueError("Full corpus training coverage mismatch")
    if not selection["test_not_used"] or selection["selected"] not in ("baseline", "fitted"):
        raise ValueError("Model selection was not locked before test")
    if test["split"] != "test" or test["samples_scored"] != manifest["frames_by_split"]["test"]:
        raise ValueError("Full corpus test coverage mismatch")
    if any(report["manifest_signature"] != manifest["signature"] for report in (baseline, fitted, test)):
        raise ValueError("Full corpus manifest signature mismatch")
    if len(test["records"]) != len(manifest["records"]):
        raise ValueError("Full corpus test record count mismatch")
    for key, val in test["macro_mean"].items():
        if not np.isclose(val, np.mean([r["metrics"][key] for r in test["records"]]), rtol=1e-10):
            raise ValueError(f"Test macro mean mismatch: {key}")
    labels = {r["id"]: r["labels"] for r in manifest["records"]}
    records = [dict(id=r["id"], labels=labels[r["id"]], samples=r["samples"],
                    metrics=r["metrics"], intervals=[dict(start_sample=s["start_sample"],
                    stop_sample=s["stop_sample"], metrics=s["metrics"]) for s in r["intervals"]])
               for r in test["records"]]
    return dict(run=FULL_RUN.name, completed_utc=status["utc"], selected=selection["selected"],
                criterion=selection["criterion"], train_samples=coverage["frames_covered"],
                train_records=len(coverage["per_record"]), samples_by_split=manifest["frames_by_split"],
                independence=manifest["independence"], validation={"baseline":baseline["macro_mean"],
                "fitted":fitted["macro_mean"]}, test=test["macro_mean"], records=records,
                metric_protocol=test["metric_protocol"])


def architecture_results(full: dict | None) -> dict | None:
    """Keep architecture selection on validation; mark test reuse as exploratory."""
    if full is None:
        return None
    status_path = GRU_RUN / "experiment-status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("phase") != "complete":
        return None
    riccardo_paths = {split: RICCARDO_RUN / f"{split}.json" for split in ("validation", "test")}
    if not all(path.exists() for path in riccardo_paths.values()):
        return None
    manifest = json.loads((GRU_RUN / "manifest.json").read_text(encoding="utf-8"))
    if manifest["signature"] != json.loads((FULL_RUN / "manifest.json").read_text(encoding="utf-8"))["signature"]:
        raise ValueError("GRU and MLP manifest signatures differ")
    coverage = json.loads((GRU_RUN / "training/coverage.json").read_text(encoding="utf-8"))
    training_status = json.loads((GRU_RUN / "training/status.json").read_text(encoding="utf-8"))
    if (coverage["frames_covered"] != manifest["frames_by_split"]["train"]
            or coverage["holdout_wet_training_samples"] != 0
            or len(coverage["per_record"]) != len(manifest["records"])
            or training_status["phase"] != "training_complete"
            or not training_status["cuda"]):
        raise ValueError("GRU training coverage or held-out Wet isolation mismatch")
    validation = {key: json.loads((path / name).read_text(encoding="utf-8")) for key, path, name in (
        ("baseline", FULL_RUN, "baseline-validation.json"),
        ("mlp", FULL_RUN, "fitted-validation.json"),
        ("gru", GRU_RUN, "gru-validation.json"))}
    validation["riccardovib"] = json.loads(riccardo_paths["validation"].read_text(encoding="utf-8"))
    test = {key: json.loads((path / name).read_text(encoding="utf-8")) for key, path, name in (
        ("mlp", FULL_RUN, "selected-test.json"),
        ("gru", GRU_RUN, "gru-test.json"))}
    test["riccardovib"] = json.loads(riccardo_paths["test"].read_text(encoding="utf-8"))
    author_calibration = json.loads((ROOT / "runs/pretrained-comparison/conditioning-calibration.json").read_text(encoding="utf-8"))["selected"]
    for report in (validation["riccardovib"], test["riccardovib"]):
        if (report.get("model") != "RiccardoVib CL1B ED" or report.get("smoke_records") != 0
                or report.get("conditioning") != author_calibration
                or report.get("metric_protocol") != METRIC_PROTOCOL):
            raise ValueError("RiccardoVib full-corpus report is missing or incomplete")
    labels = {r["id"]: r["labels"] for r in manifest["records"]}
    for split, reports in (("validation", validation), ("test", test)):
        expected = manifest["frames_by_split"][split]
        reference = next(iter(reports.values()))
        reference_records = {r["id"]: r for r in reference["records"]}
        if set(reference_records) != set(labels):
            raise ValueError(f"{split} records do not match manifest")
        for name, report in reports.items():
            if (report["manifest_signature"] != manifest["signature"] or report["split"] != split
                    or report["samples_scored"] != expected or len(report["records"]) != len(labels)):
                raise ValueError(f"{name} {split} report mismatch")
            for row in report["records"]:
                original = reference_records[row["id"]]
                if (row["samples"] != original["samples"] or
                        [(x["start_sample"], x["stop_sample"]) for x in row["intervals"]] !=
                        [(x["start_sample"], x["stop_sample"]) for x in original["intervals"]]):
                    raise ValueError(f"{name} {split} interval mismatch: {row['id']}")
            for metric in ARCH_METRICS:
                actual = np.mean([row["metrics"][metric] for row in report["records"]])
                if not np.isclose(actual, report["macro_mean"][metric], rtol=1e-10):
                    raise ValueError(f"{name} {split} macro mean mismatch: {metric}")
    comparison = json.loads((GRU_RUN / "comparison.json").read_text(encoding="utf-8"))
    if comparison["validation_winner"] != "mlp":
        raise ValueError("Unexpected architecture selection")
    for name in ("baseline", "mlp", "gru"):
        report = validation[name]
        for metric in ARCH_METRICS:
            if not np.isclose(comparison[name][metric], report["macro_mean"][metric], rtol=1e-10):
                raise ValueError(f"Architecture comparison mismatch: {name}/{metric}")
    by_record = {split: {name: {r["id"]: r for r in report["records"]}
                         for name, report in reports.items()}
                 for split, reports in (("validation", validation), ("test", test))}
    return {
        "run": GRU_RUN.name,
        "manifest_signature": manifest["signature"],
        "training": {"gpu": training_status["gpu"], "samples": coverage["frames_covered"],
                     "records": len(coverage["per_record"]), "holdout_wet_training_samples": 0},
        "validation": {name: report["macro_mean"] for name, report in validation.items()},
        "test": {name: report["macro_mean"] for name, report in test.items()},
        "test_samples": test["gru"]["samples_scored"],
        "records": [{"id": record["id"], "ratio": labels[record["id"]]["ratio"],
                     "validation": {name: by_record["validation"][name][record["id"]]["metrics"]
                                    for name in validation},
                     "test": {name: by_record["test"][name][record["id"]]["metrics"]
                              for name in test}}
                    for record in manifest["records"]],
        "validation_winner": "mlp",
        "test_scope": "exploratory_reuse"
    }


def sequence_results() -> dict | None:
    status_path = S4S6_RUN / "status.json"
    if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8")).get("phase") not in ("building_page", "complete"):
        return None
    protocol = json.loads((S4S6_RUN / "protocol.json").read_text(encoding="utf-8"))
    comparison = json.loads((S4S6_RUN / "comparison.json").read_text(encoding="utf-8"))
    original = json.loads((FULL_RUN / "manifest.json").read_text(encoding="utf-8"))
    if (protocol["manifest_signature"] != original["signature"]
            or comparison["protocol"] != protocol
            or protocol["train_intervals"] != 8680
            or protocol["validation_intervals"] != 1240
            or protocol["train_files"] != 620
            or protocol["validation_files"] != 620):
        raise ValueError("S4/S6 protocol does not match the full-corpus manifest")
    models = {}
    for key in ("s4_tfilm", "s6_tfilm"):
        report = json.loads((S4S6_RUN / f"{key}-best-validation.json").read_text(encoding="utf-8"))
        history = json.loads((S4S6_RUN / f"{key}-history.json").read_text(encoding="utf-8"))["history"]
        if (report["architecture"] != key or len(history) != protocol["epochs"]
                or report["files"] != 620 or report["windows"] != 1240
                or report["scored_samples"] != protocol["validation_scored_samples"]
                or len(report["records"]) != 620):
            raise ValueError(f"Incomplete {key} validation report")
        for metric in ("esr", "mse", "mae"):
            if not np.isclose(np.mean([r[metric] for r in report["records"]]), report["macro_mean"][metric], rtol=1e-10):
                raise ValueError(f"Invalid {key} macro mean")
        models[key] = dict(selected_epoch=report["selected_epoch"],
                           validation=report["macro_mean"], pooled=report["pooled"],
                           history=history, records=report["records"])
    if set(r["id"] for r in models["s4_tfilm"]["records"]) != set(r["id"] for r in models["s6_tfilm"]["records"]):
        raise ValueError("S4/S6 validation files differ")
    return dict(run=S4S6_RUN.name, protocol=protocol, models=models,
                validation_winner=comparison["validation_winner"], scope=comparison["scope"])


def five_model_results() -> dict | None:
    source_run = (S6_PINN_RUN / "common-validation"
                  if (S6_PINN_RUN / "common-validation/comparison.json").exists() else FIVE_RUN)
    path = source_run / "comparison.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((FULL_RUN / "manifest.json").read_text(encoding="utf-8"))
    keys = {"baseline", "mlp", "gru", "s4", "s6", "riccardovib"}
    if source_run != FIVE_RUN:
        keys.add("s6_pinn")
    if (report["manifest_signature"] != manifest["signature"]
            or report["split"] != "validation" or report["files"] != 620
            or report["windows"] != 1240 or report["samples_scored"] != 1269760
            or set(report["macro_mean"]) != keys
            or {r["id"] for r in report["records"]} != {r["id"] for r in manifest["records"]}
            or len(report["records"]) != 620 or len(report["intervals"]) != 1240):
        raise ValueError("Five-model comparison is incomplete or uses a different split")
    window_doc = json.loads((source_run / "windows.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(window_doc["windows"], sort_keys=True).encode()).hexdigest()
    if digest != report["window_signature"] or digest != window_doc["window_signature"]:
        raise ValueError("Common window signature mismatch")
    for model in keys:
        source = report["sources"][model]
        if hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Fixed {model} artifact changed since evaluation")
        for metric in ARCH_METRICS:
            actual = np.mean([r["metrics"][model][metric] for r in report["records"]])
            if not np.isclose(actual, report["macro_mean"][model][metric], rtol=1e-10):
                raise ValueError(f"Invalid five-model mean: {model}/{metric}")
        count = sum(r["samples"] for r in report["records"])
        squared = sum(r["samples"] * r["metrics"][model]["mse"] for r in report["records"])
        energy = sum(r["target_energy"] for r in report["records"])
        if not np.isclose(squared/max(energy, count*1e-8), report["pooled"][model]["esr"], rtol=1e-10):
            raise ValueError(f"Invalid five-model pooled ESR: {model}")
    return report


def build_legacy(output: Path) -> None:
    source = ROOT / "runs/pretrained-comparison"
    report = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    progression = json.loads((ROOT / "runs/gpu-continuation-2000-to-5000/progression.json")
                             .read_text(encoding="utf-8"))
    report.pop("weights", None)  # The portable report does not need local machine paths.
    report["metric_protocol"] = METRIC_PROTOCOL
    benchmark_hash = hashlib.sha256((source / "comparison.json").read_bytes())
    report["training"] = [{"step": s["step"], "fixed_loss": s["fixed_loss"]}
                          for s in progression["stages"]]
    labels = ["reference_dry", "reference_wet", *report["macro_mean"]]
    verified = 0
    for record in report["records"]:
        match = re.fullmatch(r"TubeTech_a_(\d+)_r_(\d+)_r_(\d+)_t_(\d+)_g_(\d+)", record["id"])
        if not match:
            raise ValueError(f"Unknown recording naming: {record['id']}")
        attack, release, ratio, threshold, gain = map(int, match.groups())
        record["knobs"] = dict(attack=attack, release=release, ratio=ratio,
                               threshold=-threshold, gain=gain)
        signals = {}
        record["audio"] = {}
        record["waveforms"] = {}
        record["residuals"] = {}
        for label in labels:
            path = source / "audio" / label / f"{record['id']}.wav"
            signal, rate = sf.read(path, dtype="float64")
            if rate != 48000 or signal.shape != (record["samples_scored"],):
                raise ValueError(f"Unaligned audio: {path}")
            if not np.isfinite(signal).all():
                raise ValueError(f"Non-finite audio: {path}")
            signals[label] = signal
            wav_bytes = path.read_bytes()
            benchmark_hash.update(wav_bytes)
            record["audio"][label] = base64.b64encode(wav_bytes).decode("ascii")
            record["waveforms"][label] = envelope(signal)
            verified += 1
        wet = signals["reference_wet"]
        for label, expected in record["metrics"].items():
            residual = signals[label] - wet
            mse = float(np.mean(residual ** 2))
            actual = {"mse": mse, "mae": float(np.mean(np.abs(residual))),
                      "esr": mse / float(np.mean(wet ** 2))}
            for key, value in actual.items():
                if not np.isclose(value, expected[key], rtol=1e-5, atol=1e-10):
                    raise ValueError(f"Metric mismatch: {record['id']} {label} {key}")
            record["residuals"][label] = envelope(residual)
        record["frames"] = frame_analysis(signals)
        record["spectral_details"] = {}
        for label, metrics in record["metrics"].items():
            spectral = mrstft(signals[label], wet)
            metrics["mrstft"] = spectral["total"]
            metrics["rms_db_mae"] = region_metrics(record["frames"], label)["rms_db_mae"]
            record["spectral_details"][label] = spectral["resolutions"]
    for label, metrics in report["macro_mean"].items():
        for key, value in metrics.items():
            computed = np.mean([r["metrics"][label][key] for r in report["records"]])
            if not np.isclose(value, computed, rtol=1e-12):
                raise ValueError(f"Macro mean mismatch: {label}/{key}")
        for key in ("mrstft", "rms_db_mae"):
            metrics[key] = float(np.mean([r["metrics"][label][key] for r in report["records"]]))
    report["benchmark_id"] = benchmark_hash.hexdigest()
    full_results = full_corpus_results()
    if full_results is not None:
        report["full_corpus"] = full_results
    architecture = architecture_results(full_results)
    if architecture is not None:
        report["architecture"] = architecture
    sequence = sequence_results()
    if sequence is not None:
        report["sequence_study"] = sequence
    common = five_model_results()
    if common is not None:
        report["five_models"] = common
    if (S6_PINN_RUN / "protocol.json").exists():
        report["s6_pinn_study"] = {"run": S6_PINN_RUN.name}
        for key, filename in (("protocol", "protocol.json"), ("status", "status.json"),
                              ("history", "history.json"), ("best", "best-validation.json")):
            if (S6_PINN_RUN / filename).exists():
                report["s6_pinn_study"][key] = json.loads((S6_PINN_RUN / filename).read_text(encoding="utf-8"))
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = (UI / "index.html").read_text(encoding="utf-8")
    if common is not None and "s6_pinn" in common["sources"]:
        template = template.replace("六種方法", "七種方法").replace("五模型＋純演算法", "六模型＋純演算法")
        template = template.replace("SIX METHODS", "SEVEN METHODS")
        template = template.replace("S4、S6、RiccardoVib。", "S4、S6、S6＋TFiLM＋PINN、RiccardoVib。")
        template = template.replace("<br><b>RiccardoVib：</b>既有 CL1B 權重與固定條件校準。",
            f"<br><b>S6＋TFiLM＋PINN：</b>代表視窗訓練 16 輪，固定第 {common['sources']['s6_pinn']['selected_epoch']} 輪；共用電路方程約束與狀態輸出。"
            "<br><b>RiccardoVib：</b>既有 CL1B 權重與固定條件校準。")
        template = template.replace("這次只重新推論，沒有重新訓練或依新結果挑選權重。",
            "原六種方法保留既有結果；新增 S6＋TFiLM＋PINN 重新訓練 16 輪，以同一組驗證視窗選模後加入。")
    html = template.replace("/*__STYLE__*/", (UI / "style.css").read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", payload)
    script = "\n".join((UI / name).read_text(encoding="utf-8")
                       for name in ("app.js", "analysis.js", "full-test.js", "architecture.js", "sequence-study.js", "five-models.js", "s6-pinn-study.js"))
    html = html.replace("/*__SCRIPT__*/", script)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    # Keep an inspectable, audio-free companion without altering the original benchmark.
    metrics_report = {**report, "records": [
        {k: v for k, v in r.items() if k not in ("audio", "waveforms", "residuals")}
        for r in report["records"]]}
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics_report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified {verified} WAV clips and all 90 per-record metrics.")
    print("Computed 60 additional spectral/envelope scores; no subjective scores or regions were fabricated.")
    print(f"Built {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


def build(output: Path, *, public_site: bool = False) -> None:
    """Build the consolidated report; historical data remain in their run folders."""
    if __package__:
        from pinn.build_unified_comparison import build_report
    else:
        from build_unified_comparison import build_report
    build_report(output, five_model_results(),
                 audio_base='audio/listening' if public_site else '../runs/pretrained-comparison/audio',
                 write_metrics=not public_site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/cl1b-comparison.html")
    parser.add_argument("--public-site", action="store_true",
                        help="Build site/index.html with publishable audio paths and no metrics sidecar")
    args = parser.parse_args()
    output = ROOT / "site/index.html" if args.public_site else args.output
    build(output, public_site=args.public_site)
