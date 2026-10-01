"""Add the unfitted circuit to the existing common-window model comparison."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from ht1b.config import Controls, read_config
from ht1b.fast_solver import FastCircuitSolver
from pinn.evaluate_five_models import ROOT, OUT, METRICS, score, summarize, sha
from s6.representative_experiment import load_batch, write_json, utc, WARMUP, SCORE


def run():
    output = ROOT / "runs/six-model-common-validation-20260925"
    if (output / "comparison.json").exists():
        raise FileExistsError("Six-method comparison already exists")
    report = json.loads((OUT / "comparison.json").read_text())
    manifest = json.loads((ROOT / "runs/full-corpus-gpu-20260925/manifest.json").read_text())
    if report["manifest_signature"] != manifest["signature"]:
        raise ValueError("Manifest mismatch")
    originals = {r["id"]: r for r in manifest["records"]}
    index = json.loads((OUT / "windows.json").read_text())
    if index["window_signature"] != report["window_signature"]:
        raise ValueError("Window signature mismatch")
    rows = [dict(id=r["id"], path=str(ROOT / "data/tubetech-cl-1b-v1" / originals[r["id"]]["path"]),
                 start=r["warmup_start"], controls=(0.,0.,0.,0.), gain=0.) for r in index["windows"]]
    if len(rows) != report["windows"] or len(report["intervals"]) != len(rows):
        raise ValueError("Window count mismatch")
    dry, wet, _, _ = load_batch(rows, torch.device("cpu"))
    config = ROOT / "runs/full-corpus-gpu-20260925/initial-config.json"
    circuit, _ = read_config(config)
    grouped, scores = defaultdict(list), []
    for i, row in enumerate(rows):
        span = report["intervals"][i]
        if span["id"] != row["id"] or span["warmup_start"] != row["start"] or span["stop_sample"]-span["start_sample"] != SCORE:
            raise ValueError("Window order or scoring length mismatch")
        solver = FastCircuitSolver(circuit, Controls(**originals[row["id"]]["controls"]), 48000)
        prediction = solver.process(dry[i].numpy().astype(np.float64))[WARMUP:]
        values = score(prediction, wet[i,WARMUP:].numpy())
        scores.append(values)
        grouped[row["id"]].append(values)
        span["metrics"]["baseline"] = {k:values[k] for k in METRICS}
    for record in report["records"]:
        values = grouped[record["id"]]
        np.testing.assert_allclose(sum(v["target_energy"] for v in values), record["target_energy"], rtol=1e-12)
        record["metrics"]["baseline"] = summarize(values)
    report["macro_mean"]["baseline"] = {k:float(np.mean([r["metrics"]["baseline"][k] for r in report["records"]])) for k in METRICS}
    report["pooled"]["baseline"] = summarize(scores)
    report["sources"]["baseline"] = dict(path=str(config.relative_to(ROOT)),sha256=sha(config),
        method="unfitted original circuit parameters; zero state then identical 4096-sample Dry warmup; CPU numerical solver")
    report["baseline_added_utc"] = utc()
    report["parent_comparison_sha256"] = sha(OUT / "comparison.json")
    write_json(output / "windows.json", index)
    write_json(output / "comparison.json", report)
    write_json(output / "status.json", dict(phase="complete",utc=utc(),models=6,files=620,windows=1240))
    print(json.dumps(dict(baseline_macro=report["macro_mean"]["baseline"],baseline_pooled=report["pooled"]["baseline"])))


if __name__ == "__main__":
    run()
