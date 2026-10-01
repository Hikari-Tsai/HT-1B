"""Write the completed full-corpus experiment's auditable Markdown test report."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from build_comparison_page import ROOT, full_corpus_results


METRICS = ("esr", "mse", "mae", "mrstft", "rms_db_mae")


def fmt(value: float, key: str) -> str:
    return f"{value:.6g}" if key != "mse" else f"{value:.6e}"


def metric_table(rows: list[tuple[str, dict]]) -> str:
    lines = ["| 評估對象 | ESR ↓ | MSE ↓ | MAE ↓ | MR-STFT ↓ | RMS dB MAE ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for title, values in rows:
        lines.append("| " + title + " | " + " | ".join(fmt(values[k], k) for k in METRICS) + " |")
    return "\n".join(lines)


def group_table(records: list[dict], label: str) -> str:
    values = sorted({r["labels"][label] for r in records})
    title = "Ratio" if label == "ratio" else "Threshold (dB)"
    lines = [f"| {title} | 檔案數 | ESR ↓ | MR-STFT ↓ | RMS dB MAE ↓ |",
             "|---|---:|---:|---:|---:|"]
    for value in values:
        group = [r for r in records if r["labels"][label] == value]
        averages = [statistics.mean(r["metrics"][key] for r in group)
                    for key in ("esr", "mrstft", "rms_db_mae")]
        lines.append(f"| {value} | {len(group)} | " + " | ".join(f"{v:.6g}" for v in averages) + " |")
    return "\n".join(lines)


def build(output: Path) -> None:
    results = full_corpus_results()
    if results is None:
        raise RuntimeError("Full-corpus test is not complete; report would be premature")
    records = results["records"]
    test = results["test"]
    selected = results["selected"]
    selected_name = "PhysicsNeMo 擬合參數" if selected == "fitted" else "未訓練電路基準"
    validated = results["validation"]
    vals = sorted(r["metrics"]["esr"] for r in records)
    p90 = vals[int(0.9 * (len(vals) - 1))]
    worst = sorted(records, key=lambda r: r["metrics"]["esr"], reverse=True)[:10]
    split = results["samples_by_split"]
    hours = lambda n: n / 48000 / 3600
    lines = [
        "# CL-1B 全資料訓練與獨立區段測試報告", "",
        f"- 實驗：`{results['run']}`；完成時間：{results['completed_utc']}（UTC）。",
        f"- 在全部 620 個檔案的訓練區間完成一輪，精確涵蓋 {results['train_samples']:,} 個樣本（{hours(split['train']):.3f} 配對音訊小時）。",
        f"- 驗證集：{split['validation']:,} 樣本（{hours(split['validation']):.3f} 小時）；測試集：{split['test']:,} 樣本（{hours(split['test']):.3f} 小時）。時數跨旋鈕設定累加，不是獨特演奏時數。",
        f"- 選模規則：只比較驗證集 macro ESR，鎖定 **{selected_name}** 後，對測試集評分一次；測試結果未參與選模。", "",
        "## 驗證集選模", "",
        metric_table([("未訓練電路基準", validated["baseline"]),
                      ("PhysicsNeMo 全資料擬合參數", validated["fitted"])]), "",
        f"驗證 ESR 變化（擬合相對基準）：{(validated['fitted']['esr'] / validated['baseline']['esr'] - 1) * 100:+.2f}%。",
        "驗證位於共同內容時間軸的約 90–107.7 秒與 149.4–160 秒。此表只用於選模，不等於獨立測試表現。", "",
        "## 鎖定模型的測試結果", "",
        metric_table([(f"{selected_name} · 620 檔 macro mean", test)]), "",
        f"逐檔 ESR 中位數：{statistics.median(vals):.6g}；90 百分位：{p90:.6g}。Macro mean 是 620 檔各自分數的等權平均，不是把所有樣本合併計算的 ESR。", "",
        f"測試 macro ESR 比選模時的驗證 ESR 高 {(test['esr'] / validated[selected]['esr'] - 1) * 100:.1f}%；兩集合的素材位置不同，這項差距不能單獨解讀為泛化退化。", "",
        "測試區間位於共同內容時間軸的約 199.7 秒至各檔末尾；評估時從各原始檔 sample 0 起，使用完整 Dry 歷史連續推進電路狀態，僅在測試區間對照 Wet。未做輸出增益、延遲或目標狀態擬合。", "",
        "### 按旋鈕條件檢視", "",
        "以下每格為該組檔案的等權平均，便於找出誤差集中在哪些設定；不同組別的素材總長可能不同。", "",
        group_table(records, "ratio"), "", group_table(records, "threshold_db"), "",
        "### 測試 ESR 最高的 10 個設定", "",
        "| 檔案 ID | ESR ↓ | MR-STFT ↓ | 測試秒數 |", "|---|---:|---:|---:|",
    ]
    for r in worst:
        lines.append(f"| `{r['id']}` | {r['metrics']['esr']:.6g} | {r['metrics']['mrstft']:.6g} | {r['samples']/48000:.2f} |")
    lines.extend(["", "## 解讀範圍", "",
                  "- 素材區分依 Dry 聲學變化與重複片段比對推定；沒有歌曲或演奏者的人工標籤，因此不能證明測試素材完全獨立。不同旋鈕設定下的同一素材被分到相同 split。",
                  "- 舊互動頁的 5 段約 2 秒三模型比較採不同資料與評估長度，不能與本報告數值直接排成同一榜。RiccardoVib 模型尚未在本輪完整測試集評估，因此沒有三模型獨立測試排名。",
                  "- 全資料訓練同時調整了取樣方法、PINN 輸入與網路設定；驗證差異不能單獨歸因於資料量。",
                  "- ESR、MSE、MAE 是時域誤差；MR-STFT 是多解析度頻譜誤差，RMS dB MAE 是 10 ms 包絡誤差。指標定義見互動頁與 `src/ht1b/audio_metrics.py`。", "",
                  "## 可稽核資料", "",
                  f"- `runs/{results['run']}/training/coverage.json`：全訓練樣本與逐檔覆蓋。",
                  f"- `runs/{results['run']}/baseline-validation.json`、`fitted-validation.json`：選模依據。",
                  f"- `runs/{results['run']}/selection.json`：測試前鎖定的模型。",
                  f"- `runs/{results['run']}/selected-test.json`：620 檔逐檔、逐區間和 macro 結果。",
                  "- `runs/full-corpus-20260925-v3/manifest.json`：內容分割與原始樣本邊界。", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/full-corpus-test-report.md")
    build(parser.parse_args().output)
