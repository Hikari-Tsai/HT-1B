# Windows / WSL GPU 訓練與演算法比較

`pinn` 放操作文件與本輪執行腳本；核心模型及訓練迴圈在 `src/ht1b`。

本機 GPU：NVIDIA GeForce RTX 5060 Ti 16 GB。訓練環境放在 WSL Ubuntu 的 `/home/hikari/.venvs/ht1b`，與 Windows `.venv` 分開；依 `uv.lock` 安裝 PhysicsNeMo 2.2.2 和 PyTorch 2.14.0。訓練使用 CUDA / float64，CUDA 不可用時會報錯，不會切到 CPU。

在 WSL、專案根目錄執行（每輪使用新的 output 目錄）：

```bash
cd /mnt/c/Users/user/Documents/ChatGPT/HT-1B
UV_PROJECT_ENVIRONMENT=/home/hikari/.venvs/ht1b uv sync --locked --extra train --extra test --python 3.12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hikari/.venvs/ht1b/bin/python -u \
  pinn/run_gpu_training.py --output runs/gpu-pilot-new --steps 2000
```

腳本先執行 CUDA float64 運算檢查，選取 25 組涵蓋五種 attack、release、ratio、threshold 檔位的錄音。20 組訓練，5 組保留設定做驗證；每組使用原始音檔前 2 秒，batch 512、寬度 64、3 層、learning rate 0.001、seed 7，每 100 步保存 checkpoint。可用 `--seconds` 和 `--steps` 調整範圍；這個腳本是初步實驗，不是全 620 檔、全長訓練。

比較涵蓋全部 5 組保留設定及 1 組訓練設定：

- 基準：預設電路參數 + CPU 數值求解器。
- 訓練後：GPU PINN 辨識的電路參數 + 同一 CPU 數值求解器。
- 參考：實機錄音的右聲道 wet。兩者使用相同輸入、取樣率、控制值、初始狀態與時間區段。

CPU 求解是現有離線演算法的實作；神經網路訓練在 GPU。`report.json` 的 PINN 軌跡重建只適用於訓練錄音，其數字與上述部署比較分開記錄。

產物位於指定 output 目錄：

| 檔案 | 用途 |
|---|---|
| `status.json` | 執行階段或失敗原因 |
| `run-config.json` | GPU、套件、訓練參數與假設 |
| `manifest.json` | 20 組 train、5 組 validation 的來源清單 |
| `initial-config.json` | 本輪初始電路參數快照 |
| `history.jsonl`、`last.pt`、`fitted.json` | loss、可續訓 checkpoint、辨識參數 |
| `report.json` | 訓練摘要、GPU 資訊、CUDA 峰值配置記憶體 |
| `baseline-evaluation.json`、`fitted-evaluation.json` | 逐錄音 MSE、MAE、ESR 與求解耗時 |
| `comparison.json`、`COMPARISON.md` | 比較與五組保留設定的平均誤差 |
| `audio-reference/`、`audio-baseline/`、`audio-fitted/` | 相同區段的 dry、wet 與演算法輸出 |

驗證保留的是旋鈕設定，尚未證明音訊素材獨立，不能宣稱跨素材泛化。Manual 模式、零整數延遲、起點完全釋放及旋鈕映射沿用原專案假設。誤差降低百分比不是硬體還原度；試聽時請保持相同播放增益。

## 從 checkpoint 分階段續訓

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hikari/.venvs/ht1b/bin/python -u \
  pinn/resume_gpu_training.py \
  --run runs/gpu-pilot-20260925-0300 \
  --output runs/gpu-continuation-2000-to-5000 \
  --target-step 5000 --interval 1000
```

`--target-step` 是目標總步數；从第 2,000 步接到第 5,000 步只增加 3,000 步。每一階段保留自己的 checkpoint、loss 紀錄與驗證音檔；沿用原資料、初始 config、optimizer 和訓練選項。程式檢查每次恢復的固定 loss 與前次結束一致，並驗證相同五組保留設定。原始實驗目錄保持不變，輸出目錄必須是新的。

`progression.json` / `COMPARISON.md` 彙整各階段結果。`last.pt` / `fitted.json` 是最後完成的步數；`best.pt` / `best-fitted.json` 依保留設定平均 ESR 選擇，候選包含起始的第 2,000 步。同一保留集用於選擇模型，不能當成另一個獨立測試集。
