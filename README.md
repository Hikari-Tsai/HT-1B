# HT-1B：標準訓練與電路求解流程

本專案將 CL 1B 的縮減電路方程式用在兩條路徑：

- **直接求解**：數值求解器讀取輸入音訊，計算電路狀態並輸出 WAV，不需要神經網路。
- **PhysicsNeMo 訓練**：使用配對音檔與方程式約束，辨識模型參數，再把參數交給直接求解器處理聲音。

下方步驟可直接在 macOS 的終端機執行。首次建議使用已提供錄音的**前 2 秒**確認流程；這不是完整資料集訓練，也不能代表對新音訊的還原度。

**模型範圍**：已保留前端電阻網路、C3 充放電與控制模式，GRE 使用雙狀態近似。真空管採無記憶飽和近似、變壓器視為理想元件，尚未完成整機逐元件復刻。[方程式與假設](docs/EQUATIONS.md)

## 1. 安裝與檢查環境

需要 Python 3.12 與 `uv`。在本機執行：

```bash
cd /Users/hikaritsai/HT-1B
uv sync --locked --extra train --extra test --python 3.12
source .venv/bin/activate

python -c "import torch, physicsnemo; print('Torch:', torch.__version__); print('PhysicsNeMo:', physicsnemo.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest -q
ht1b --help
```

目前使用 `nvidia-physicsnemo[sym]==2.2.2`，不需要另外安裝舊的 `nvidia-physicsnemo-sym`。

- Apple Silicon／沒有 NVIDIA GPU：使用 `--device cpu`。這是本專案已驗證的環境。
- 已配置可用 CUDA 的環境：可以使用 `--device cuda`；上方 CUDA 檢查需為 `True`。本機未測試 CUDA 執行。
- 不使用 `mps`，因為目前訓練要求 float64。

若不用 uv，可自行建立 Python 3.12 虛擬環境後執行 `python -m pip install -e '.[train,test]'`；這個方法不會自動依 uv.lock 固定所有依賴版本。

## 2. 確認音檔格式與命名

你的資料集是**單一立體聲 WAV**：左聲道為輸入 dry，右聲道為壓縮後輸出 wet。

```text
TubeTech_a_0_r_0_r_10_t_0_g_0.wav
         │   │    │    │   └─ makeup gain：0 dB
         │   │    │    └──── threshold：0
         │   │    └───────── ratio：10:1
         │   └────────────── release 檔位：0，最快
         └────────────────── attack 檔位：0，最快
```

| 參數 | 資料集允許值 |
|---|---|
| Attack | 0、1、2、3、4，從快到慢，不是毫秒 |
| Release | 0、1、2、3、4，從快到慢，不是秒 |
| Ratio | 2、4、6、8、10，代表 2:1 至 10:1 |
| Threshold | 0、−10、−20、−30、−40 |
| Gain | 由 `g_` 後的數值讀取，按 dB 解讀 |

不要把左右聲道分別做音量正規化，也不要將它們混成 mono。`prepare` 會將檔名轉成模型控制值，並保留原始標籤與映射假設。**檔位到電阻值、threshold 與 ratio 的映射仍是近似，並非已量測的硬體校準表。**

## 3. 建立本次資料清單，先取得未訓練基準

以下使用獨立的本次執行名稱，避免覆蓋既有結果。**後續步驟請在同一個終端機執行**，以保留這兩個變數。

```bash
HT1B_RUN="run-$(date +%Y%m%d-%H%M%S)"
HT1B_WAV="/Users/hikaritsai/Downloads/TubeTech_a_0_r_0_r_10_t_0_g_0.wav"

ht1b prepare "$HT1B_WAV" \
  --output "data/$HT1B_RUN.json" \
  --config configs/default.json \
  --seconds 2 \
  --mode manual \
  --delay-samples 0

ht1b evaluate "data/$HT1B_RUN.json" \
  --config configs/default.json \
  --split train \
  --output "runs/$HT1B_RUN/baseline.json"
```

此步會建立 manifest（資料清單），指向原始音檔，不會修改原檔。

- `--seconds 2`：只使用原檔前 2 秒。省略會選取完整音檔；不是自動分割成 2 秒小段。
- `--mode manual`：目前對這批四參數掃描資料的假設，檔名本身沒有模式。若已知是 Fixed 或 Fix/Man，改成 `fixed` 或 `fix-man`。
- `--delay-samples 0`：本檔前段互相關得到的候選整數延遲為 0。若 wet 比 dry 晚 32 個 samples，設定為 `32`。
- 初始狀態預設完全釋放。不要隨意裁取錄音中段，再將狀態當作零。

可直接開啟 `data/<本次名稱>.json` 檢查來源路徑、控制值與 `mapping_assumptions`。資料夾與檔案說明見 [詳細訓練文件](docs/TRAINING.md)。

## 4. 執行第一次 PhysicsNeMo 訓練

```bash
ht1b train "data/$HT1B_RUN.json" \
  --config configs/default.json \
  --output "runs/$HT1B_RUN" \
  --steps 500 \
  --batch-size 256 \
  --width 64 \
  --layers 3 \
  --learning-rate 0.001 \
  --physics-weight 1.0 \
  --data-weight 1.0 \
  --seed 7 \
  --checkpoint-every 100 \
  --device cpu
```

| 參數 | 用途 |
|---|---|
| `steps` | optimizer 更新次數，不是完整掃過資料的 epoch 數 |
| `batch-size` | 每步抽取的時間點數；程式先選一個 train 錄音，再抽該錄音的時間點 |
| `width`、`layers` | 軌跡網路的寬度與層數 |
| `learning-rate` | 網路與待辨識參數的學習率 |
| `physics-weight` | 方程式殘差的權重，必須大於零 |
| `data-weight` | 配對輸出波形誤差的權重 |
| `seed` | 初始化與抽樣的隨機種子 |
| `checkpoint-every` | 每隔多少步保存一次 checkpoint，最後一步也會保存 |

終端機會印出 `step`、`loss`、`physics`、`data`。總 loss 包含方程式殘差、按參考能量縮放的波形誤差，以及小幅參數正則項。Loss 下降只代表訓練目標改善，仍須做下一步的直接求解驗證。

訓練完成後：

```text
runs/<本次名稱>/
├── baseline.json   # 未訓練直接求解器的比較數字
├── history.jsonl   # 每一步的 loss
├── last.pt         # 網路、物理參數、optimizer 與隨機狀態
├── fitted.json     # 自動匯出的電路參數
└── report.json     # 訓練摘要、資料假設與 PINN 軌跡重建指標
```

## 5. 用訓練結果重新求解，輸出音檔與比較數字

```bash
ht1b evaluate "data/$HT1B_RUN.json" \
  --config "runs/$HT1B_RUN/fitted.json" \
  --split train \
  --output "runs/$HT1B_RUN/deployment-train.json" \
  --audio-dir "output/$HT1B_RUN"

python - "$HT1B_RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path("runs") / sys.argv[1]
before = json.loads((root / "baseline.json").read_text())["records"]
after = json.loads((root / "deployment-train.json").read_text())["records"]
for name in after:
    print(name)
    for metric in ("mse", "mae", "esr"):
        print(f"  {metric}: {before[name][metric]:.6g} -> {after[name][metric]:.6g}")
print("處理後音檔：", Path("output") / sys.argv[1])
PY
```

這裡是**直接求解器**重新處理輸入，右聲道只用來評分。輸出為 FLOAT WAV，不會偷偷正規化或截去超過 0 dBFS 的樣本；報告會列出峰值。

- MSE、MAE、ESR 越低表示波形誤差越小，不能轉成「還原度百分比」。
- 此處 `--split train` 評估的是同一訓練區間，**不是獨立驗證集**。
- `report.json` 的 `trajectory_reconstruction` 是座標 PINN 的抽樣重建；不能取代這裡的部署結果。
- 直接求解目前是離線 CPU 參考實作。先前 2 秒、48 kHz 音訊約花 18 秒，不能當作即時外掛。

## 6. 續訓與參數匯出

沿用原 manifest、原始 config 與訓練選項。`--steps 1000` 代表**再跑 1000 步**，因此接續上面的 500 步後會到 1500 步。

```bash
ht1b train "data/$HT1B_RUN.json" \
  --config configs/default.json \
  --output "runs/$HT1B_RUN" \
  --resume "runs/$HT1B_RUN/last.pt" \
  --steps 1000 \
  --batch-size 256 \
  --width 64 \
  --layers 3 \
  --learning-rate 0.001 \
  --physics-weight 1.0 \
  --data-weight 1.0 \
  --seed 7 \
  --checkpoint-every 100 \
  --device cpu
```

**不要把 `--config` 改成 fitted.json 來續訓**：程式會校對最初的 config 與資料簽章，實際訓練狀態由 last.pt 還原。恢復到新終端機時，先啟用 `.venv`，再把 `HT1B_RUN` 設成之前的資料夾名稱，不要重新產生名稱。

fitted.json 會自動更新，也可以另存一份：

```bash
ht1b export "runs/$HT1B_RUN/last.pt" "configs/$HT1B_RUN-fitted.json"
```

匯出的是可重用的**電路參數**。其中 controls 欄位是預設控制值，並非原錄音設定；處理資料集錄音時繼續用 `evaluate`，由 manifest 提供每個錄音的設定。

要處理一般新 mono 音檔，可用 `simulate`，但需選定模型的正規化旋鈕值：

```bash
ht1b simulate /path/to/input-mono.wav output/new-audio.wav \
  --config "runs/$HT1B_RUN/fitted.json" \
  --threshold 0.7 --ratio 0.5 --attack 0.15 --release 0.3 \
  --makeup-db 0 --mode manual
```

這組旋鈕值只是示例。`simulate --ratio 0.5` 是模型電位器位置，不等於直接指定 0.5:1；請勿把檔名中的 ratio 10 或 attack 4 直接放進此命令。

## 7. 多檔訓練與真正的驗證集

先將要用的錄音放在指定目錄，建立一份清單。下面的路徑需換成你的資料夾：

```bash
HT1B_BATCH="batch-$(date +%Y%m%d-%H%M%S)"
ht1b prepare /path/to/dataset/TubeTech_*.wav \
  --output "data/$HT1B_BATCH.json" \
  --config configs/default.json \
  --seconds 2 --mode manual
```

`prepare` 會把所有紀錄預設為 `train`，**不會自動分割驗證集**。訓練前開啟此 JSON，把要保留的整筆錄音改成：

```json
"split": "validation"
```

至少保留一筆 train 和一筆 validation；也可指定 `test`。優先按照**原始音訊素材**分組，同一素材在不同旋鈕設定下的錄音不要跨到 train／validation 造成內容洩漏。只有一個檔案時，先做上述流程驗證，不能宣稱已做獨立素材驗證。

分割完成後再訓練並評估：

```bash
ht1b train "data/$HT1B_BATCH.json" \
  --config configs/default.json \
  --output "runs/$HT1B_BATCH" \
  --steps 2000 --batch-size 256 --device cpu

ht1b evaluate "data/$HT1B_BATCH.json" \
  --config configs/default.json \
  --split validation \
  --output "runs/$HT1B_BATCH/validation-baseline.json"

ht1b evaluate "data/$HT1B_BATCH.json" \
  --config "runs/$HT1B_BATCH/fitted.json" \
  --split validation \
  --output "runs/$HT1B_BATCH/validation-fitted.json" \
  --audio-dir "output/$HT1B_BATCH-validation"
```

目前訓練迴圈不會自動按 validation 選最佳 checkpoint，也沒有 early stopping；last.pt 就是最後一次保存的狀態。需自行比較驗證集的結果。

**長音檔**：省略 prepare 的 `--seconds` 可載入全長，但現有座標 PINN 在長時間軸上的快速動態可能欠擬合。它沒有自動滑動視窗與狀態銜接；不要為了切片而把每段狀態重置為零。2 秒流程通過之後，再擴大資料長度與設定覆蓋範圍。

## 8. 不使用實機資料的環境自我檢查

下面會產生有明確標記的合成左右聲道配對檔，確認安裝與訓練管線能運作：

```bash
HT1B_DEMO="demo-$(date +%Y%m%d-%H%M%S)"
ht1b demo-data "data/$HT1B_DEMO" --sample-rate 8000 --seconds 0.15
ht1b train "data/$HT1B_DEMO/manifest.json" \
  --output "runs/$HT1B_DEMO" --steps 150 --device cpu
ht1b evaluate "data/$HT1B_DEMO/manifest.json" \
  --config "runs/$HT1B_DEMO/fitted.json" \
  --split validation --output "runs/$HT1B_DEMO/validation.json"
```

合成資料由同一個近似模型產生，只能驗證程式流程，不能證明已還原真實 CL 1B。

## 常見問題

| 現象 | 處理方式 |
|---|---|
| `ht1b: command not found` | `cd` 至專案並執行 `source .venv/bin/activate`，確認已安裝專案 |
| 找不到 `physicsnemo` | 安裝時需包含 `--extra train`；只裝求解器依賴不含訓練框架 |
| `Manifest already exists` | 重用現有清單或選新名稱，不要覆蓋正在訓練的資料清單 |
| `Output already has a checkpoint` | 接續訓練要加 `--resume`；新實驗使用新輸出目錄 |
| `Resume data/config signature mismatch` | 還原原始音訊、資料清單、控制值與 config，或另開新實驗 |
| `Resume training options mismatch` | width、layers、batch-size、learning-rate、loss 權重與 seed 要和原訓練一致 |
| `No recordings match selected split` | prepare 預設全部為 train；先在 manifest 指定 validation，或用 `--split train` 做訓練區間檢查 |
| 檔名解析失敗 | 檢查兩個 r 的順序與允許值；其他命名可手動建立 manifest |
| Loss 下降，但直接求解誤差變大 | 檢查對齊、初始狀態與控制映射，分開查看物理殘差和資料誤差，不只比較總 loss |
| 求解不收斂 | 可在 simulate／evaluate 加 `--substeps 2`；這是縮小積分步長，不是完整抗混疊過取樣 |

## 程式與進一步說明

| 檔案 | 用途 |
|---|---|
| [equations.py](src/ht1b/equations.py) | NumPy／Torch／SymPy 共用的電路表達式 |
| [solver.py](src/ht1b/solver.py) | 有狀態的隱式數值求解器 |
| [physicsnemo_model.py](src/ht1b/physicsnemo_model.py) | 自訂 CircuitPDE、軌跡 PINN、受限物理參數 |
| [training.py](src/ht1b/training.py) | 訓練、checkpoint、續訓與匯出 |
| [dataset.py](src/ht1b/dataset.py) | 資料集檔名與控制值映射 |
| [詳細訓練文件](docs/TRAINING.md) | 自訂 manifest、獨立 mono 配對、延遲、初始狀態 |
| [方程式與近似](docs/EQUATIONS.md) | 電路公式、求解方法與模型邊界 |
| [實測紀錄](docs/VALIDATION.md) | 已完成的數值、合成與真實錄音驗證 |

`CircuitPDE` 是本專案繼承 PhysicsNeMo `PDE` 的類別，用來計算方程式殘差，並非另外的套件。軌跡網路擬合錄音內部狀態；部署新音訊時使用匯出參數與直接求解器，不將單一錄音的時間網路當成通用音訊模型。

先前已以提供音檔的前 2 秒完成 500 步訓練與直接求解驗證，尚未完成全 210 秒訓練及跨素材／跨旋鈕的實機驗證。單一設定無法可靠辨識所有模型參數。原始研究脈絡見 [研究交接](docs/RESEARCH_HANDOFF.md)。音檔、權重與原始手冊保留在本機，不加入版本追蹤。
