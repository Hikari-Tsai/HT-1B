# PhysicsNeMo：S4＋TFiLM 與 S6＋Temporal FiLM 訓練

兩套新模型的程式、設定、測試與文件集中在專案根目錄的 `s6/`；資料夾同時包含 S4 和 S6。以下命令都在 **HT-1B 專案根目錄**執行，入口已改為 `python -m s6`。

```text
s6/
├── __main__.py        # prepare / check / train / evaluate
├── models.py         # PhysicsNeMo 模型
├── layers.py         # S4、S6、TFiLM 等元件
├── data.py           # 左 dry／右 wet 讀檔
├── losses.py
├── training.py
├── configs/          # 兩套預設設定
├── tests/
├── docs/plans/
└── README.md
```

沿用 HT-1B 的環境與檔名解析函式，並非可單獨複製出去的獨立套件。

兩套程式都讀取**單一立體聲 WAV：左聲道是 dry 輸入，右聲道是 wet 目標**。模型只看到 dry 與旋鈕；wet 僅用於 loss／評分。輸出是單聲道預測。

本次交付只撰寫程式與執行非訓練檢查，**沒有啟動訓練或更新模型權重**。下方 `train` 指令供之後手動執行。

## 1. 模型與 PhysicsNeMo 的關係

`s6.models.AudioModel` 繼承真正的 `physicsnemo.core.Module`，使用 PhysicsNeMo 2.2.2 的 metadata、`.save()`、`.from_checkpoint()` 與 `.mdlus` 模型格式；訓練迴圈使用 PyTorch autograd／AdamW。CPU 和 CUDA 共用同一個可微實作，不依賴 CUDA-only Mamba 套件。

這是**使用 PhysicsNeMo 的監督式神經音訊建模**，不是加入電路方程式殘差的 PINN。原有 `ht1b train`、`CircuitPDE`、直接電路求解器仍是獨立流程；新入口為 `python -m s6`。不把無物理意義的神經隱藏狀態強行當作電容電壓。

| 模型 | 計算路徑 | 預設規模 |
|---|---|---|
| `s4_tfilm` | 輸入投影 → 多個「Linear／tanh → 複數對角 SSM → LSTM TFiLM → tanh」→ 輸出投影／tanh | 8 blocks、16 channels、每通道 32 個複數模態、128-sample TFiLM block |
| `s6_tfilm` | 最近 64 點 → 投影 → S6 → FiLM／GLU → GRU Temporal FiLM／GLU → S6 → 增益 → 乘上 dry | 2 blocks、8 channels、16 states、2 倍內部展開 |

S4 採 S4D 類型的複數對角狀態空間核心，用 ZOH 離散化、因果 FFT 卷積及顯式初始狀態項；不是完整 DPLR S4。TFiLM 使用區段 MaxPool＋LSTM，產生每通道的縮放與偏移。

S6 使用 `A=-exp(log_A)`、輸入相依的 B/C/Δ，以及 Mamba-1 類型的遞迴更新：

```text
h[n] = exp(Δ[n] A) h[n-1] + Δ[n] B[n] u[n]
z[n] = C[n] h[n] + D u[n]
```

每個 S6 區塊含因果 depthwise convolution、SiLU 門控及 GELU 輸出。條件模組使用 64 點歷史的 128 點 FFT 幅度特徵：threshold／ratio 經全連接層控制 FiLM，attack／release 經 GRU 控制 Temporal FiLM；GLU 使用 softsign。

### 與作者程式的差異

這是**依討論架構重新撰寫的 PyTorch／PhysicsNeMo 參考實作**，不是原作者 TensorFlow 權重的移植，也沒有宣稱重現論文指標。

- S4 的離散化、初始化、複數狀態數及可選連接並非逐項複製 NablaFX；不能直接比參數量或載入它的權重。
- S6 採標準 selective SSM 與時間軸因果卷積，維度可調；並非逐行移植 Riccardo 的小型 Keras block。頻譜以線性投影壓縮。
- FiLM 用 `(1 + gamma) * z + beta`，S6 增益用 `1 + Linear(z)`，以接近單位增益初始化。增益不經 sigmoid，也不以 wet/dry 相除建立標籤。
- 檔名 `gain_db` 作為模型輸出後的固定倍率 `10**(gain_db/20)`，不送入四旋鈕條件網路。這是 makeup gain 的工程假設；現有 g=0 資料不受影響，其他 gain 設定仍需驗證。
- 共用的 loss、TBPTT 長度與訓練設定是本專案的起始設定，並非論文最佳超參數。

## 2. 安裝

沿用專案既有環境，不需新增依賴：

```bash
cd /Users/hikaritsai/HT-1B
uv sync --locked --extra train --extra test --python 3.12
source .venv/bin/activate
python -m s6 --help
```

新模型使用 float32，S4 內部為 complex64。先使用 `--device cpu`；有 NVIDIA GPU 可改 `cuda`。本次只檢查 CPU。未啟用 AMP、MPS、分散式訓練或編譯。

S6 的 selective scan 是逐取樣 Python 參考迴圈，容易理解但訓練可能很慢。CUDA 能執行不等於已獲得 fused Mamba kernel 的效能，也未驗證即時外掛處理速度。

## 3. 建立資料清單，不會訓練

檔名繼續使用：

```text
TubeTech_a_0_r_0_r_10_t_0_g_0.wav
         attack release ratio threshold gain
```

神經模型使用以下控制值，**不套用電路 PINN 的近似電阻映射**：

| 順序 | 原始標籤 | 正規化 |
|---|---|---|
| 0 | threshold：0 至 −40 | `-threshold_db / 40` |
| 1 | ratio：2 至 10 | `(ratio - 2) / 8` |
| 2 | attack 檔位：0 至 4 | `attack_index / 4` |
| 3 | release 檔位：0 至 4 | `release_index / 4` |

Attack/release 檔位不是秒。檔名解析沿用資料集的五個離散值；自訂 manifest 的 `parameters` 可填範圍內插值。此版不支援 Fixed／Fix-Man 等模式切換條件。

只有目前這一份 210 秒音檔時，可做時間分割：

```bash
python -m s6 prepare \
  /Users/hikaritsai/Downloads/TubeTech_a_0_r_0_r_10_t_0_g_0.wav \
  --output data/neural-pairs.json \
  --sample-rate 48000 \
  --validation-fraction 0.2 \
  --gap-seconds 1 \
  --delay-samples 0
```

前 80% 為 train，之後跳過 1 秒 guard，再使用剩下部分作 validation；兩者各自暖機。不覆蓋已存在的清單。

這只是**同一錄音的時間保留驗證**，不能證明跨音源或跨旋鈕泛化。若對多個檔案省略 `--validation-files`，每個檔案都採上述時間分割。

已有獨立驗證音檔時，明確分組：

```bash
python -m s6 prepare /path/to/train/TubeTech_*.wav \
  --validation-files /path/to/validation/TubeTech_*.wav \
  --output data/neural-heldout.json --sample-rate 48000
```

此時每個檔案使用全長，不做時間分割；`validation-fraction`／`gap-seconds` 不參與分組。請將相同原始素材的不同旋鈕版本放在同一 split。程式能檢查相同路徑的區間重疊，但不能從音檔內容自動辨識重複素材。

自訂格式如下，所有 sample 區間均為左閉右開：

```json
{
  "schema_version": 1,
  "records": [
    {
      "id": "voice-train",
      "stereo_pair": "recordings/pair.wav",
      "split": "train",
      "source_id": "voice-take-01",
      "start_sample": 0,
      "stop_sample": 480000,
      "delay_samples": 0,
      "parameters": {
        "threshold_db": -20,
        "ratio": 6,
        "attack_index": 1,
        "release_index": 3,
        "gain_db": 0
      }
    }
  ]
}
```

以上只有一筆示例，啟動訓練前還必須加入 `validation` 紀錄。可另外指定 `test`，訓練迴圈不讀取它的樣本。若提供 `source_id`，同來源跨 split 會被拒絕；時間切分測試請省略這個分組欄位。相對路徑以 manifest 所在目錄為基準。

延遲的定義為 `dry[n]` 對齊 `wet[n + delay_samples]`；正值表示 wet 較晚。支援正負整數延遲，不自動估計或重採樣。檔案必須是雙聲道 WAV，取樣率符合 config，且數值有限。左右聲道不分別正規化、不混成 mono。

## 4. 只檢查，不訓練

```bash
python -m s6 check data/neural-pairs.json \
  --config s6/configs/neural-s4-tfilm.json --device cpu

python -m s6 check data/neural-pairs.json \
  --config s6/configs/neural-s6-tfilm.json --device cpu
```

`check` 校對所有紀錄的 metadata、範圍與分割，讀取每個 split 第一筆紀錄的第一個 chunk，進行隨機初始化模型的前向運算。它不建立 optimizer、不更新權重、不寫 checkpoint。它也不掃描整份音訊或驗證訓練效果。

## 5. 之後手動啟動訓練

**以下指令會真的訓練；本次沒有執行。** 選擇新目錄，兩個模型不要共用訓練輸出目錄。

```bash
python -m s6 train data/neural-pairs.json \
  --config s6/configs/neural-s4-tfilm.json \
  --output runs/neural-s4-tfilm --device cpu

python -m s6 train data/neural-pairs.json \
  --config s6/configs/neural-s6-tfilm.json \
  --output runs/neural-s6-tfilm --device cpu
```

| 設定 | 行為 |
|---|---|
| `epochs` | 完整掃過所有 train 錄音的總回合數；續訓時也代表總目標 |
| `chunk_samples=1024` | TBPTT 的梯度截斷長度，48 kHz 約 21.3 ms，不是完整記憶長度 |
| `warmup_samples=49152` | 每筆紀錄開始先用 1.024 秒建立狀態，不計 loss；需為 chunk 長度的整數倍 |
| `grad_clip=1` | 梯度 norm 上限；非有限 loss／梯度會中止 |
| `l1_weight`、`esr_weight`、`spectral_weight` | 波形 L1、ESR、多解析度 STFT loss 的權重 |
| `fft_sizes` | STFT 尺度；很短的尾段補零後計算頻譜，不把補零樣本納入波形 loss |

採單筆串流（batch=1）：每個 epoch 可打亂**錄音順序**，同一錄音的 chunks 永遠依時間排列。SSM、卷積歷史、LSTM／GRU 狀態會一路保留，每次權重更新後只 detach 計算圖。換錄音或 split 才重設。

這種 TBPTT 保留長歷史的狀態值，但梯度仍只跨一個 chunk；短 chunk 不保證能學好數秒 release。可逐步增加 chunk（S4 必須是 film_block 的整數倍），並同步調整 warmup。預設 1.024 秒暖機也不保證涵蓋最慢 release；若裁取錄音中段，應依歷史改成更長，例如 491520 samples（10.24 秒），並確保每筆紀錄仍有可評分樣本。

S4 TFiLM 需要等一個 128-sample block 才能計算該 block 的調節，48 kHz 約 2.67 ms 的緩衝量；不是逐取樣零前視。S6 使用目前與過去的 64 點，歷史視窗不是 64 點的未來前視。這些數字都不是整個應用的端到端延遲測量。

訓練 loss：

```text
L = w_l1 * mean(abs(prediction - wet))
  + w_esr * mean(error²) / max(mean(wet²), 1e-8)
  + w_spectral * mean_over_FFT_sizes(spectral_convergence + log_magnitude_L1)
```

沒有把 wet/dry 當增益標籤，避免輸入零點的除零問題。驗證集每回合串流評估 MSE、MAE、ESR，以全體有效樣本累積，不對 chunks 的 ESR 簡單平均；用 validation ESR 選最佳 checkpoint。ESR 有靜音能量下限，不能解讀成還原度百分比。

## 6. 輸出、續訓與評估

```text
runs/neural-s6-tfilm/
├── config.json
├── manifest.json          # 原資料清單的快照；相對音檔路徑仍以原清單位置解讀
├── history.jsonl
├── latest.json            # 最近完整 epoch 的路徑
├── best.json              # validation ESR 最佳 epoch 的路徑
├── report.json
└── checkpoints/
    └── epoch-000001/
        ├── model.mdlus    # PhysicsNeMo 模型結構＋權重
        └── training.pt    # optimizer、epoch、config、資料 hash、RNG 狀態
```

每個 epoch 完成訓練與驗證才存一次完整 checkpoint，再更新指標檔。中斷於 epoch 中途時，從上一個完整 epoch 重跑；不保存錄音中段的 runtime state。所有 epochs 的模型都保留，長期訓練需自行管理空間。`.pt`／`.mdlus` 和 `runs/` 已被 Git 忽略。

先查 `latest.json`，將 `--resume` 指向它列出的實際 epoch 目錄。例如：

```bash
python -m s6 train data/neural-pairs.json \
  --config s6/configs/neural-s6-tfilm.json \
  --output runs/neural-s6-tfilm \
  --resume runs/neural-s6-tfilm/checkpoints/epoch-000001 \
  --device cpu
```

若總回合數已完成，先把 config 的 `epochs` 提高。其他 config 值與原始 manifest／音檔內容必須一致；續訓會核對完整資料 SHA-256。沿用原始 manifest 路徑，不將輸出目錄的快照直接當成可移動的資料集。CPU／CUDA 切換可能改變數值，沒有跨硬體 bitwise 一致的保證。

依 `best.json` 指向的目錄評估；以下 epoch 編號只是範例：

```bash
python -m s6 evaluate data/neural-pairs.json \
  --checkpoint runs/neural-s6-tfilm/checkpoints/epoch-000001 \
  --split validation --device cpu \
  --output output/neural-s6-validation.json
```

有獨立 test 紀錄時改 `--split test`，不要用 test 選模型。這個命令只評分，不更新權重或輸出音檔。

單獨載入模型與持續處理的介面：

```python
import torch
from s6.models import AudioModel

model = AudioModel.from_checkpoint("runs/my-run/checkpoints/epoch-000001/model.mdlus")
model.eval()
state = None
controls = torch.tensor([[0.5, 0.5, 0.25, 0.75]], dtype=torch.float32)
# dry_chunk: float32 [batch, samples]，已對齊，S4 chunk 需為 film_block 整數倍。
with torch.no_grad():
    wet_chunk, state = model(dry_chunk, controls, state)
# 下一個連續 chunk 繼續傳 state；新錄音再設 None。
```

## 7. 驗證範圍與程式位置

以下只做數值與程式契約測試，**不執行 optimizer.step，也不呼叫 trainer**：

```bash
python -m pytest s6/tests/test_neural.py -q
```

測試涵蓋乾濕聲道／延遲、控制標籤、分割重疊、chunk 與整段處理一致性、S6 因果性、S4 區段緩衝、有限梯度、loss、PhysicsNeMo 未訓練模型存取、唯讀 check 與驗證尾段計分。舊的 `tests/test_training.py` 會訓練 PINN；若只想做非訓練檢查，使用 `pytest --ignore=tests/test_training.py`。

本機驗證：35 項非訓練測試通過；提供的 48 kHz 立體聲錄音通過兩套預設架構的 1024-sample 唯讀前向檢查。

尚未執行這兩套新模型的 optimizer 更新、完整訓練、續訓後收斂實驗、CUDA 效能或硬體還原度測試；不能從前向／梯度測試推論聲音品質。

| 檔案 | 功能 |
|---|---|
| `s6/data.py` | raw 旋鈕、manifest、lazy stereo reader、對齊與分割檢查 |
| `s6/layers.py` | 對角 SSM、selective SSM、TFiLM、GLU、state detach |
| `s6/models.py` | PhysicsNeMo AudioModel 與兩套架構 |
| `s6/losses.py` | 波形與 multi-resolution STFT loss |
| `s6/training.py` | TBPTT、validation、checkpoints、續訓與唯讀檢查 |
| `s6/__main__.py` | CLI |

研究來源：[NablaFX](https://github.com/mcomunita/nablafx)、[Frontiers 音效建模比較](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full)、[Riccardo 光學壓縮器 S6 論文](https://arxiv.org/abs/2408.12549)、[Mamba](https://arxiv.org/abs/2312.00752)。
