# 2026-09-24 驗證紀錄

環境：Apple Silicon macOS、Python 3.12、PhysicsNeMo 2.2.2、PyTorch 2.14.0、CPU／float64。套件鎖定於 uv.lock。結果是本次實際執行紀錄，不是理論預估。

## 自動測試

`python -m pytest -q` 檢查：

- 零輸入、分段與整段處理一致。
- GRE 無驅動時，與獨立手算 backward-Euler RC 衰減一致。
- 被動前端電阻網路 KCL、三種控制模式。
- 縮小時間步長的收斂、極端旋鈕的有限輸出及狀態邊界。
- 左乾右濕振幅與延遲對齊、錯誤聲道／長度／取樣率拒絕。
- 三種模式下 PhysicsNeMo 符號式與數值式一致、參數梯度有效。
- PINN 初始狀態與邊界。
- 真實 optimizer 訓練 loss 下降、checkpoint、參數匯出。
- 中斷續訓與不中斷優化的參數一致。
- 命令列合成資料 → 求解 → 驗證流程。
- 同名雙 r 欄位解析、0–4 檔位映射。

此測試確認軟體／數值一致性，沒有證明簡化模型等於實體電路。

## 合成資料

8000 Hz、0.15 秒、兩個不同波形；由直接求解器生成 dry/wet stereo。第一個用來訓練，第二個是獨立合成波形 holdout。

- 150 步訓練，固定評估點總 loss：3.483845 → 0.085568。
- 匯出參數後，數值求解器在合成 holdout 上 ESR：0.000289406。
- 合成 truth 原本就是同一個模型的預設參數，初始 solver 本來就匹配；此測試是流程驗證，**不能當作真實參數辨識或實機泛化成功**。
- 產物：`runs/synthetic-smoke/`、`data/synthetic/`（本機、未追蹤）。

## 使用者的 CL 1B 配對音檔

來源：`/Users/hikaritsai/Downloads/TubeTech_a_0_r_0_r_10_t_0_g_0.wav`。

- 48000 Hz、雙聲道 PCM16、210 秒、10,080,000 frames。
- 左聲道輸入、右聲道壓縮輸出；保留原始比例。
- 原檔只讀取，未更動或覆寫。
- 前 5 秒、±4096 samples 的互相關檢查，最大值候選延遲為 0 samples；不代表沒有亞取樣延遲或頻率相位差。
- **只選前 2 秒 96,000 frames**，假設起點已釋放、Manual 模式、使用文件中的檔位映射。
- CPU 訓練 500 步，batch=256，寬度 64、3 層，lr=0.001。
- 固定評估點總 loss：0.472168 → 0.054651。

| 評估項目 | 未訓練直接求解器 | 匯出參數後的直接求解器 |
|---|---:|---:|
| MSE | 0.001051946 | 0.000616602 |
| MAE | 0.021810518 | 0.016867030 |
| ESR | 0.125973363 | 0.073839798 |
| 2 秒音訊計算耗時 | 約 18.6 秒 | 約 18.1 秒 |

這些數字來自**同一個訓練區間**。沒有執行完整 210 秒訓練、沒有不同真實素材 holdout、沒有跨參數驗證。不能解讀為整機還原度、通用準確度或即時效能。

PINN 對最多 4096 個抽樣點的軌跡重建 ESR 約 0.057760；與表中的直接求解器 ESR 不同，兩者不可混用。

本機產物：

- `data/tubetech-prefix.json`：來源與參數／假設。
- `runs/tubetech-smoke/last.pt`：可續訓 checkpoint。
- `runs/tubetech-smoke/fitted.json`：已辨識的電路近似參數。
- `runs/tubetech-smoke/report.json`：訓練與重建報告。
- `runs/tubetech-smoke/deployment-train.json`：直接求解數字。
- `runs/tubetech-baseline.json`：未訓練比較。
- `output/tubetech-baseline/`、`output/tubetech-fitted/`：2 秒的直接求解輸出。

## 尚未覆蓋的硬體範圍

GRE、真空管與變壓器的物理還原、全五檔位組合、非 Manual 模式實機對照、長 release、多種輸入電平、取樣率轉換／反混疊與即時 CPU 優化，均未完成實機驗證。程式提供的是可執行且可校準的基礎，不是已驗證可替代 CL 1B 的商用音訊模型。
