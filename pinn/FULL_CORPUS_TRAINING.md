# 完整資料訓練與內容隔離評估

本輪使用 Kaggle v1 的全部 **620 個 WAV**；125 個長 210 秒、494 個長 220 秒，
1 個約 219.915 秒。總長約 37.54 小時，是不同旋鈕設定的配對錄音時數，不代表
37.54 小時獨特素材。原始檔案不裁切、不正規化、不修改。

## 分割規格

正式清單：`runs/full-corpus-20260925-v3/manifest.json`。
v1 是偵測到異常時間軸後中止的掃描；v2 尚未合併重複演奏片段，均不供正式訓練。

- 以五個 release／時長族群的完整 Dry 參考，計算 MFCC 音色變化；相鄰區間的
  特徵差異形成 novelty curve，自動找出 17 個聲學段落。
- 以正規化 inner product（cosine）分組音色特徵，再搜尋 2 秒 Dry 波形的
  增益／極性不變近似重複（相關係數 ≥ 0.90，跳過自身 ±5 秒）。目前找到 15 個
  有方向的匹配，將涉及區段合併為 11 個群組；這不是 15 個獨特素材。
- 同一群組在所有旋鈕檔案中共用 split，不按檔案隨機分配。
- 驗證：共同內容時間軸的 **90–107.7 秒、149.4–160 秒**，兩段有重複素材，必須一起保留。
- 測試：**199.7 秒至檔尾**。其餘全部訓練。短於 220 秒的檔案按實際檔尾結束。
- `TubeTech_a_4_r_2_r_2_t_30_g_0.wav` 內容提前 0.75 秒；所有分割邊界對應提前，
  原始 Dry/Wet 保持對齊且不修改。校正後 Dry 包絡相關約 0.997。
- 每個原始樣本恰好屬於一個 split；不丟掉安靜片段、區塊餘數或檔尾。

| 集合 | 配對音訊時數 | 用途 |
|---|---:|---|
| train | 29.518681 | 梯度更新，全 620 檔的可用區間 |
| validation | 4.873889 | 選擇訓練後參數或未訓練基準 |
| test | 3.149074 | 選擇鎖定後評估一次，不選模型 |

這是**依 Dry 聲學分組推定的內容隔離**，不是經人工確認的歌曲／演奏者獨立測試。
相似度可找到近似重複，但不能保證找出所有不同變奏、移調或同曲片段。
原始素材標籤缺失，因此報告必須持續保留這個限制。

## 訓練方式

`pinn/train_full_corpus.py` 使用 RTX 5060 Ti 的 CUDA / float64，CUDA 不可用會報錯。
按完整聲學區段串流讀檔（最多 30 秒一個 I/O 區塊），每批最多 16,384 個**連續樣本**。
每輪打亂區塊順序，但完整走訪所有批次；一個 epoch 精確包含 5,100,828,000 個
訓練樣本，不能以固定 5,000 steps 當作全資料訓練。

為避免巨大 one-hot 輸入，座標 PINN 改用每錄音 16 維 embedding 與較多 Fourier
時間特徵；PhysicsNeMo MLP 仍為寬 64、3 層。區塊使用整檔絕對時間與同一 embedding，
不在切片起點假設完全釋放。錄音中段的內部狀態是 PINN 的潛在估計，並非量測值或
已求解的真實前置狀態；每個原始錄音 t=0 才沿用零初始狀態假設。

損失沿用物理殘差、能量正規化波形 MSE 與物理 prior；網路 learning rate 0.001，
物理參數 learning rate 0.0001。此輪先完成一次全覆蓋，未直接添加頻譜訓練 loss。
網路、取樣方式與學習率均有調整，因此此輪不是只改資料量的單變因消融實驗。
原先 3,000 步模型與比較頁仍保留。

每約 60 秒在完整區塊邊界原子寫入 checkpoint、狀態與逐檔／逐群樣本覆蓋數。
`--resume` 支援從 `last.pt` 恢復，並驗證 manifest、模型與 optimizer 選項一致。

## 評估與長時間執行

`pinn/run_full_experiment.py` 依序執行：

1. 一次完整 GPU 訓練 pass。
2. 在全部 validation 區間比較初始電路與訓練後電路，按 macro ESR 選擇。
3. 將選擇寫入 `selection.json` 並鎖定 `selected-config.json`。
4. 僅此時執行全部 test 區間，輸出 MSE、MAE、ESR、MR-STFT 與 RMS dB MAE。

部署評估從原始檔案 sample 0 持續處理 **Dry 歷史**，狀態不在保留區段起點重置；
Wet 只在指定 split 的評分區間使用。不同區段不直接串接後計算 STFT。
此方式以真實 Dry 歷史為條件評估，並非零歷史起始的測試；訓練不讀取保留區間作梯度更新。

完整音訊使用新增的可選 Numba scalar implicit solver 加速，與原本 SciPy coupled
backward-Euler solver 在多個旋鈕、物理參數與 substep 設定通過輸出／狀態對照。
此加速器只支援 manual 模式，原 SciPy 版本保持原狀。
WSL 環境額外安裝 `numba==0.67.0`、`llvmlite==0.49.0`；其餘模型套件未更換。

```bash
cd /mnt/c/Users/user/Documents/ChatGPT/HT-1B
/home/hikari/.venvs/ht1b/bin/python -u pinn/run_full_experiment.py \
  --manifest runs/full-corpus-20260925-v3/manifest.json \
  --dataset data/tubetech-cl-1b-v1 \
  --output runs/full-corpus-gpu-20260925
```

狀態：`runs/full-corpus-gpu-20260925/experiment-status.json` 與
`training/status.json`；樣本使用證據在 `training/coverage.json`。
完整覆蓋與完整測試只有在對應狀態完成後才能宣稱，不以「已啟動」代替「已完成」。

## 本輪完成結果

`runs/full-corpus-gpu-20260925` 已完成一輪全資料訓練：620 檔訓練區間共
5,100,828,000 樣本，覆蓋率 100%。驗證集 macro ESR：未訓練基準
0.363170，PhysicsNeMo 擬合參數 0.251958；依預定規則鎖定擬合參數後，
620 檔測試區間的 macro ESR 為 0.272594，MR-STFT 為 1.096236。
完整五項指標、逐設定分布、限制與稽核路徑見
[`output/full-corpus-test-report.md`](../output/full-corpus-test-report.md)，
互動篩選見 [`output/cl1b-comparison.html`](../output/cl1b-comparison.html)。
