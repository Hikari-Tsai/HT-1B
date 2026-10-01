# 因果 GRU＋PhysicsNeMo 全資料實驗

本輪只更換訓練時的**狀態軌跡網路**；物理電路方程式、七個受限電路參數、
48 kHz 隱式數值求解器與原始資料分割沿用前一輪。它是反向 PINN，部署時仍
匯出電路參數供求解器使用，不把逐檔軌跡網路當作通用音訊模型。

`pinn/train_gru_pinn.py` 使用 PyTorch 單層 GRU（hidden 64）讀取每個 10 ms
Dry 框的 RMS、峰值及五個旋鈕控制值；**該框更新後的 GRU 狀態只供下一框使用**，
當前樣本只看到本身 Dry、先前框的 GRU 歷史、控制值與絕對時間，不偷看同框未來。
後端使用 PhysicsNeMo `FullyConnected`（寬 64、`num_layers=3`、tanh）輸出
C3 電壓與兩個 GRE 潛在狀態。狀態輸出受界並在整檔 t=0 滿足零初始狀態。

資料依錄音檔順序與原始時間向前串流；每錄音檔的 GRU 隱狀態跨 15,360 樣本
微批次傳遞並於微批次邊界截斷梯度，**不在 30 秒片段或保留區段重置**。
驗證／測試區間僅用 Dry 推進歷史，Wet 不參與參數更新。每個 train 樣本
都進入波形與電路殘差損失；目標沿用物理殘差＋區塊能量正規化波形 MSE＋
`1e-4` 物理參數先驗。以 Adam 同時訓練 GRU／解碼器（LR 0.001）與共用電路參數
（LR 0.0001），使用 RTX 5060 Ti CUDA、float64。

使用同一個 `runs/full-corpus-20260925-v3/manifest.json`，必須完整走訪
5,100,828,000 個 train 樣本。`--smoke-chunks` 只供環境測試，不能當作正式結果。
每約 60 秒寫入可續訓 checkpoint、覆蓋量、訓練狀態與匯出參數。

```bash
cd /mnt/c/Users/user/Documents/ChatGPT/HT-1B
/home/hikari/.venvs/ht1b/bin/python -u pinn/run_gru_experiment.py \
  --manifest runs/full-corpus-20260925-v3/manifest.json \
  --dataset data/tubetech-cl-1b-v1 \
  --output runs/gru-pinn-gpu-20260925
```

包裝程式在全覆蓋後，用與 MLP 相同的 validation 區間及五項指標評估匯出電路，
以 validation macro ESR 比較 MLP、GRU 與未訓練基準。之後也會評估原 test
區間，但它已在先前實驗開封，**只能當架構探索的再使用測試**，不能宣稱全新
獨立測試。正式泛化結論仍需新的 Dry／Wet 素材。
