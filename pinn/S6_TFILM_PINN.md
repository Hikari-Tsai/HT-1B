# S6＋TFiLM＋PINN：2026-09-29

沿用原 S6/TFiLM backbone，將增益輸出 head 改成三個有界物理狀態：
e = 30 sigmoid(raw_e − 6)，f/s = sigmoid(raw_f/s − 3)。
藉共享的 reduced-circuit waveform 公式產生 Wet，不保留繞過物理公式的自由音訊分支。
這是簡化電路，不是完整 CL-1B 網表。S6 隱藏狀態並非直接當作電路狀態。

`ht1b.physicsnemo_model.circuit_residual_terms` 同時供舊 `CircuitPDE`
與本次 batched Torch 訓練使用；方程式只有一份。測試核對 Torch 與
PhysicsNeMo SymPy 編譯結果一致、狀態接續、因果性及梯度。

## 資料、模型與目標

- 原 v3 manifest，620 檔，8,680 train／1,240 validation 區段。
- 16 epochs，batch 256，seed 7。相同代表視窗位置與 shuffle，非完整資料樣本覆蓋。
- 每視窗 5,120 samples：前 4,096 Dry-only no_grad 暖機，後 1,024 計算 loss。
- 暖機最後 e/f/s 接入第一個計分樣本的後向差分，不在錄音中段強制物理狀態為零。
- 網路 hidden state 逐視窗重設；短暖機不足以確立長 release 歷史，這個限制與舊比較一起揭露。
- 網路使用 normalized filename labels；電路使用 manifest approximate potentiometer positions。
  makeup 僅在電路公式中套用一次，兩組 conditioning 不可混用。
- fresh network，原始 Circuit 參數。初始化平滑常數 latent head，避免初始逐樣本噪音的 stiff derivative。
- loss = L1 + 0.1 ESR + 0.1 MR-STFT + mean(C3/GRE-fast/GRE-slow residual²) + 0.0001 physical prior。
- 神經網路 float32 / lr 0.001；共享電路、latent 差分 float64 / physical lr 0.0001。
- AdamW，weight_decay 0，clip_norm 1。只有 train Wet 進入 optimizer。
- GPU 冒煙步驟後重新初始化正式模型；不把 smoke 更新帶入正式訓練。
- 每輪驗證，以 macro ESR 最低選模；重載最佳 checkpoint，再算五項共同指標。
- 未開啟 test；validation 已重用，不宣稱新獨立測試。
- 同時改變輸出 head 與目標，不是只加 loss 的單變因消融。
- 本次直接以 neural state + circuit readout 推論；舊 MLP/GRU 比較是 fitted-parameter numerical solver。

## 自動執行

`python -u -m s6.pinn_experiment --wait-pids <existing WSL job PIDs>`

以 /proc 的 PID 與啟動識別值等待既有作業；等待時不建立 CUDA 模型。
使用者後續要求直接共用 GPU，因此已停止尚未訓練的等待程序，以 `--start-queued`
啟動同一實驗。此選項僅接受沒有 checkpoint、且原程序已退出的第 0 輪排隊目錄；
舊排隊紀錄保存在 `previous-queue/`，其他專案程序不受影響。
既有作業結束後先執行 GPU smoke，再跑完整 16 輪及最終共同驗證。
輸出 `runs/s6-tfilm-pinn-gpu-20260929/`，包括 protocol、queue、status、history、
gpu-smoke、best/last checkpoint、final-validation 及 common-validation 報告。
原六模型的成績保持不變；新增第七模型，共用相同 windows signature。
最後自動執行 `pinn.build_comparison_page.build`。

## 網頁擴充設計

Extension 模式，保留原本錨點、篩選器、排序、CSV、音訊及歷史實驗。
沿用米白／墨綠、Bahnschrift／微軟正黑體、現有小圓角與間距，新增色取原 S6 色與文字色混合。
視覺變異 1、動態 1、資訊密度 7、素材依賴 1、既有風格保留 10。
新實驗狀態與方法可折疊檢視；完成後出現在共同比較卡片、圖表、矩陣與 CSV。
頁面為離線快照，訓練中以排程回報提供進度，不能冒稱即時資料。
