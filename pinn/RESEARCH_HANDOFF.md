# HT-1B 研究交接

更新日期：2026-09-19

## 使用者意圖與目前狀態

使用者先要求搜尋 GitHub 的 CL 1B 演算法，若無則從電路圖推導。已找到神經網路實作，隨後討論白箱、灰箱、神經網路架構、PINN，以及直接從電路圖寫演算法。

最新指示：「我希望接下來移動到新的專案 HT-1B」。本目錄為獨立專案；HT-76 原始碼沒有搬移或修改。

目前僅完成研究交接，未執行模型訓練、音訊推論或 C++ DSP 實作。使用者尚未明確選定最終建模方法。不要把先前助手提出的方案視為已批准的規格。

## 建模方向的討論

- 白箱：依電路圖與元件模型，使用節點分析、狀態空間或 WDF，再離散化成逐樣本 DSP。
- 灰箱：保留已知電路及控制結構，用可校準的方程式近似未知元件。
- 神經網路：LSTM/GRU、TCN/WaveNet、CNN+LSTM encoder-decoder、S4D/S6 等；旋鈕 conditioning 與主幹架構須分開考慮。
- PINN：透過方程式殘差約束訓練；不同於把小型網路直接放入微分方程式的 UDE/混合 Neural ODE。
- 助手最近建議（尚未定案）：先做電路推導與可校準的 GRE 行為模型，有量測後再決定是否加入神經網路。

## 已核對的電路圖

來源：
https://funkwerkes.com/web/wp-content/techdocs/MixedProAudio/Tube-Tech-CL1B-Compressor-SM.pdf

本地副本：`references/CL1B-service-manual.pdf`，14 頁，原始下載檔。

已將 PDF 第 10–14 頁渲染並人工視讀：

- 第 10–11 頁：側鏈電路，含 LF347N、整流二極體、RC、固定與可調路徑、GRE 驅動與計量電路。
- 第 12–13 頁：ECC83/ECC82 音訊放大級、輸入輸出變壓器、回授與電源。
- 第 14 頁：前面板旋鈕及模式切換連線。
- 圖面標示 1993 年、Rev 1.0；不可假設適用所有硬體版本。
- 關鍵缺口：GRE（gain reduction element）畫成模組，圖上未交代完整內部結構或動態曲線。
- 變壓器接線與型號不等於完整磁性元件模型；真空管與光學元件也需適當參數及校準。
- 尚未逐節點建立 netlist，沒有完成電路求解器或確認所有掃描元件值。

官方操作及校準手冊：
https://www.tube-tech.com/wp-content/uploads/2020/05/Tube-Tech-Manual-CL-1B-200513.pdf

確認：音訊路徑含輸入變壓器、增益衰減元件、真空管放大級及輸出變壓器；有 Fixed、Manual、Fix/Man 模式。混合模式的 release 取決於峰值長度與旋鈕設定，不能直接當成單一固定 release。不同版本與序號的元件差異需要留意。

來源副本僅為研究便利保存。保留原始來源及原作者權利；不要推定可以隨產品或公開 Git 倉庫再散布維修手冊。

## 2023 年神經網路論文及程式

論文：Fully Conditioned and Low-latency Black-box Modeling of Analog Compression
https://www.dafx.de/paper-archive/2023/DAFx23_paper_10.pdf

程式：
https://github.com/RiccardoVib/CONDITIONED-MODELING-OF-OPTICAL-COMPRESSOR

已透過 GitHub API 與 raw source 確認：

- Python / TensorFlow / Keras，程式檔頭標示 LGPL v3 或更新版本。
- `Fully_Conditioned_Black_Box_Model_for_Compression/Code/Models.py` 含 Conv1D + LSTM + Dense 的 encoder-decoder。
- `Models/CL1BModel/Checkpoints/best/` 含 checkpoint 與 weights.h5。
- 未執行推論或驗證權重與當前 runtime 的相容性。

研究對象是實體 CL 1B 的輸入輸出行為，不是逐元件電路重建。四個條件參數為 threshold、ratio、attack、release；輸出 gain 固定，輸出級失真不在研究範圍。

性能核對：

| CL 1B 模型 | epochs | 測試 MSE | 測試 MAE | 參數量 |
|---|---:|---:|---:|---:|
| 比較用 TCN | 50 | 9.54e-4 | 1.33e-2 | 51,464 |
| 作者 ED | 50 | 1.74e-5 | 1.93e-3 | 24,912 |
| 作者 ED | 200 | 8.21e-6 | 未列 | 24,912 |

- 表 2、3 明確標示新音訊、已見 conditioning values；不可當作任意未見旋鈕設定的保證。
- 48 kHz 下最佳選定 ED 固有延遲 16 samples，約 0.33 ms；論文指定雙重 I/O buffering 假設為 80 samples，約 1.66 ms，非任意 DAW 的端到端延遲。
- 重度壓縮及較長時間常數更困難，分段邊界可能產生額外音調；存在可聽見的瑕疵。
- 指標：MSE、MAE；另試驗 ESR、多解析度 STFT 訓練 loss，最終採 MSE。也以波形、RMS 包絡與非正式聆聽分析。
- 沒有完整 CPU 實測基準或正式盲聽分數；MSE 不等於還原度百分比。
- 表 7 FLOPs/sample 與 GFLOPS 欄位依 48 kHz 換算有疑點，若引用計算成本應重新核算。

## 後續研究來源

較新光學壓縮器 SSM 研究，CL 1B / LA-2A：
https://arxiv.org/abs/2408.12549
https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs

研究以 S6 / selective SSM、FiLM / GLU 等預測增益係數；GitHub README 列有 LSTM、ED、S4D、Mamba 比較模型及權重。僅查閱論文與 README，尚未下載／執行驗證。

可比較多種黑箱與灰箱架構的 NablAFx：
https://github.com/mcomunita/nablafx

即時 C++ 神經網路推論：
https://github.com/jatinchowdhury18/RTNeural

Neural ODE 音訊電路研究：
https://arxiv.org/abs/2205.01897

PINN：
https://maziarraissi.github.io/PINNs/

UDE：
https://arxiv.org/abs/2001.04385

## 建議的下一個具體工作

在使用者決定繼續開發後，先將電路圖整理為元件清單、連線表及未知參數清單。分開標記「圖上確定」、「元件模型假設」、「需實機校準」，再決定首版實作範圍。

驗證可涵蓋：靜態壓縮曲線、不同持續時間的 tone burst attack/release、三種模式、頻率響應、THD/IMD、長時間穩定性、取樣率與旋鈕自動化、CPU 與延遲、音量匹配聆聽。尚無本專案實測結果。

## 2026-09-24 實作更新

使用者已授權兩條實作路徑：直接求解演算法、PhysicsNeMo 音檔訓練。已新增 Python package、CLI、共用方程式、隱式求解器、PhysicsNeMo 2.2.2 PINN、訓練／續訓／匯出與測試。

使用者提供 Downloads 下的 TubeTech_a_0_r_0_r_10_t_0_g_0.wav，並確認資料格式：左乾右濕、a/release 為 0–4 檔位、ratio 2/4/6/8/10、threshold 0/-10/-20/-30/-40。已使用前 2 秒完成初步訓練和直接求解驗證。

模型是縮減電路灰箱，未完成全元件 netlist；真空管、變壓器、GRE 等近似及驗證範圍見 EQUATIONS.md、TRAINING.md、VALIDATION.md。以這些文件取代本交接上文「尚未實作」的歷史狀態。
