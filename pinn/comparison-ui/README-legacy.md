# CL 1B 互動比較頁

使用 `runs/pretrained-comparison/comparison.json`、其 `audio/` 下的 40 段 WAV，
以及 `runs/gpu-continuation-2000-to-5000/progression.json` 產生離線 HTML。

在專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe pinn/build_comparison_page.py
```

輸出為 `output/cl1b-comparison.html`（約 20 MB）。直接以瀏覽器開啟即可，
不需要伺服器、網路、CDN 或 Python。音訊已嵌入 HTML；複製檔案時會一併帶走
評估片段，分享前請留意原始資料來源的授權。

完整資料實驗結束後，產生器會核對 `runs/full-corpus-gpu-20260925/` 的選模紀錄、
訓練覆蓋率、manifest signature 與 620 檔測試指標，再於頁面頂部加上獨立測試區塊。
該區塊可切換五項指標，按 Attack／Release／Ratio／Threshold 篩選，檢視
逐檔誤差及下載全部篩選結果 CSV。獨立測試只有驗證後鎖定的模型，與原本五段
短片段三模型比較分開呈現。

GRU＋PINN 實驗完成後，產生器另外核對 `runs/gru-pinn-gpu-20260925/` 的
完整訓練覆蓋率、保留 Wet 未參與訓練、manifest 簽章、620 檔評估區間和各指標平均值。
頁面會加入獨立的架構比較區：可切換驗證／探索性測試、五項指標與 Ratio，
查看逐檔差異並匯出 CSV。架構選擇仍依驗證 ESR；測試集已在先前實驗開封，
GRU 的測試結果只供探索性比較，不稱為新的獨立測試，也不補造純演算法測試分數。

RiccardoVib 的固定權重若已用 `pinn/evaluate_riccardovib_full.py` 在相同的 620 檔
驗證與測試區間完成評估，會加入上述架構比較與逐檔表。作者原始訓練素材可能重疊，
因此其分數只描述同一輸入上的誤差，不構成與本方模型的公平泛化排名。

詳細文字報告可執行：

```powershell
.\.venv\Scripts\python.exe pinn/build_full_test_report.py
```

產生 `output/full-corpus-test-report.md`；頁面提供相鄰連結。

功能包括：

- ESR / MSE / MAE、五組設定與平均值的比較。
- MR-STFT（FFT 512/1024/2048，hop 128/256/512）與 10 ms RMS 包絡 dB MAE。
- PhysicsNeMo 2,000 / 3,000 / 4,000 / 5,000 步切換。
- Dry、Wet、純演算法、PhysicsNeMo、RiccardoVib 模型的同位置音軌切換。
- 真實振幅包絡、模型相對 Wet 的殘差包絡、共用播放音量與循環播放。
- 固定 Loss 與驗證 ESR 的獨立零基準折線圖。
- 深淺色模式、完整 CSV 與選定音軌下載、模型來源與評估限制。
- 手動標記 Attack / Release 區段，以 10 ms 格點計算 MSE、MAE、ESR、RMS dB MAE；
  不自動推斷類比 Threshold 所對應的數位振幅，不宣稱測得硬體時間常數。
- 個人盲測：三模型、隱藏 Wet 與 Dry 隨機排列，全部分數初始留白，試聽與填寫後才能送出。
  當輪固定設定與 checkpoint；完成後才揭示身份，提供 JSON 匯出。這不是正式 MUSHRA 研究。
- 盲測與手動標記嘗試保存在瀏覽器 localStorage（依 benchmark ID 隔離）；不可用時明確顯示
  記憶體限定狀態。未完成的盲測僅保留於當次開啟。分享 HTML 不會附帶瀏覽器內的評分。

產生器會重新讀取每段 WAV，檢查 48 kHz、單聲道及 95,968 樣本對齊，
並從音訊重算全部 90 個逐段誤差指標，確認與原始評估結果一致。
WAV 儲存精度引起的差異容許相對誤差 1e-5、絕對誤差 1e-10。
平均值另外由原始逐段結果重算核對。

新指標由 `src/ht1b/audio_metrics.py` 實作。MR-STFT 是頻譜收斂誤差與自然對數振幅 L1
之和，跨三種解析度等權平均；periodic Hann、置中反射 padding、未正規化 FFT，平方振幅
下限 1e-8。RMS 使用 480 樣本不重疊視窗，dBFS 下限 -100，末格用剩餘樣本；絕對 dB
差按樣本數加權。兩者不做增益匹配。這是明確定義的本機評估規格，不能直接拿數字和
論文表格相比。完整設定與逐解析度分數輸出於 `output/cl1b-comparison.metrics.json`。

手動區段統計保存每格的誤差和與目標能量，格點對齊後的 MSE／MAE／ESR 相當於直接
計算該區間原始樣本。包絡誤差是輔助指標，不等於 RMS 圖之外的 Attack／Release 時間估計。

修改介面請編輯本目錄的 `index.html`、`style.css`、`app.js`、`analysis.js`、
`full-test.js`、`architecture.js`，再重建輸出。
`index.html` 是模板，須經產生器填入資料與樣式後才能使用。

驗證：`python -m pytest tests/test_audio_metrics.py -q`（包含選用的 Torch STFT 對照），
以及 `node pinn/comparison-ui/check_logic.cjs`（需先產生 metrics JSON）。後者驗證區段統計、
盲測狀態與儲存失敗處理，不是瀏覽器視覺或音訊播放測試。
