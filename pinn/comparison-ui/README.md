# HT-1B 六模型共同驗證頁

主頁比較六個模型（MLP＋PINN、GRU＋PINN、S4＋TFiLM、S6＋TFiLM、S6＋TFiLM＋PINN、RiccardoVib），另列純演算法基準。資料取自 `runs/s6-tfilm-pinn-gpu-20260929/common-validation/`，共 620 檔、1,240 視窗。這是探索性重用驗證，沒有新的獨立測試。

## 重建與檢查

在專案根目錄、已安裝專案相依套件的 Python 環境執行：

```text
python -m pinn.build_comparison_page
node pinn/comparison-ui/check_logic.cjs
```

輸出 `output/cl1b-comparison.html` 與 `.metrics.json`。產生器核對資料與視窗簽章、七種方法來源檔案 SHA-256、逐檔平均、pooled ESR、訓練紀錄與歷史測試覆蓋。只重建報告，不啟動訓練或重新推論。

## 介面與資料範圍

- 總表同時顯示五項指標；依選定指標排序六模型，純演算法固定列於基準列。
- Attack／Release／Ratio／Threshold 篩選、逐檔等權與合併樣本彙整；總表及逐檔 CSV 跟隨篩選。
- 逐檔資料採 20 筆分頁。訓練區塊可切換 MLP、GRU、S4、S6、S6＋PINN；MLP／GRU 顯示單批 loss 對步數，其他三模型顯示 16 輪驗證 ESR。
- MLP 的紀錄來自 `training/history.jsonl`（每 50 步），GRU 由 stdout 狀態快照還原（約每分鐘）；兩者只完成一輪，採用最終參數，沒有逐輪最佳 checkpoint。紀錄可分頁與完整匯出 CSV，不補造缺少的步數或 loss。
- 方法表區分共同短視窗驗證與歷史完整區段測試。未執行的測試明確標示，不補造分數。
- 六模型目前沒有新的獨立測試；共同評分位置不代表相同訓練預算。
- 不含舊的局部模型音軌、盲測、重複排名；不改動既有瀏覽器盲測儲存資料。
- 無外部 JS、CSS、CDN 或伺服器需求。電路說明連結需要同目錄的 Markdown 與 SVG。

編輯 `unified/index.html`、`unified/style.css`、`unified/core.js`、`unified/app.js`，再重建。`pinn/build_unified_comparison.py` 負責整併報告；原入口保留 `five_model_results()` 的資料檢核。

`check_logic.cjs` 核對來源一致性、70 組彙整值、空篩選、CSV／JSON、分頁、訓練紀錄、主題切換和頁面結構。它是 Node 資料與狀態檢查，不是瀏覽器視覺測試。

## 歷史版本

整理前的完整 HTML 保存在 `output/archive/cl1b-comparison-before-cleanup-20260929.html`。原始實驗資料仍在 `runs/`，舊模板及 JS 仍保留於本目錄根層，`build_legacy()` 保留舊版產生方式；舊說明在 `README-legacy.md`。主頁預設只生成新版共同評估，不再附加歷史局部排名。
