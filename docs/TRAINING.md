# PhysicsNeMo 音檔訓練

## 資料格式

推薦使用 `ht1b prepare` 處理指定的五檔位資料集。自訂 manifest：

```json
{
  "schema_version": 1,
  "records": [
    {
      "id": "take-001",
      "stereo_pair": "take-001.wav",
      "split": "train",
      "controls": {
        "threshold": 0.7,
        "ratio": 0.5,
        "attack": 0.15,
        "release": 0.3,
        "makeup_db": 0,
        "mode": "manual"
      },
      "delay_samples": 0,
      "initial_state": [0, 0, 0]
    }
  ]
}
```

這些 `controls` 是**模型的正規化控制值**；不要把 ratio 10、attack 4 直接放進去。資料集原始標籤應交由 `prepare` 解析，原標籤會保存在 metadata。

也可用 `input: "dry.wav"`、`output: "wet.wav"` 代替 `stereo_pair`；兩檔都須是 mono、等長、相同取樣率。路徑相對於 manifest，或使用絕對路徑。程式拒絕空檔、非有限樣本、聲道／取樣率／長度不符，沒有靜默混音或重取樣。

只有乾聲時可省略 output，但必須明確設定 `--data-weight 0`。這只能解既定方程式，沒有實機資料就不能辨識硬體特性。

## 延遲與初始狀態

- 正 delay_samples：wet 落後 dry，對齊為 dry[:-delay] 與 wet[delay:]。
- 負 delay_samples：會裁掉 dry 的歷史；必須明確提供新起點的 initial_state。
- start_sample > 0 也需要 initial_state，不能把有壓縮記憶的片段默認為零。
- initial_state = [C3 電壓 V, GRE fast occupancy, GRE slow occupancy]。
- `prepare --seconds N` 永遠從原檔起點取前 N 秒，不是抽取中間片段。
- 開頭假設已完全恢復；若錄音未保留恢復／預熱時間，需調整初始狀態。
- 目前只有整數取樣延遲；亞取樣延遲、極性反轉、時鐘漂移需另外處理。
- 不獨立正規化左右聲道。A/D、D/A 的電壓校準由 config 的兩個 volts_per_fs 決定。

## 訓練與部署是兩個步驟

```bash
ht1b train data/manifest.json --output runs/fit --steps 2000 --device cpu
ht1b export runs/fit/last.pt configs/identified.json
ht1b evaluate data/manifest.json --config configs/identified.json \
  --split validation --output runs/fit/validation.json
```

程式使用 float64，支援 CPU／CUDA；MPS 不支援這裡要求的 float64，因此不選用。

- last.pt：網路、物理參數、optimizer、隨機狀態與資料簽章。
- history.jsonl：每一步的 loss、physics、data。
- fitted.json：可直接交給數值求解器的參數。
- report.json：固定抽樣點的前後 loss、錄音重建數字、資料假設。

續訓檢查音訊／旋鈕／初始狀態／config 簽章及訓練選項，避免誤接不同資料。

座標 PINN 的時間表達能力有限，長音檔含高速變化時可能欠擬合。首輪使用從起點取的短片段檢查損失與對齊；更長資料可再研究多重射擊／連續狀態的分段 PINN，不能任意重置每段狀態。`--steps` 只是優化步數，不保證收斂或唯一辨識。

## 驗證

在 manifest 將**不同錄音**標記為 `validation` 或 `test`。訓練只使用 train；evaluate 用匯出的方程式參數從頭求解指定 split。

同一素材僅旋鈕不同的錄音，仍可能有內容重複。要驗證真正泛化，需要同時保留不同素材與未見參數組合。單一 10:1／threshold 0 的檔案無法可靠辨識全部時間常數、閾值與 ratio 映射；參數也可能相互補償。

- MSE：平均平方波形誤差。
- MAE：平均絕對波形誤差。
- ESR：誤差能量／參考輸出能量（分母加下限保護靜音）。
- trajectory_reconstruction：神經網路對訓練軌跡的重建，僅抽樣最多 4096 點。
- evaluate 報告：真正部署的數值求解器，逐樣本計算整個選定區間。

不要把 ESR 換算成「還原度百分比」，也不能用訓練軌跡的誤差代替直接求解器的測試誤差。

## 來源

使用官方 [PhysicsNeMo 2.x 安裝方式](https://docs.nvidia.com/physicsnemo/26.08/getting-started/installation.html) 與 [v2.2.2 PDE API](https://github.com/NVIDIA/physicsnemo/blob/v2.2.2/physicsnemo/sym/eq/pde.py)。本機已實際安裝並執行 `nvidia-physicsnemo[sym]==2.2.2`，不是僅以 PyTorch 程式冒充框架。
