# 電路方程式與近似邊界

模型依據本地 `docs/references/CL1B-service-manual.pdf` 的 1993 Rev 1.0 圖面，第 10–11、14 頁。不同版本硬體可能有差異。

- [維修手冊來源](https://funkwerkes.com/web/wp-content/techdocs/MixedProAudio/Tube-Tech-CL1B-Compressor-SM.pdf)
- [官方操作手冊](https://www.tube-tech.com/wp-content/uploads/2020/05/Tube-Tech-Manual-CL-1B-200513.pdf)

本實作為灰箱縮減模型。圖面數值、理想化元件與未知參數必須分開理解。

## 1. 前面板電阻網路

將輸入變壓器等效為理想電壓源 u（V）。u = WAV 樣本 × input_volts_per_fs，預設 10 V/FS 只是待校準假設，不代表錄音介面的實際刻度。

令 a 為 threshold 電位器上端節點、b 為 GRE／gain 電位器上端節點：

```
Rs = 100 kohm
Rt = 100 kohm
Rratio = 10 kohm × ratio_position
Rg = 100 kohm
G = 1/Rg + G_GRE

a = (u/Rs) / (1/Rs + 1/Rt + G/(1 + Rratio×G))
b = a/(1 + Rratio×G)
```

這是兩個電阻節點的 KCL 消元解。Rratio=0 時仍有效，不會除以零。圖上的 R1=68.1 kohm 只負載理想輸入源，因此在這個近似中不影響 a、b；非理想變壓器模型則需要保留它。

假設 log pot 的分壓函數：

```
phi(p) = (10^(k p)-1)/(10^k-1), k=2, p∈[0,1]
d = max(2 × abs(a) × phi(threshold) - 0.03 V, 0)
```

2 倍增益與 0.03 V 偏置來自理想化側鏈放大／整流分析；有限頻寬、精密整流器換向細節省略。threshold 使用正規化電位器位置，非直接 dBu。

## 2. C3、U2B、attack/release 支路

```
e = C3 電壓
C3 = 10 uF
vo = clip(Aol × (d-e), -Vrail, +Vrail)
RA = 274 ohm + 500 kohm × phi(attack)
RR = 47.5 kohm + 500 kohm × release
RB = 274 kohm + release_trim
Vth = (vo × RB + 15 V × RR)/(RR+RB)
Rth = RR × RB/(RR+RB)
iA = max(vo-e-Vd, 0)/RA
iR = max(e-Vth-Vd, 0)/Rth
C3 × de/dt = iA-iR
```

Vth／Rth 是 RELEASE-1 節點看出去的 Thevenin 等效，保留 +15 V 偏壓與 vo 回路。默认 Aol=1000、Vrail=13.5 V、Vd=0.55 V、release_trim=235 kohm 都是明確近似／未知校準量。

先前討論採用 Shockley 二極體的連續模型；此版本**改用固定壓降、分段線性二極體**，以得到易檢查的支路消元式，並避免指數模型在訓練初期溢位。不是聲稱兩個模型完全相同。

U2B 視為無動態、有限增益且有限擺幅的元件，沒有 slew rate／頻寬狀態。因此不是完整 LF347 transistor-level 模型。

## 3. 模式選擇與 GRE

```
Fixed:   vc = d
Manual:  vc = max(e,0)
Fix/Man: vc = max(d,e)
```

對應精密二極體路徑與模式開關的理想化選擇／OR。C8=1 nF 在此採準靜態處理：

```
drive = vc × 100k/(20k+100k) × control_trim_fraction
q = drive/(drive + gre_half)
```

也就是保留 R17／P2 的 DC 負載，略去约 16.7 us 的 C8 極點。`drive` 是驅動等效電壓，**不是已驗證的 LED 電流**。

GRE 未提供完整內部電路，採兩個 0–1 狀態 f、s：

```
tau_f = attack_fast if q>f else release_fast
tau_s = attack_slow if q>s else release_slow
df/dt = (q-f)/tau_f
ds/dt = (q-s)/tau_s
G_GRE = gre_dark + gre_span × (gre_mix × f + (1-gre_mix) × s)
```

G_GRE 回到前面板網路，使偵測訊號與增益衰減**同時耦合**。GRE 的導電度曲線與所有時間常數皆為待辨識假設。Fixed 的動態由這些狀態產生，不保證符合手冊的 1 ms／50 ms。

## 4. 音訊輸出近似

```
yV = headroom × tanh(amp_gain × 10^(makeup_db/20) × b/headroom)
yWAV = yV/output_volts_per_fs
```

這是有限擺幅的無記憶放大器替代模型，**不是 ECC83／ECC82 的方程式**。真空管偏壓、耦合電容、推挽級、輸入／輸出變壓器頻率響應、磁飽和／磁滯、噪聲、供電變動均未實作。模型目前專注壓縮控制與 GRE 動態，不能用這個 tanh 宣稱已還原真空管音色。

## 5. 直接數值求解

狀態 z=[e,f,s]，由上述支路代數消元後得到 dz/dt=F(z,u)。每步解：

```
z_new - z_old - dt F(z_new,u_new) = 0
```

使用 SciPy hybr 求根；不收斂或超界時使用有界 least_squares。三個未知量一起求解，避免用上一取樣的增益狀態代替本步回授。按方程式剛性縮放殘差，逐步檢查有限值、狀態邊界與殘差；失敗拋出 ConvergenceError，不能靜默忽略。

每個輸入取樣在子步期間保持不變（ZOH）。輸出對應更新後狀態，初始狀態在第一個樣本之前。分段 process 不重置狀態。

## 6. PhysicsNeMo

`CircuitPDE(PDE)` 使用同一個 `evaluate` 表達式生成：

```
r_C3 = (C3 e_t - iA + iR) × RA/Vrail
r_f  = tau_f f_t + f-q
r_s  = tau_s s_t + s-q
```

调用 `make_computations()` 使用 PhysicsNeMo 2.2.2 將 SymPy 式編譯成可微分 Torch 運算。時間導數顯式提供 `(z(t)-z(t-dt))/dt`；這是與直接求解器相同的 backward-Euler 離散殘差，不是偷換為某個空間導數。這裡不需要 PhysicsInformer 的空間微分工具。

網路輸入是整段錄音時間／Fourier 特徵／錄音識別，輸出有界狀態，硬性滿足錄音起點的初始狀態。目標為物理殘差 + 輸出波形 MSE／參考能量 + 小幅參數先驗。gre_span、gre_half、四個 GRE 時間常數、amp_gain 以有界變換訓練。

訓練批次抽樣 t 與 t-dt 都屬於同一條完整軌跡，不在 batch 邊界重置狀態。這是離散殘差的 inverse PINN，**不是** 已學會任意音檔映射的 S4／RNN。
