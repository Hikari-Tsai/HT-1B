# HT-1B：簡化電路、純演算法與 PINN 物理方程

整理日期：2026-09-29
對應實作：HT-1B 專案目前的簡化電路模型與 Manual 模式實驗。

> 本文描述程式實際使用的灰箱等效模型，不是完整 CL-1B 維修電路圖，也不是可直接製作硬體的電路設計。文中的數值以「純演算法」未經資料擬合的初始設定為準，不代表各個訓練後模型的參數。

## 1. 我們要解決什麼問題？

給定未壓縮的 Dry 音訊與旋鈕設定，求出壓縮器內部狀態隨時間的變化，再計算對應的 Wet 輸出。

純演算法與 PINN 共用同一套簡化物理方程，差別在求解方式與參數取得方式。

| 方法 | 狀態與參數如何取得 | 輸出方式 |
|---|---|---|
| 純演算法 | 使用固定的初始電路參數，逐樣本求解狀態 | 由電路輸出公式計算 Wet |
| MLP／GRU＋PINN | 以 Dry／Wet 與物理殘差學習狀態、辨識部分電路參數 | 目前比較頁使用學得的參數，再交給數值求解器輸出 |
| S6＋TFiLM＋PINN | 網路預測三個物理狀態，同時學習部分電路參數 | 直接將網路預測的狀態帶入電路公式輸出 |

這是一組只有時間變化的**非線性常微分方程（ODE）**，沒有空間微分。程式中的類別雖然叫 `CircuitPDE`，但這個電路問題實際上是 ODE。

## 2. 簡化等效電路圖

![HT-1B 目前程式使用的簡化等效電路，包含音訊路徑、側鏈控制與三個物理狀態](./cl1b-reduced-circuit.png)

[開啟 SVG 向量版本](./cl1b-reduced-circuit.svg)

- **上方音訊路徑**：Dry 經電阻網路、GRE 光控衰減與輸出放大，產生 Wet。
- **下方側鏈路徑**：偵測音訊強度，經 Attack／Release 形成電容電壓，再驅動光學快、慢狀態。
- **綠色實線**：音訊／電氣連接。
- **藍色虛線**：偵測或控制關係，不是實際導線。
- **GRE**：增益衰減元件，在模型中以受內部狀態控制的導電度表示。

GRE 導電度增加時，節點向地的等效負載增加，音訊衰減加強。它同時會改變前端節點與偵測訊號，因此音訊與側鏈控制是互相耦合的。

## 3. 符號與三個未知狀態

| 符號 | 意義 | 單位／範圍 |
|---|---|---|
| $x(t)$ | Dry 音訊樣本值 | 數位音訊振幅 |
| $u(t)$ | 模型的輸入等效電壓 | V |
| $a(t),b(t)$ | 前端電阻網路的兩個節點電壓 | V |
| $y(t)$ | 模型預測的 Wet 樣本值 | 數位音訊振幅 |
| $e(t)$ | C3 電容電壓，代表包絡控制狀態 | V |
| $f(t)$ | 光學元件的快反應狀態 | 無因次，0～1 |
| $s(t)$ | 光學元件的慢反應狀態 | 無因次，0～1 |
| $d(t)$ | 整流與 Threshold 處理後的偵測目標 | V |
| $q(t)$ | 光學狀態的目標值 | 無因次，0～1 |
| $i_A(t),i_R(t)$ | C3 的充電、放電電流 | A |
| $G_{\mathrm{GRE}}$ | 光學增益衰減元件的等效導電度 | S |
| $M$ | Make-up Gain | dB |
| $p_{\mathrm{threshold}},p_{\mathrm{ratio}},p_{\mathrm{attack}},p_{\mathrm{release}}$ | 電路模型使用的正規化電位器位置 | 0～1 |

注意：電路模型的電位器位置，是資料標籤映射後的近似物理控制值；不能直接把「Ratio = 4:1」的數字 4 代入 $p_{\mathrm{ratio}}$。S6 網路的條件編碼與電路的電位器編碼也不是同一組數值。

## 4. 核心物理方程

我們要求解的三個狀態是：

$$
\mathbf z(t)=\begin{bmatrix}e(t)&f(t)&s(t)\end{bmatrix}^{\mathsf T}
$$

核心動態方程為：

$$
\boxed{
\begin{aligned}
C_3\frac{de}{dt}&=i_A-i_R\\
\tau_f\frac{df}{dt}&=q-f\\
\tau_s\frac{ds}{dt}&=q-s
\end{aligned}}
$$

三條方程分別表示：

1. **電容電流平衡**：流入與流出的電流差，決定 C3 電壓變化速度。
2. **快光學響應**：狀態 $f$ 以時間常數 $\tau_f$ 接近目標 $q$。
3. **慢光學響應**：狀態 $s$ 以時間常數 $\tau_s$ 接近目標 $q$。

第一條依據電容關係與支路電流平衡；後兩條是目前採用的光學元件雙狀態等效模型，不是已知完整內部電路的逐元件方程。

## 5. 純演算法：從 Dry 到前端節點

### 5.1 輸入刻度

初始模型將 Dry 樣本換成等效電壓：

$$
u=10x
$$

$10\ \mathrm{V/FS}$ 是待校準的模型假設，不代表錄音介面的實際電壓刻度。

### 5.2 光學導電度

一般形式為：

$$
G_{\mathrm{GRE}}=G_{\mathrm{dark}}+G_{\mathrm{span}}\bigl[\mu f+(1-\mu)s\bigr]
$$

純演算法初始值代入後：

$$
\boxed{G_{\mathrm{GRE}}=10^{-8}+0.001\,[0.8f+0.2s]}
$$

其中 $G_{\mathrm{dark}}$ 與 $G_{\mathrm{span}}$ 的單位都是 S。導電度越大，等效電阻越小。

### 5.3 電阻節點方程

定義總負載導電度：

$$
G_L=\frac{1}{R_g}+G_{\mathrm{GRE}}
$$

固定電阻與 Ratio 支路為：

$$
R_s=R_t=R_g=100\,\mathrm{k\Omega},\qquad
R_\rho=10\,\mathrm{k\Omega}\times p_{\mathrm{ratio}}
$$

前端電阻網路消元後，可以寫成：

$$
\frac{a-u}{R_s}+\frac{a}{R_t}+bG_L=0,
\qquad a-b-R_\rho bG_L=0
$$

解出兩個節點：

$$
\boxed{
a=\frac{u/R_s}
{\frac{1}{R_s}+\frac{1}{R_t}+\frac{G_L}{1+R_\rho G_L}},
\qquad
b=\frac{a}{1+R_\rho G_L}
}
$$

這些代數式已包含電阻網路的 KCL 關係；當 $R_\rho=0$ 時仍有效，不會除以零。

## 6. 側鏈偵測與 C3 充放電

### 6.1 旋鈕曲線與整流偵測

模型假設對數電位器曲線為：

$$
\phi(p)=\frac{10^{2p}-1}{10^2-1},\qquad p\in[0,1]
$$

偵測節點 $a$ 的振幅，經 Threshold 控制後得到：

$$
d=\max\left(2|a|\,\phi(p_{\mathrm{threshold}})-0.03,\ 0\right)
$$

這裡的 2 倍增益與 0.03 V 偏置屬於簡化側鏈設定；沒有模擬整流器的完整高頻或換向行為。

### 6.2 側鏈運算放大器

比較目標 $d$ 與目前電容電壓 $e$：

$$
v_o=\operatorname{clip}\left(1000(d-e),-13.5,13.5\right)
$$

`clip` 表示將輸出限制在 $\pm13.5$ V。這是有限增益、有限擺幅且沒有內部動態的近似。

### 6.3 Attack／Release 電阻

以下阻值均以 Ω 表示：

$$
\begin{aligned}
R_A&=274+500000\,\phi(p_{\mathrm{attack}})\\
R_R&=47500+500000\,p_{\mathrm{release}}\\
R_B&=274000+235000=509000
\end{aligned}
$$

$R_A$ 是 Attack 充電支路；$R_R$ 是 Release 支路；$R_B$ 包含偏壓電阻與假定的 235 kΩ trim。$R_R$ 與前面的 Ratio 電阻 $R_\rho$ 是不同元件。

### 6.4 Release 的戴維寧等效

Release 並非只有一顆電阻直接接地；模型保留 $v_o$ 與 +15 V 偏壓的影響：

$$
V_{\mathrm{th}}=\frac{v_oR_B+15R_R}{R_R+R_B},
\qquad
R_{\mathrm{th}}=\frac{R_RR_B}{R_R+R_B}
$$

### 6.5 充電、放電電流

二極體採固定 0.55 V 壓降的分段線性模型：

$$
\boxed{
i_A=\frac{\max(v_o-e-0.55,0)}{R_A},
\qquad
i_R=\frac{\max(e-V_{\mathrm{th}}-0.55,0)}{R_{\mathrm{th}}}
}
$$

將電流代入 $C_3=10\,\mu\mathrm F$ 的電容方程：

$$
\frac{de}{dt}=\frac{i_A-i_R}{10\times10^{-6}}
$$

充電電流大於放電電流時，$e$ 上升；反之下降。

## 7. 光學元件的快、慢動態

### 7.1 Manual 模式的控制目標

Manual 模式取 $\max(e,0)$ 為控制電壓。R17、控制電位器與 trim 的初始等效係數為：

$$
\alpha=\frac{100\,\mathrm{k\Omega}}{20\,\mathrm{k\Omega}+100\,\mathrm{k\Omega}}\times0.5=\frac{5}{12}
$$

因此：

$$
v_{\mathrm{drive}}=\frac{5}{12}\max(e,0),
\qquad
\boxed{q=\frac{v_{\mathrm{drive}}}{v_{\mathrm{drive}}+0.12}}
$$

$v_{\mathrm{drive}}$ 是等效驅動電壓，不是已驗證的 LED 電流。0.12 V 對應初始 `gre_half`。

### 7.2 上升與下降的時間常數

純演算法基準使用：

$$
\tau_f=
\begin{cases}
1\,\mathrm{ms},&q>f\\
50\,\mathrm{ms},&q\le f
\end{cases}
\qquad
\tau_s=
\begin{cases}
10\,\mathrm{ms},&q>s\\
500\,\mathrm{ms},&q\le s
\end{cases}
$$

相應狀態更新速率為：

$$
\frac{df}{dt}=\frac{q-f}{\tau_f},
\qquad
\frac{ds}{dt}=\frac{q-s}{\tau_s}
$$

這四個時間常數描述 GRE 等效模型的內在快、慢動態，與前面 Attack／Release 旋鈕控制的 C3 支路不是同一組參數。它們皆可在 PINN 訓練中被辨識調整。

## 8. 輸出音訊公式

輸出級的一般形式為：

$$
y=\frac{H}{V_{\mathrm{out/FS}}}
\tanh\left(\frac{A\,10^{M/20}\,b}{H}\right)
$$

初始值是 $A=2.05$、$H=24$ V、$V_{\mathrm{out/FS}}=10$ V/FS，因此：

$$
\boxed{
y=\frac{24}{10}
\tanh\left(\frac{2.05\cdot10^{M/20}\cdot b}{24}\right)
}
$$

這是有限擺幅的無記憶放大器近似。`tanh` 不代表已還原 ECC83／ECC82 真空管的實際音色或電路行為。

## 9. 純演算法的數值求解

### 9.1 為什麼不能只照順序算一次？

光學狀態 $f,s$ 決定導電度，導電度影響 $a$，$a$ 決定偵測目標 $d$，$d$ 再影響 $e$ 與新的 $f,s$。因此三個狀態需要在同一時間步內一致求解。

把所有代數關係代入後，整個系統可寫成：

$$
\dot{\mathbf z}=F(\mathbf z,u;\boldsymbol\theta,\mathbf p)
$$

$\boldsymbol\theta$ 代表電路參數，$\mathbf p$ 代表旋鈕設定。

### 9.2 後向 Euler

目前共同比較採 48 kHz、每個音訊樣本一個步長：

$$
\Delta t=\frac{1}{48000}\ \mathrm s
$$

每步求解：

$$
\boxed{
\mathbf z_n-\mathbf z_{n-1}
-\Delta t\,F(\mathbf z_n,u_n)=0
}
$$

右側使用本步新狀態 $\mathbf z_n$，所以這是隱式方程，需要求根；不是直接把舊狀態代入一次的前向 Euler。

### 9.3 目前加速求解器

給定候選電容電壓 $e_n$，可以先算出 $q_n$，再消去快、慢狀態：

$$
f_n=\frac{f_{n-1}+(\Delta t/\tau_f)q_n}{1+\Delta t/\tau_f},
\qquad
s_n=\frac{s_{n-1}+(\Delta t/\tau_s)q_n}{1+\Delta t/\tau_s}
$$

在這個隱式更新中，新狀態位於舊狀態與 $q_n$ 之間，因此程式可用 $q_n$ 與舊狀態的大小關係選擇 attack／release 時間常數。

剩下的純量方程為：

$$
h(e_n)=e_n-e_{n-1}
-\frac{\Delta t}{C_3}\bigl[i_A(e_n)-i_R(e_n)\bigr]=0
$$

目前 `FastCircuitSolver` 使用帶區間保護的解析 Newton 法；Newton 候選點超出區間時，改採二分步驟。這是與三狀態耦合後向 Euler 等價的消元加速方式，並非改成另一套物理模型。

原始 SciPy 求解器則保留三個未知量一起求根，作為參考實作。

### 9.4 初始狀態與暖機

純演算法在未提供其他初始狀態時使用：

$$
\mathbf z_0=[0,0,0]^{\mathsf T}
$$

同一段連續處理不會在每個資料批次重設狀態。但目前網頁的共同短視窗比較，是每個視窗重設，再輸入 4,096 個 Dry 樣本暖機，最後對 1,024 個樣本計分。這與從完整錄音起點持續處理的長時間評估不同。

## 10. PINN 使用的物理約束

### 10.1 將方程改寫成殘差

將三條動態方程移到等號同一側：

$$
\begin{aligned}
r_e&=\frac{R_A}{V_{\mathrm{rail}}}
\left(C_3\dot e-i_A+i_R\right)\\
r_f&=\tau_f\dot f+f-q\\
r_s&=\tau_s\dot s+s-q
\end{aligned}
$$

理想情況下三者都為零。$R_A/V_{\mathrm{rail}}$ 用來調整第一條殘差的尺度，初始 $V_{\mathrm{rail}}=13.5$ V。

物理 loss 為：

$$
\boxed{
L_{\mathrm{physics}}=
\frac{1}{3}\left(
\langle r_e^2\rangle+
\langle r_f^2\rangle+
\langle r_s^2\rangle
\right)
}
$$

$\langle\cdot\rangle$ 表示對本批計分樣本平均。

### 10.2 時間導數如何取得？

目前訓練程式顯式使用後向差分：

$$
\dot{\mathbf z}_n\approx
\frac{\mathbf z_n-\mathbf z_{n-1}}{\Delta t}
=48000(\mathbf z_n-\mathbf z_{n-1})
$$

因此 PINN 使用的是離散時間的物理殘差，與後向 Euler 的時間離散形式對應。它不是對空間座標微分，也不是以網路對連續時間的自動微分取代這個差分。

S6＋TFiLM＋PINN 會將 Dry 暖機最後預測的物理狀態接到第一個計分樣本，不在錄音中段強制把 $e,f,s$ 歸零；網路的循環隱藏狀態仍逐視窗初始化。

### 10.3 物理 loss 與音訊 loss 的分工

- 物理 loss：預測的狀態是否遵守這套簡化電路方程？
- 音訊 loss：由這些狀態算出的 Wet，是否接近真實錄音？
- 參數先驗：學得參數是否過度偏離初始設定？

MLP／GRU＋PINN 使用能量正規化的波形 MSE，加上物理 loss 與參數先驗。

本次 S6＋TFiLM＋PINN 的實際訓練目標則為：

$$
L_{\mathrm{audio}}=
L_1+0.1L_{\mathrm{ESR}}+0.1L_{\mathrm{MRSTFT}}
$$

$$
\boxed{
L_{\mathrm{total}}=
L_{\mathrm{audio}}+L_{\mathrm{physics}}
+10^{-4}L_{\mathrm{prior}}
}
$$

此處先驗是可訓練電路參數在內部 raw 參數空間相對初始值的均方偏移。不同模型的音訊 loss 不完全相同，不能直接把它們的訓練總 loss 作同尺度排名。

## 11. 哪些電路參數可以被學習？

目前 `LearnedCircuit` 對以下七個參數使用有界變換：

| 程式名稱 | 意義 | 初始值 | 訓練界限 |
|---|---|---:|---:|
| `gre_span` | GRE 導電度變化幅度 | 0.001 S | 0.000001～0.02 S |
| `gre_half` | 驅動電壓到目標狀態的半飽和值 | 0.12 V | 0.001～3 V |
| `gre_attack_fast` | 快狀態上升時間常數 | 0.001 s | 0.0001～0.05 s |
| `gre_attack_slow` | 慢狀態上升時間常數 | 0.01 s | 0.001～0.5 s |
| `gre_release_fast` | 快狀態下降時間常數 | 0.05 s | 0.002～2 s |
| `gre_release_slow` | 慢狀態下降時間常數 | 0.5 s | 0.02～20 s |
| `amp_gain` | 輸出放大器增益 | 2.05 | 0.25～12 |

純演算法使用初始值；訓練後模型應以自己的 checkpoint 或匯出設定為準。

## 12. 模型與評估的限制

1. 電阻網路與 C3 支路有電路關係作依據，但 GRE 雙時間尺度、導電度曲線及部分校準值屬於等效假設。
2. 真空管偏壓、完整放大級、輸入／輸出變壓器頻率響應、磁飽和、磁滯、噪聲與供電變動沒有完整實作。
3. C8 的快速動態採準靜態處理；其約 16.7 μs 極點未保留為狀態。
4. 側鏈二極體採固定壓降模型；運算放大器沒有 slew rate／頻寬狀態。
5. 物理殘差小，只表示較符合目前這套模型；不保證真實 Wet 誤差也一定較小。
6. 本次 S6＋TFiLM＋PINN 同時更改輸出 head 與訓練目標，不是只增加物理 loss 的單變因消融。
7. 共同比較的 85.33 ms 暖機與 21.33 ms 計分視窗不足以驗證長 Release 行為；驗證集已重用，不能稱為新的獨立測試。

## 13. 實作依據與檔案

| 內容 | 檔案 |
|---|---|
| 共用電路代數式與動態方程 | [equations.py](../src/ht1b/equations.py) |
| 元件與控制參數定義 | [config.py](../src/ht1b/config.py) |
| 純演算法初始參數 | [initial-config.json](../runs/full-corpus-gpu-20260925/initial-config.json) |
| 目前的加速數值求解器 | [fast_solver.py](../src/ht1b/fast_solver.py) |
| PhysicsNeMo 共用物理殘差與可學習參數 | [physicsnemo_model.py](../src/ht1b/physicsnemo_model.py) |
| S6＋TFiLM＋PINN 模型 | [pinn_model.py](../s6/pinn_model.py) |
| S6＋TFiLM＋PINN 實驗與 loss | [pinn_experiment.py](../s6/pinn_experiment.py) |
| 原始模型假設與近似說明 | [EQUATIONS.md](../docs/EQUATIONS.md) |
| 電路圖產生程式 | [draw_reduced_circuit.py](../pinn/draw_reduced_circuit.py) |
| 互動結果頁 | [cl1b-comparison.html](./cl1b-comparison.html) |

## 14. 閱讀與分享

本文使用 Markdown 表格與 LaTeX 數學區塊。公式需使用支援數學排版的 Markdown 閱讀器。

分享電路圖時，請將以下檔案放在同一資料夾，保留相對路徑：

```text
cl1b-circuit-and-pinn.md
cl1b-reduced-circuit.png
cl1b-reduced-circuit.svg
```

上方程式來源連結依原專案目錄排列；單獨分享這三個檔案時，圖仍可顯示，但程式來源連結需要另外提供專案。
