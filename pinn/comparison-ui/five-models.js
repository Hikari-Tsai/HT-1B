"use strict";

if (data.five_models) {
  const study = data.five_models;
  const methods = [
    {key:"baseline", name:"純演算法", color:"baseline", detail:"原始電路參數 · 未經資料擬合"},
    {key:"mlp", name:"MLP＋PINN", color:"ours", detail:"學習電路參數 · 完整訓練區間"},
    {key:"gru", name:"GRU＋PINN", color:"gru", detail:"學習電路參數 · 完整訓練區間"},
    {key:"s4", name:"S4＋TFiLM", color:"s4-sequence", detail:`代表視窗 · 第 ${study.sources.s4.selected_epoch} 輪`},
    {key:"s6", name:"S6＋TFiLM", color:"s6-sequence", detail:`代表視窗 · 第 ${study.sources.s6.selected_epoch} 輪`},
    {key:"riccardovib", name:"RiccardoVib", color:"author", detail:"固定權重 · 原始訓練可能重疊"}
  ];
  if (study.sources.s6_pinn) {
    methods.splice(5,0,{key:"s6_pinn",name:"S6＋TFiLM＋PINN",color:"s6-pinn",
      detail:`共用電路約束 · 第 ${study.sources.s6_pinn.selected_epoch} 輪`});
    $("five-sort").innerHTML += '<option value="s6_pinn">S6＋TFiLM＋PINN</option>';
    $("five-cards").setAttribute("class","five-cards has-pinn");
  }
  const metrics = ["esr", "mse", "mae", "mrstft", "rms_db_mae"];
  const filterFields = {attack:"attack_index", release:"release_index", ratio:"ratio", threshold:"threshold_db"};
  const view = {metric:"esr", page:0};
  const pageSize = 20;
  const n = (value, metric=view.metric) => !Number.isFinite(value) ? "—" : metric === "mse" || (value !== 0 && Math.abs(value) < .0001)
    ? value.toExponential(3) : value.toLocaleString("zh-TW", {minimumFractionDigits:metric === "mae" ? 6 : 3, maximumFractionDigits:metric === "mae" ? 6 : 5});
  const mean = values => values.length ? values.reduce((sum,v) => sum+v, 0)/values.length : null;
  $("five-models").hidden = false;
  for (const [filter, field] of Object.entries(filterFields)) {
    const values = [...new Set(study.records.map(r => r.labels[field]))].sort((a,b) => a-b);
    $("five-"+filter).innerHTML += values.map(v => `<option value="${v}">${v}${filter === "ratio" ? ":1" : filter === "threshold" ? " dB" : ""}</option>`).join("");
  }
  function rows() {
    return study.records.filter(row => Object.entries(filterFields).every(([filter, field]) => {
      const chosen = $("five-"+filter).value;
      return chosen === "all" || row.labels[field] === Number(chosen);
    }));
  }
  function aggregate(records, method, metric) {
    if (!records.length) return null;
    if ($("five-aggregate").value === "macro") return mean(records.map(r => r.metrics[method][metric]));
    const count = records.reduce((sum,r) => sum+r.samples,0);
    if (metric === "esr") {
      const squared = records.reduce((sum,r) => sum+r.metrics[method].mse*r.samples,0);
      const energy = records.reduce((sum,r) => sum+r.target_energy,0);
      return squared / Math.max(energy, count*study.protocol.esr_floor);
    }
    return records.reduce((sum,r) => sum+r.metrics[method][metric]*r.samples,0)/count;
  }
  function renderFive() {
    const selected = rows();
    const metric = view.metric;
    const summary = Object.fromEntries(methods.map(m => [m.key, Object.fromEntries(metrics.map(k => [k, aggregate(selected,m.key,k)]))]));
    const values = methods.map(m => summary[m.key][metric]);
    const maximum = selected.length ? Math.max(...values) : 0;
    const minimum = selected.length ? Math.min(...values) : null;
    const mode = $("five-aggregate").value === "macro" ? "逐檔等權平均" : "合併樣本計分";
    document.querySelectorAll("[data-five-metric]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.fiveMetric === metric)));
    $("five-count").textContent = `${selected.length}／620 檔 · ${selected.length*2} 個共同視窗`;
    const help = {esr:"ESR 對低能量素材較敏感；逐檔平均與合併樣本的排序可能不同。",
      mse:"原始振幅的平方誤差；較大音量的誤差影響較大。", mae:"原始振幅的平均絕對誤差。",
      mrstft:"短視窗 MR-STFT：FFT 256／512／1,024；與歷史長片段使用的解析度不同。",
      rms_db_mae:"10 ms RMS 包絡的 dB 誤差；每個 21.33 ms 視窗包含兩個完整格點及一個尾段。"};
    $("five-help").textContent = `${mode} · ${help[metric]}`;
    $("five-cards").innerHTML = methods.map(m => `<article class="five-card" style="--method:var(--${m.color})"><div class="five-card-title"><span class="five-dot"></span><h3>${m.name}</h3></div><strong>${n(summary[m.key][metric])}</strong><div class="five-unit">${definitions[metric].name} ↓${selected.length && summary[m.key][metric] === minimum ? " · 此指標最低" : ""}</div><p>${m.detail}</p></article>`).join("");
    $("five-bars").innerHTML = methods.map(m => `<div class="five-bar-row" style="--method:var(--${m.color})"><span>${m.name}</span><div class="five-bar-track"><i style="width:${maximum ? summary[m.key][metric]/maximum*100 : 0}%"></i></div><b>${n(summary[m.key][metric])}</b></div>`).join("");
    const winners = selected.length ? methods.filter(m => summary[m.key][metric] === minimum).map(m => m.name).join("、") : "";
    $("five-insight").textContent = selected.length ? `${winners} 在此篩選範圍的${definitions[metric].name}最低。這是相同短視窗上的輸出誤差比較；訓練預算與原始素材重疊情況不同。` : "這組旋鈕篩選沒有檔案，請調整條件或重設篩選。";
    $("five-matrix").innerHTML = `<table><caption class="sr-only">${methods.length} 種方法、五項指標，${mode}</caption><thead><tr><th scope="col">方法</th>${metrics.map(k => `<th scope="col">${definitions[k].name} ↓</th>`).join("")}</tr></thead><tbody>${methods.map(m => `<tr><th scope="row">${m.name}</th>${metrics.map(k => {
      const value = summary[m.key][k];
      const best = selected.length && value === Math.min(...methods.map(other => summary[other.key][k]));
      return `<td${best ? ' class="five-best"' : ""}>${n(value,k)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
    const sort = $("five-sort").value;
    const sorted = [...selected].sort((a,b) => b.metrics[sort][metric]-a.metrics[sort][metric] || a.id.localeCompare(b.id));
    const pages = Math.max(1,Math.ceil(sorted.length/pageSize));
    view.page = Math.min(view.page,pages-1);
    const pageRows = sorted.slice(view.page*pageSize,(view.page+1)*pageSize);
    $("five-records").innerHTML = `<table><caption>${definitions[metric].name} · 每檔兩個相同驗證視窗</caption><thead><tr><th scope="col">錄音 / 旋鈕</th>${methods.map(m => `<th scope="col">${m.name}</th>`).join("")}</tr></thead><tbody>${pageRows.length ? pageRows.map(r => {
      const best = Math.min(...methods.map(m => r.metrics[m.key][metric]));
      return `<tr><th scope="row"><span>${r.id}</span><small>A${r.labels.attack_index} · R${r.labels.release_index} · ${r.labels.ratio}:1 · ${r.labels.threshold_db} dB</small></th>${methods.map(m => `<td${r.metrics[m.key][metric] === best ? ' class="five-best"' : ""}>${n(r.metrics[m.key][metric])}</td>`).join("")}</tr>`;
    }).join("") : `<tr><td colspan="${methods.length+1}">沒有符合條件的錄音。</td></tr>`}</tbody></table>`;
    $("five-page").textContent = selected.length ? `第 ${view.page+1}／${pages} 頁 · ${view.page*pageSize+1}–${Math.min((view.page+1)*pageSize,sorted.length)}／${sorted.length} 檔` : "0 檔";
    $("five-prev").disabled = view.page === 0;
    $("five-next").disabled = view.page === pages-1;
  }
  document.querySelectorAll("[data-five-metric]").forEach(button => button.addEventListener("click", () => {view.metric=button.dataset.fiveMetric;view.page=0;renderFive();}));
  for (const id of ["five-aggregate","five-sort",...Object.keys(filterFields).map(k => "five-"+k)]) $(id).addEventListener("change", () => {view.page=0;renderFive();});
  $("five-reset").addEventListener("click", () => {Object.keys(filterFields).forEach(k => $("five-"+k).value="all");view.page=0;renderFive();});
  $("five-prev").addEventListener("click", () => {view.page=Math.max(0,view.page-1);renderFive();});
  $("five-next").addEventListener("click", () => {view.page++;renderFive();});
  $("five-csv").addEventListener("click", () => {
    const header=["scope","window_signature","record","attack","release","ratio","threshold_db","metric",...methods.map(m=>m.key)];
    const lines=rows().flatMap(r=>metrics.map(k=>["common_validation",study.window_signature,r.id,r.labels.attack_index,r.labels.release_index,r.labels.ratio,r.labels.threshold_db,k,...methods.map(m=>r.metrics[m.key][k])]));
    download("\uFEFF"+[header,...lines].map(row=>row.join(",")).join("\r\n"),`cl1b-${methods.length}-methods-common-validation.csv`,"text/csv;charset=utf-8");
  });
  renderFive();
}

// Preserve existing links into the now-collapsed experiment archive.
function revealComparisonAnchor() {
  const id = decodeURIComponent(window.location.hash.slice(1));
  const target = id && document.getElementById(id);
  if (!target) return;
  let parent = target.parentElement;
  while (parent) {if (parent.tagName === "DETAILS") parent.open = true;parent=parent.parentElement;}
}
window.addEventListener("hashchange",revealComparisonAnchor);
revealComparisonAnchor();
