"use strict";

if (data.sequence_study) {
  const study = data.sequence_study;
  const names = {s4_tfilm: "PhysicsNeMo · S4＋TFiLM", s6_tfilm: "PhysicsNeMo · S6＋TFiLM"};
  const keys = ["s4_tfilm", "s6_tfilm"];
  const labels = new Map((data.full_corpus?.records || []).map(row => [row.id, row.labels]));
  const byId = Object.fromEntries(keys.map(key => [key, new Map(study.models[key].records.map(row => [row.id, row]))]));
  const allIds = study.models.s4_tfilm.records.map(row => row.id);
  const ratios = [...new Set(allIds.map(id => labels.get(id)?.ratio))].filter(Number.isFinite).sort((a,b) => a-b);
  const metricNames = {esr:"ESR", mse:"MSE", mae:"MAE"};
  let metric = "esr";
  const mean = values => values.reduce((a,b) => a + b, 0) / values.length;
  const number = value => metric === "mse" ? value.toExponential(3) : value.toLocaleString("zh-TW", {minimumFractionDigits:3, maximumFractionDigits:5});
  document.getElementById("sequence-study").hidden = false;
  document.getElementById("seq-ratio").innerHTML += ratios.map(r => `<option value="${r}">${r}:1</option>`).join("");
  const p = study.protocol;
  const percent = (p.train_input_samples_per_epoch * p.epochs / p.train_frames_in_full_manifest * 100).toFixed(2);
  document.getElementById("sequence-scope").textContent = `兩模型在 ${p.gpu}（CUDA ${p.cuda}）上各訓練 ${p.epochs} 輪。每輪走訪 620 檔、8,680 個訓練區段各一個 ${p.window_samples.toLocaleString()} 樣本視窗，其中前 ${p.warmup_samples.toLocaleString()} 樣本暖機、後 ${p.scored_samples_per_window.toLocaleString()} 樣本計算訓練損失。累計輸入視窗約等於完整訓練樣本的 ${percent}%（不同輪次位置變動，非全樣本覆蓋）。驗證使用 620 檔、1,240 個固定視窗。`;

  function selectedIds() {
    const ratio = document.getElementById("seq-ratio").value;
    return ratio === "all" ? allIds : allIds.filter(id => labels.get(id)?.ratio === Number(ratio));
  }
  function render() {
    const ids = selectedIds();
    const values = Object.fromEntries(keys.map(key => [key, mean(ids.map(id => byId[key].get(id)[metric]))]));
    const max = Math.max(...Object.values(values));
    document.querySelectorAll("[data-seq-metric]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.seqMetric === metric)));
    document.getElementById("seq-readout").textContent = `驗證集 · ${ids.length}／620 檔 · ${metricNames[metric]} 越低越好。${ids.length === 620 ? "最佳輪次以完整驗證集逐檔等權平均 ESR 選出。" : "篩選僅改變顯示，不重新選模型。"}`;
    document.getElementById("seq-cards").innerHTML = keys.map(key => `<div class="architecture-card" style="--arch-color:var(--${key === "s4_tfilm" ? "ours" : "gru"})"><div class="architecture-card-head"><span>${names[key]}</span><small>最佳第 ${study.models[key].selected_epoch} 輪</small></div><strong>${number(values[key])}</strong><span class="architecture-card-unit">${metricNames[metric]} ↓ · 逐檔等權平均${values[key] === Math.min(...Object.values(values)) ? " · 較低" : ""}</span><div class="architecture-card-track"><span style="width:${max ? values[key] / max * 100 : 0}%"></span></div></div>`).join("");
    const historyMax = Math.max(...keys.flatMap(key => study.models[key].history.map(row => row.validation_macro_mean[metric])));
    document.getElementById("seq-history").innerHTML = `<div class="seq-history-legend"><span>S4＋TFiLM</span><span>S6＋TFiLM</span></div>` + study.models.s4_tfilm.history.map((row, i) => `<div class="seq-history-row"><span>${i+1}</span>${keys.map(key => {const v = study.models[key].history[i].validation_macro_mean[metric]; return `<div class="seq-history-track" title="${names[key]} · 第 ${i+1} 輪 · ${number(v)}"><i class="${key}" style="width:${historyMax ? Math.max(1, v / historyMax * 100) : 0}%"></i></div>`;}).join("")}</div>`).join("");
    const sorted = ids.map(id => ({id, a:byId.s4_tfilm.get(id)[metric], b:byId.s6_tfilm.get(id)[metric]})).sort((x,y) => Math.abs(y.b-y.a)-Math.abs(x.b-x.a));
    document.getElementById("seq-records").innerHTML = `<table><thead><tr><th>檔案</th><th>S4</th><th>S6</th><th>S6 − S4</th></tr></thead><tbody>${sorted.slice(0,15).map(r => `<tr><td>${r.id}</td><td>${number(r.a)}</td><td>${number(r.b)}</td><td class="${r.b < r.a ? "architecture-better" : ""}">${r.b > r.a ? "+" : ""}${number(r.b-r.a)}</td></tr>`).join("")}</tbody></table>`;
  }
  document.querySelectorAll("[data-seq-metric]").forEach(button => button.addEventListener("click", () => {metric = button.dataset.seqMetric; render();}));
  document.getElementById("seq-ratio").addEventListener("change", render);
  document.getElementById("seq-csv").addEventListener("click", () => {
    const rows = [["record","ratio","metric","s4_tfilm","s6_tfilm"], ...selectedIds().flatMap(id => ["esr","mse","mae"].map(m => [id, labels.get(id)?.ratio, m, byId.s4_tfilm.get(id)[m], byId.s6_tfilm.get(id)[m]]))];
    download("\uFEFF" + rows.map(r => r.join(",")).join("\r\n"), "cl1b-s4-s6-validation.csv", "text/csv;charset=utf-8");
  });
  render();
}
