"use strict";

if (data.full_corpus) {
  const full = data.full_corpus;
  const fullState = {metric:"esr"};
  const fullIds = ["attack", "release", "ratio", "threshold"];
  const fullNumber = (v, metric=fullState.metric) =>
    metric === "mse" ? v.toExponential(3) : v.toLocaleString("zh-TW", {maximumFractionDigits:metric === "rms_db_mae" ? 3 : 5});
  const fullMean = values => values.reduce((sum, v) => sum + v, 0) / values.length;
  const fullValue = r => r.metrics[fullState.metric];
  const fullSpan = r => r.intervals.map(s=>`${(s.start_sample/48000).toFixed(2)}–${(s.stop_sample/48000).toFixed(2)} s`).join("；");
  const selectors = {
    attack:r=>r.labels.attack_index,
    release:r=>r.labels.release_index,
    ratio:r=>r.labels.ratio,
    threshold:r=>r.labels.threshold_db
  };
  const filtered = () => full.records.filter(r => fullIds.every(key => {
    const value = $(`full-${key}`).value;
    return value === "all" || selectors[key](r) === Number(value);
  }));
  const fullOption = (key, value) => key === "ratio" ? `${value}:1` : key === "threshold" ? `${value} dB` : value;
  for (const key of fullIds) {
    const values = [...new Set(full.records.map(selectors[key]))].sort((a,b)=>a-b);
    $(`full-${key}`).innerHTML += values.map(value => `<option value="${value}">${fullOption(key,value)}</option>`).join("");
  }
  $("independent-test").hidden = false;
  const selectedName = full.selected === "fitted" ? "PhysicsNeMo 擬合參數" : "未訓練電路基準";
  $("full-test-selection").textContent = `驗證 ESR 最低：${selectedName}`;
  $("full-test-validation").textContent = `基準 ${full.validation.baseline.esr.toFixed(5)} · 擬合 ${full.validation.fitted.esr.toFixed(5)}。選擇鎖定後，測試資料才開封。`;
  $("full-test-facts").innerHTML = [
    ["訓練樣本", (full.train_samples / 1e9).toFixed(3) + "B", `${full.train_records} 個檔案 · 完整覆蓋`],
    ["測試錄音", full.records.length.toLocaleString("zh-TW"), `${(full.samples_by_split.test / 48000 / 3600).toFixed(2)} 配對音訊小時`],
    ["測試 ESR", full.test.esr.toFixed(5), `${selectedName} · 逐檔等權平均`]
  ].map(([label,value,note]) => `<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join("");

  function renderFull() {
    const rows = filtered();
    const metric = fullState.metric;
    const metricName = definitions[metric].name;
    document.querySelectorAll("[data-full-metric]").forEach(button => button.setAttribute("aria-pressed", button.dataset.fullMetric === metric));
    $("full-test-column").textContent = `${metricName} ↓`;
    $("full-test-value").textContent = rows.length ? fullNumber(fullMean(rows.map(fullValue))) : "—";
    $("full-test-count").textContent = `${rows.length} / ${full.records.length} 檔 · ${definitions[metric].help}`;
    const grouped = [...new Set(rows.map(r=>r.labels.ratio))].sort((a,b)=>a-b).map(ratio => {
      const group = rows.filter(r=>r.labels.ratio===ratio);
      return {ratio, count:group.length, value:fullMean(group.map(fullValue))};
    });
    const max = Math.max(0,...grouped.map(g=>g.value));
    $("full-test-ratio-chart").innerHTML = grouped.length ? grouped.map(g=>
      `<div class="full-ratio-row"><span>${g.ratio}:1 <small>n=${g.count}</small></span><span class="full-ratio-track"><span style="width:${max ? g.value/max*100 : 0}%"></span></span><b>${fullNumber(g.value)}</b></div>`
    ).join("") : `<p class="muted">沒有符合條件的檔案。</p>`;
    const sorted = [...rows].sort((a,b)=>fullValue(b)-fullValue(a));
    $("full-test-caption").textContent = `${metricName} · ${rows.length} 檔符合條件 · 依誤差由高至低排列`;
    $("full-test-rows").innerHTML = sorted.slice(0,25).map(r =>
      `<tr><td><strong>${r.id}</strong><small>A${r.labels.attack_index} · R${r.labels.release_index} · ${r.labels.ratio}:1 · ${r.labels.threshold_db} dB</small></td><td>${fullNumber(fullValue(r))}</td><td>${fullSpan(r)}</td><td>${(r.samples/48000).toFixed(2)} s</td></tr>`
    ).join("") || `<tr><td colspan="4">沒有符合條件的檔案。</td></tr>`;
  }
  document.querySelectorAll("[data-full-metric]").forEach(button => button.addEventListener("click", () => {
    fullState.metric = button.dataset.fullMetric;
    renderFull();
  }));
  fullIds.forEach(key => $(`full-${key}`).addEventListener("change", renderFull));
  $("full-test-csv").addEventListener("click", () => {
    const header = ["record","attack","release","ratio","threshold_db","start_s","stop_s","samples","esr","mse","mae","mrstft","rms_db_mae"];
    const rows = filtered().flatMap(r => r.intervals.map(s => [r.id,r.labels.attack_index,r.labels.release_index,r.labels.ratio,r.labels.threshold_db,s.start_sample/48000,s.stop_sample/48000,s.stop_sample-s.start_sample,...["esr","mse","mae","mrstft","rms_db_mae"].map(key=>s.metrics[key])]));
    download("\uFEFF" + [header,...rows].map(row=>row.join(",")).join("\r\n"),"cl1b-full-corpus-heldout-test.csv","text/csv;charset=utf-8");
  });
  renderFull();
}
