"use strict";

if (data.architecture) {
  const architecture = data.architecture;
  const archState = {split:"validation", metric:"esr"};
  const archNames = {baseline:"純演算法", mlp:"PhysicsNeMo · MLP＋PINN", gru:"PhysicsNeMo · GRU＋PINN", riccardovib:"RiccardoVib"};
  const archShort = {baseline:"純演算法", mlp:"MLP＋PINN", gru:"GRU＋PINN", riccardovib:"RiccardoVib"};
  const archMethods = () => archState.split === "validation" ? ["baseline", "mlp", "gru", "riccardovib"] : ["mlp", "gru", "riccardovib"];
  const archNumber = value => archState.metric === "mse" ? value.toExponential(3)
    : value.toLocaleString("zh-TW", {minimumFractionDigits:3, maximumFractionDigits:5});
  const archMean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const archRows = () => architecture.records.filter(record => $("arch-ratio").value === "all" || record.ratio === Number($("arch-ratio").value));
  const archValue = (record, method) => record[archState.split][method][archState.metric];
  const archTone = {baseline:"baseline", mlp:"ours", gru:"gru", riccardovib:"author"};
  const archRatios = [...new Set(architecture.records.map(record => record.ratio))].sort((a,b) => a-b);
  $("arch-ratio").innerHTML += archRatios.map(ratio => `<option value="${ratio}">${ratio}:1</option>`).join("");
  $("architecture").hidden = false;

  function renderArchitecture() {
    const rows = archRows();
    const metric = archState.metric;
    const methods = archMethods();
    const splitName = archState.split === "validation" ? "驗證" : "探索性測試";
    document.querySelectorAll("[data-arch-split]").forEach(button => button.setAttribute("aria-pressed", button.dataset.archSplit === archState.split));
    document.querySelectorAll("[data-arch-metric]").forEach(button => button.setAttribute("aria-pressed", button.dataset.archMetric === metric));
    $("arch-scope").textContent = archState.split === "validation"
      ? `驗證集 · ${rows.length}／620 檔 · ${definitions[metric].help}。本方 MLP／GRU 的架構選擇只依驗證 ESR；RiccardoVib 原始訓練素材可能重疊。`
      : `已開封測試集的探索性重用 · ${rows.length}／620 檔 · ${definitions[metric].help}。純演算法未在此測試集評估；RiccardoVib 原始訓練素材可能重疊。`;
    const means = Object.fromEntries(methods.map(method => [method, $("arch-ratio").value === "all"
      ? architecture[archState.split][method][metric]
      : archMean(rows.map(record => archValue(record, method)))]));
    const winner = methods.reduce((best, method) => means[method] < means[best] ? method : best, methods[0]);
    const chartMax = Math.max(...Object.values(means));
    $("arch-cards").innerHTML = methods.map(method => `<div class="architecture-card" style="--arch-color:var(--${archTone[method]})"><div class="architecture-card-head"><span>${archNames[method]}</span><small>${splitName} · ${rows.length} 檔</small></div><strong>${archNumber(means[method])}</strong><span class="architecture-card-unit">${definitions[metric].name} ↓ · 逐檔等權平均${method === winner ? " · 數值最低" : ""}</span><div class="architecture-card-track"><span style="width:${chartMax ? means[method] / chartMax * 100 : 0}%"></span></div></div>`).join("");
    const groups = [...new Set(rows.map(record => record.ratio))].sort((a,b) => a-b).map(ratio => ({ratio, records:rows.filter(record => record.ratio === ratio)}));
    const groupMax = Math.max(0, ...groups.flatMap(group => methods.map(method => archMean(group.records.map(record => archValue(record, method))))));
    $("arch-ratio-chart").innerHTML = groups.map(group => `<div class="architecture-ratio-group"><div class="architecture-ratio-label">${group.ratio}:1 <small>n=${group.records.length}</small></div>${methods.map(method => {
      const value = archMean(group.records.map(record => archValue(record, method)));
      return `<div class="architecture-bar" style="--arch-color:var(--${archTone[method]})"><span>${archShort[method]}</span><div class="architecture-bar-track"><i style="width:${groupMax ? value / groupMax * 100 : 0}%"></i></div><b>${archNumber(value)}</b></div>`;
    }).join("")}</div>`).join("");
    const sortMethod = $("arch-record-sort").value;
    const sorted = [...rows].sort((a,b) => Math.abs(archValue(b,sortMethod) - archValue(b,"mlp")) - Math.abs(archValue(a,sortMethod) - archValue(a,"mlp")));
    $("arch-record-caption").textContent = `${splitName} · ${definitions[metric].name} · ${archShort[sortMethod]} 與 MLP 差異最大的 12 檔；正差代表前者誤差較高。`;
    $("arch-record-table").innerHTML = `<table><thead><tr><th scope="col">檔案</th><th scope="col">MLP</th><th scope="col">GRU</th><th scope="col">RiccardoVib</th><th scope="col">${archShort[sortMethod]} − MLP</th></tr></thead><tbody>${sorted.slice(0,12).map(record => {
      const delta = archValue(record,sortMethod) - archValue(record,"mlp");
      return `<tr><td>${record.id}</td><td>${archNumber(archValue(record,"mlp"))}</td><td>${archNumber(archValue(record,"gru"))}</td><td>${archNumber(archValue(record,"riccardovib"))}</td><td class="${delta < 0 ? "architecture-better" : ""}">${delta > 0 ? "+" : ""}${archNumber(delta)}</td></tr>`;
    }).join("")}</tbody></table>`;
  }
  document.querySelectorAll("[data-arch-split]").forEach(button => button.addEventListener("click", () => {
    archState.split = button.dataset.archSplit;
    renderArchitecture();
  }));
  document.querySelectorAll("[data-arch-metric]").forEach(button => button.addEventListener("click", () => {
    archState.metric = button.dataset.archMetric;
    renderArchitecture();
  }));
  $("arch-ratio").addEventListener("change", renderArchitecture);
  $("arch-record-sort").addEventListener("change", renderArchitecture);
  $("arch-csv").addEventListener("click", () => {
    const methods = archMethods();
    const header = ["split", "scope", "record", "ratio", "metric", ...methods];
    const rows = archRows().flatMap(record => ["esr","mse","mae","mrstft","rms_db_mae"].map(metric => [
      archState.split, archState.split === "test" ? "exploratory_reuse" : "model_selection", record.id, record.ratio, metric,
      ...methods.map(method => record[archState.split][method][metric])
    ]));
    download("\uFEFF" + [header, ...rows].map(row => row.join(",")).join("\r\n"), `cl1b-architecture-${archState.split}.csv`, "text/csv;charset=utf-8");
  });
  renderArchitecture();
}
