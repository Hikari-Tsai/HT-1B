"use strict";
if (data.s6_pinn_study) {
  const experiment = data.s6_pinn_study;
  const status = experiment.status || {};
  const section = $("s6-pinn-study");
  section.hidden = false;
  const phaseNames = {preparing:"準備中",queued_for_gpu:"等待其他 GPU 作業結束",gpu_smoke_test:"GPU 冒煙測試",
    training:"訓練中",validation:"輪次驗證",epoch_complete:"輪次完成",final_common_validation:"最佳模型共同驗證",
    building_page:"建立報告",complete:"完成",failed:"執行失敗"};
  const finished = !!data.five_models?.sources.s6_pinn;
  $("s6-pinn-status").textContent = finished ? `已完成共同驗證 · 最佳第 ${experiment.best.selected_epoch}／16 輪`
    : `${phaseNames[status.phase] || "準備中"} · 第 ${status.epoch || 0}／16 輪`;
  $("s6-pinn-time").textContent = `頁面快照：${status.utc ? new Date(status.utc).toLocaleString("zh-TW") : "—"}；訓練中由每十分鐘回報提供進度。`;
  $("s6-pinn-result").textContent = finished
    ? `驗證 ESR：逐檔平均 ${data.five_models.macro_mean.s6_pinn.esr.toFixed(5)}，合併樣本 ${data.five_models.pooled.s6_pinn.esr.toFixed(5)}。可在上方切換指標、旋鈕與彙整方式，和原 S6 一起比較。`
    : "結果尚未產生。訓練與最佳權重驗證完成後，程式會自動加入上方的共同比較。";
  if (status.phase === "failed") $("s6-pinn-result").textContent = `執行失敗：${status.error || "請查看訓練紀錄"}`;
  const history = experiment.history?.history || [];
  if (history.length) {
    $("s6-pinn-history").innerHTML = `<table><caption>每輪結果：訓練 loss 為當輪各批次按視窗數加權平均；驗證 ESR 為逐檔平均。</caption><thead><tr><th>輪次</th><th>音訊 loss</th><th>物理 loss</th><th>驗證 ESR ↓</th><th>驗證 C3 殘差 MSE</th><th>驗證快狀態 MSE</th><th>驗證慢狀態 MSE</th></tr></thead><tbody>${history.map(h=>`<tr><th>${h.epoch}${h.epoch === experiment.best?.selected_epoch ? " · 最佳" : ""}</th><td>${h.train_mean_losses.audio.toPrecision(5)}</td><td>${h.train_mean_losses.physics.toPrecision(5)}</td><td>${h.validation_macro_mean.esr.toPrecision(5)}</td><td>${h.validation_physics_mse.c3_kcl.toPrecision(5)}</td><td>${h.validation_physics_mse.gre_fast.toPrecision(5)}</td><td>${h.validation_physics_mse.gre_slow.toPrecision(5)}</td></tr>`).join("")}</tbody></table>`;
  }
}
