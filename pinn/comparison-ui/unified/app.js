(function(){
  'use strict';
  const C=ComparisonCore, $=id=>document.getElementById(id);
  const data=JSON.parse($('benchmark-data').textContent);
  let study=data.five_models;
  const metrics=Object.keys(C.METRICS),models=C.METHODS.filter(m=>m.key!=='baseline');
  const state={metric:'esr',page:0,historyPage:0},pageSize=20;
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const scopes={heldout:'既有保留資料 · 探索性重用',diagnostic:'訓練區間重疊 · 診斷用途',special:'專項評估 · 不混入主要排名'};
  const scope=()=>study.candidate?`${study.candidate.id}: ${scopes[study.candidate.category]}`:'reused_validation';
  const selector=$('evaluation');
  selector.innerHTML=(data.long_form?Object.entries(data.long_form.studies).sort(([a],[b])=>a.localeCompare(b)).map(([id,s])=>`<option value="${id}">${id} · ${esc(s.candidate.title)} · ${s.candidate.seconds.toFixed(1)} 秒</option>`).join(''):'')+'<option value="short">歷史：共同短視窗 · 21.33 ms</option>';
  selector.value=data.long_form?'B':'short';
  function selectStudy(){
    study=selector.value==='short'?data.five_models:{...data.long_form.studies[selector.value],protocol:data.long_form.protocol,sources:data.long_form.sources,window_signature:data.long_form.window_signature,manifest_signature:data.long_form.manifest_signature,gpu:data.long_form.protocol.gpu,cuda:data.long_form.protocol.cuda};
    const c=study.candidate;
    $('evaluation-note').textContent=c?`${scopes[c.category]}。共同時間軸 ${c.start}–${c.stop} 秒，每檔 ${c.seconds.toFixed(1)} 秒，共 ${study.files} 檔。${c.content} ${c.id==='G'?'缺少 125 檔 Release＝0；勿與 620 檔群體直接比較。':''}${c.id==='F'?'掃頻和 A 重複，獨立列示以免重複加權。':''}${c.id==='H'?'靜音專項僅 2 秒，檢查底噪與殘留輸出，不當作一般有聲排名。':''}未進行增益擬合或重新選模。`:'歷史共同短視窗驗證：每檔 2 個視窗，4,096 樣本暖機＋1,024 樣本計分。';
    $('scope-copy').textContent=data.long_form?'新長片段已用全部六模型與純演算法重新推論。這些資料曾參與先前研究，C／D／E 更與訓練區間重疊；不是新的獨立測試。各片段分開比較，保留歷史短視窗作為另一份評估。':'這批驗證資料曾用於選模；目前沒有涵蓋六模型的新獨立測試。顯示既有共同短視窗評估。';
    $('fact-files').textContent=String(study.files);
    $('fact-windows').textContent=String(study.windows);
    $('fact-length').textContent=c?`${c.seconds.toFixed(1)} 秒`:'21.33 ms';
    $('record-note').textContent=c?'沿用上方片段、旋鈕篩選與指標；每檔在同一連續片段計分。':'沿用上方旋鈕篩選與指標。每檔包含兩個相同位置的驗證視窗。';
    state.page=0;
  }
  selectStudy();
  const filters=()=>Object.fromEntries(Object.keys(C.FIELDS).map(k=>[k,$('filter-'+k).value]));
  const rows=()=>C.filterRows(study.records,filters());
  const modeName=()=> $('aggregate').value==='macro'?'逐檔等權平均':'合併樣本計分';
  const name=key=>C.METHODS.find(m=>m.key===key).name;
  const detail=key=>key==='baseline'?'原始電路參數 · 未擬合':key==='riccardovib'?'固定權重 · 未重新訓練':data.histories[key]?.kind==='epoch_validation'?`最佳第 ${data.histories[key].selected_epoch}／16 輪`:'完整訓練區間 · 學得電路參數';
  const help={esr:'ESR 對低能量素材較敏感；逐檔平均與合併樣本的排序可能不同。',mse:'以原始振幅計算平方誤差，較大的瞬間誤差影響較大。',mae:'原始振幅的平均絕對誤差。',mrstft:'共同短視窗的三尺度頻譜誤差；不要與歷史長片段的頻譜分數混排。',rms_db_mae:'10 ms 格點的 RMS 電平差，單位 dB；尾格按樣本數加權。'};
  for(const [filter,field] of Object.entries(C.FIELDS)){
    const values=[...new Set(study.records.map(r=>r.labels[field]))].sort((a,b)=>a-b);
    $('filter-'+filter).innerHTML+=values.map(v=>`<option value="${v}">${v}${filter==='ratio'?':1':filter==='threshold'?' dB':''}</option>`).join('');
  }
  $('record-sort').innerHTML=C.METHODS.map(m=>`<option value="${m.key}"${m.key==='s6_pinn'?' selected':''}>${m.name}</option>`).join('');

  function render(){
    const selected=rows(),summary=C.summarize(selected,$('aggregate').value,study.protocol.esr_floor),metric=state.metric;
    const ordered=[...models].sort((a,b)=>(summary[a.key][metric]??0)-(summary[b.key][metric]??0));
    const maximum=selected.length?Math.max(...C.METHODS.map(m=>summary[m.key][metric])):0;
    document.querySelectorAll('[data-metric]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.metric===metric)));
    $('filter-count').textContent=`${selected.length}／${study.files} 檔 · ${selected.length*(study.candidate?1:2)} 個共同${study.candidate?'片段':'視窗'}`;
    $('metric-help').textContent=`${modeName()} · 全部越低越好。依 ${C.METRICS[metric]} 排序，粗體標示六模型內最低值。${help[metric]}`;
    if(metric==='mrstft'&&study.candidate)$('metric-help').textContent=`${modeName()} · FFT 512／1,024／2,048 的長片段頻譜誤差，越低越好。與歷史短視窗的 FFT 尺度不同，不能直接比較分數高低。`;
    const cells=m=>metrics.map(k=>{
      const v=summary[m.key][k],best=selected.length&&m.key!=='baseline'&&v===Math.min(...models.map(x=>summary[x.key][k]));
      return `<td class="${k===metric?'metric-selected ':''}${best?'best':''}"><span class="value">${C.fmt(v,k)}</span>${k===metric&&maximum?`<i class="bar" aria-hidden="true" style="width:${v/maximum*80}%;--method:${m.color}"></i>`:''}</td>`;
    }).join('');
    const row=(m,i)=>`<tr><th scope="row"><span class="method-name"><span class="rank">${i===null?'—':i+1}</span><i class="dot" style="--method:${m.color}" aria-hidden="true"></i>${m.name}${i===null?'<span class="badge">基準</span>':''}</span><small>${detail(m.key)}</small></th>${cells(m)}</tr>`;
    $('comparison-table').innerHTML=selected.length?`<table class="comparison-table"><caption class="sr-only">六模型與基準的五項誤差 · ${modeName()}</caption><thead><tr><th scope="col">模型 / 推論方法</th>${metrics.map(k=>`<th scope="col" class="${k===metric?'metric-selected':''}">${C.METRICS[k]} ↓</th>`).join('')}</tr></thead><tbody>${ordered.map(row).join('')}</tbody><tfoot>${row(C.METHODS.at(-1),null)}</tfoot></table>`:'<p class="empty">沒有符合條件的錄音，請調整旋鈕或重設篩選。</p>';
    if(selected.length){
      const bestValue=summary[ordered[0].key][metric],winners=ordered.filter(m=>summary[m.key][metric]===bestValue).map(m=>m.name).join('、');
      const base=summary.baseline[metric],relation=bestValue<base?'低於':bestValue>base?'高於':'等於';
      $('ranking-insight').textContent=`${winners} 在目前範圍的 ${C.METRICS[metric]} 最低（${C.fmt(bestValue,metric)}），${relation}純演算法基準 ${C.fmt(base,metric)}。${study.candidate?'僅描述所選片段與旋鈕組合；不代表未見素材泛化。':'這是短視窗驗證結果，不能代表長片段表現。'}`;
      const a=summary.s6_pinn[metric],b=summary.s6[metric];
      $('pinn-insight').textContent=`S6＋PINN：${C.fmt(a,metric)}；S6：${C.fmt(b,metric)}。${a===b?'此指標相同':`此指標${a<b?'較低':'較高'}${b?` ${Math.abs((a-b)/b*100).toFixed(1)}%`:''}`}。切換指標可檢視取捨；此實驗也改了輸出端，無法只歸因於物理 loss。`;
    }else{
      $('ranking-insight').textContent='目前沒有可比較的檔案。';$('pinn-insight').textContent='請調整篩選條件。';
    }
    renderRecords(selected);
  }
  function renderRecords(selected){
    const key=$('record-sort').value,metric=state.metric;
    const sorted=[...selected].sort((a,b)=>b.metrics[key][metric]-a.metrics[key][metric]||a.id.localeCompare(b.id));
    const pages=Math.max(1,Math.ceil(sorted.length/pageSize));state.page=Math.min(state.page,pages-1);
    const visible=sorted.slice(state.page*pageSize,(state.page+1)*pageSize);
    $('record-table').innerHTML=`<table><caption class="sr-only">逐檔 ${C.METRICS[metric]}，依 ${name(key)} 由高至低</caption><thead><tr><th scope="col">錄音 / 旋鈕</th>${C.METHODS.map(m=>`<th scope="col">${m.name}</th>`).join('')}</tr></thead><tbody>${visible.length?visible.map(r=>`<tr><th scope="row">${esc(r.id)}<small>A${r.labels.attack_index} · R${r.labels.release_index} · ${r.labels.ratio}:1 · ${r.labels.threshold_db} dB</small></th>${C.METHODS.map(m=>`<td>${C.fmt(r.metrics[m.key][metric],metric)}</td>`).join('')}</tr>`).join(''):'<tr><td colspan="8" class="empty">沒有符合條件的錄音。</td></tr>'}</tbody></table>`;
    $('page-label').textContent=selected.length?`第 ${state.page+1}／${pages} 頁 · ${state.page*pageSize+1}–${Math.min((state.page+1)*pageSize,selected.length)}／${selected.length} 檔`:'0 檔';
    $('prev').disabled=state.page===0;$('next').disabled=state.page===pages-1;
  }
  function renderHistory(){
    const key=$('history-model').value,h=data.histories[key];
    if(h.kind==='step_training'){renderStepHistory(key,h);return;}
    const best=h.rows.find(r=>r.epoch===h.selected_epoch);
    $('history-note').textContent='曲線為驗證 macro ESR，非訓練 loss。下表 loss 是當輪按視窗數加權的批次平均；不同模型的訓練目標不同，勿直接比較總 loss。';
    $('history-prev').disabled=true;$('history-next').disabled=true;$('history-page').textContent='全部 16 輪';
    $('history-summary').textContent=`${name(key)} · 完成 ${h.epochs}／16 輪 · 依驗證 macro ESR 選擇第 ${h.selected_epoch} 輪：${C.fmt(best.esr)}。每輪走訪 8,680 個訓練區段，各取代表視窗；非完整樣本覆蓋。`;
    const max=Math.max(...h.rows.map(r=>r.esr))*1.12||1;
    const x=epoch=>60+(epoch-1)/(h.epochs-1)*850,y=v=>210-v/max*170;
    $('history-chart').innerHTML=`<svg viewBox="0 0 960 260" role="img" aria-label="${name(key)} 的 16 輪驗證 ESR；最佳第 ${h.selected_epoch} 輪"><title>驗證 macro ESR，越低越好</title>${[0,.5,1].map(f=>`<line class="grid" x1="60" x2="910" y1="${y(max*f)}" y2="${y(max*f)}"/><text x="48" y="${y(max*f)+4}" text-anchor="end">${(max*f).toFixed(2)}</text>`).join('')}<polyline class="curve" points="${h.rows.map(r=>`${x(r.epoch)},${y(r.esr)}`).join(' ')}"/>${h.rows.map(r=>`<circle class="${r.epoch===h.selected_epoch?'selected-point':'point'}" cx="${x(r.epoch)}" cy="${y(r.esr)}" r="${r.epoch===h.selected_epoch?6:3}"><title>第 ${r.epoch} 輪：${r.esr}</title></circle>`).join('')}${[1,4,8,12,16].map(e=>`<text x="${x(e)}" y="235" text-anchor="middle">${e}</text>`).join('')}<text x="910" y="20" text-anchor="end">驗證 macro ESR / 輪次</text></svg>`;
    $('history-table').innerHTML=`<table><caption class="sr-only">${name(key)} 的每輪紀錄</caption><thead><tr><th scope="col">輪次</th><th scope="col">訓練總 loss</th><th scope="col">音訊 loss</th><th scope="col">物理 loss</th><th scope="col">驗證 macro ESR</th><th scope="col">驗證 pooled ESR</th></tr></thead><tbody>${h.rows.map(r=>`<tr class="${r.epoch===h.selected_epoch?'selected-row':''}"><th scope="row">${r.epoch}${r.epoch===h.selected_epoch?' · 選用':''}</th>${[r.training_loss,r.audio_loss,r.physics_loss,r.esr,r.pooled_esr].map(v=>`<td>${C.fmt(v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  }
  function renderStepHistory(key,h){
    const last=h.rows.at(-1),max=Math.max(...h.rows.map(r=>r.training_loss))*1.12||1;
    $('history-summary').textContent=`${name(key)} · 完成 1／1 輪、${h.final_step.toLocaleString('zh-TW')} 次更新；620 檔的 ${h.frames_covered.toLocaleString('zh-TW')} 個訓練樣本已完整走訪。最佳輪次：不適用，採用最終參數。${h.selection_note}`;
    $('history-note').textContent=`曲線為已保存的單批訓練總 loss，連線僅供閱讀，非驗證成績或每輪平均。${h.sampling_note}共 ${h.rows.length.toLocaleString('zh-TW')} 筆，最新紀錄為第 ${last.step.toLocaleString('zh-TW')} 步；表格由新至舊。總 loss 包含音訊、物理與參數先驗，不能只用前兩項相加。來源：${h.source}。`;
    const x=step=>60+step/h.final_step*850,y=v=>210-v/max*170;
    const ticks=[0,.25,.5,.75,1].map(f=>Math.round(h.final_step*f));
    $('history-chart').innerHTML=`<svg viewBox="0 0 960 260" role="img" aria-label="${name(key)} 已記錄的單批訓練總 loss 對更新步數"><title>單批訓練總 loss；非驗證成績</title>${[0,.5,1].map(f=>`<line class="grid" x1="60" x2="910" y1="${y(max*f)}" y2="${y(max*f)}"/><text x="48" y="${y(max*f)+4}" text-anchor="end">${(max*f).toFixed(2)}</text>`).join('')}<polyline class="curve" points="${h.rows.map(r=>`${x(r.step).toFixed(2)},${y(r.training_loss).toFixed(2)}`).join(' ')}"/><circle class="point" cx="${x(last.step)}" cy="${y(last.training_loss)}" r="4"><title>最新紀錄：第 ${last.step} 步，loss ${last.training_loss}；非最佳 checkpoint</title></circle>${ticks.map(s=>`<text x="${x(s)}" y="235" text-anchor="middle">${s.toLocaleString('zh-TW')}</text>`).join('')}<text x="910" y="20" text-anchor="end">單批訓練總 loss / 更新步數</text></svg>`;
    const pages=Math.ceil(h.rows.length/pageSize);state.historyPage=Math.min(state.historyPage,pages-1);
    const visible=[...h.rows].reverse().slice(state.historyPage*pageSize,(state.historyPage+1)*pageSize);
    $('history-table').innerHTML=`<table><caption class="sr-only">${name(key)} 已保存的單批訓練紀錄，由新至舊</caption><thead><tr><th scope="col">更新步數</th><th scope="col">單批總 loss</th><th scope="col">單批音訊 loss</th><th scope="col">單批物理 loss</th></tr></thead><tbody>${visible.map(r=>`<tr><th scope="row">${r.step.toLocaleString('zh-TW')}</th>${[r.training_loss,r.audio_loss,r.physics_loss].map(v=>`<td>${C.fmt(v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    $('history-page').textContent=`第 ${state.historyPage+1}／${pages} 頁 · 共 ${h.rows.length.toLocaleString('zh-TW')} 筆`;
    $('history-prev').disabled=state.historyPage===0;$('history-next').disabled=state.historyPage===pages-1;
  }
  function renderCoverage(){
    const training=key=>data.coverage[key]?`${(data.coverage[key]/1e9).toFixed(3)} 十億樣本 · 完整訓練區間`:
      data.histories[key]?'16 輪 × 8,680 區段代表視窗':key==='baseline'?'未訓練；原始參數':'使用既有權重；原始素材重疊未知';
    const inference=key=>key==='baseline'?'電路數值求解':key==='mlp'||key==='gru'?'學得參數後，電路數值求解':key==='s6_pinn'?'神經網路預測物理狀態':'神經網路推論';
    $('coverage-table').innerHTML=`<table class="coverage-table"><caption class="sr-only">訓練量、推論方式與評估覆蓋</caption><thead><tr><th scope="col">方法</th><th scope="col">本次訓練來源 / 規模</th><th scope="col">推論方式</th><th scope="col">共同短視窗驗證</th><th scope="col">歷史完整區段測試</th></tr></thead><tbody>${C.METHODS.map(m=>`<tr><th scope="row">${m.name}${m.key==='baseline'?' · 基準':''}</th><td>${training(m.key)}</td><td>${inference(m.key)}</td><td>已完成</td><td>${data.availability[m.key].historical_full_test?'已完成（歷史）':'未執行'}</td></tr>`).join('')}</tbody></table>`;
    $('provenance').innerHTML=`<p>共同資料：<code>runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json</code></p><p>視窗簽章：<code>${esc(study.window_signature)}</code></p><p>切分簽章：<code>${esc(study.manifest_signature)}</code></p><p>訓練：${data.split_hours.train.toFixed(3)} 小時；驗證：${data.split_hours.validation.toFixed(3)} 小時；歷史保留測試：${data.split_hours.test.toFixed(3)} 小時。此處為全部區段時數，總表僅計分共同短視窗。</p><p>評估環境：${esc(study.gpu)} / CUDA ${esc(study.cuda)}；純演算法使用 CPU 求解。</p>${C.METHODS.map(m=>`<p>${m.name}：<code>${esc(study.sources[m.key].path)}</code><br>SHA-256：<code>${esc(study.sources[m.key].sha256)}</code></p>`).join('')}`;
    $('built-at').textContent=`頁面整理：${new Date(data.built_utc).toLocaleString('zh-TW',{timeZone:'Asia/Taipei',hour12:false})} 台北時間 · 離線快照`;
    if(data.long_form){
      $('coverage-table').innerHTML=$('coverage-table').innerHTML.replace('<th scope="col">歷史完整區段測試</th>','<th scope="col">本次 A–H 長片段／專項</th>').replace(/<td>(?:已完成（歷史）|未執行)<\/td>/g,'<td>七方法皆完成；G 為 495 檔</td>');
      $('provenance').innerHTML=`<p>目前評估：${esc(scope())}；長片段來源 <code>runs/long-clip-evaluation-20260929/comparison.json</code>。長片段從錄音起點連續帶入 Dry，RiccardoVib 依其原生 16 樣本歷史；無重新訓練。</p><p>長片段完成時間：${esc(data.long_form.completed_utc)}；長片段簽章：<code>${esc(data.long_form.window_signature)}</code></p>`+$('provenance').innerHTML.replace('共同資料：','歷史短視窗來源：').replace('視窗簽章：','目前選擇的視窗簽章：').replace('此處為全部區段時數，總表僅計分共同短視窗。','此處為原始切分區段時數，總表依上方片段選單計分。');
    }
  }
  function download(content,filename,type){
    const url=URL.createObjectURL(new Blob([content],{type})),link=document.createElement('a');
    link.href=url;link.download=filename;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>{state.metric=b.dataset.metric;state.page=0;render();}));
  selector.addEventListener('change',()=>{selectStudy();render();renderCoverage();});
  for(const id of ['aggregate','record-sort',...Object.keys(C.FIELDS).map(k=>'filter-'+k)])$(id).addEventListener('change',()=>{state.page=0;render();});
  $('reset').addEventListener('click',()=>{Object.keys(C.FIELDS).forEach(k=>$('filter-'+k).value='all');state.page=0;render();});
  $('prev').addEventListener('click',()=>{state.page=Math.max(0,state.page-1);render();});
  $('next').addEventListener('click',()=>{state.page++;render();});
  $('history-model').addEventListener('change',()=>{state.historyPage=0;renderHistory();});
  $('history-prev').addEventListener('click',()=>{state.historyPage=Math.max(0,state.historyPage-1);renderHistory();});
  $('history-next').addEventListener('click',()=>{state.historyPage++;renderHistory();});
  $('export-history').addEventListener('click',()=>{
    const key=$('history-model').value,h=data.histories[key],columns=h.kind==='step_training'?['step','training_loss','audio_loss','physics_loss']:['epoch','training_loss','audio_loss','physics_loss','esr','pooled_esr'];
    download(C.csv([['model','record_type',...columns],...h.rows.map(r=>[key,h.kind,...columns.map(k=>r[k])])]),`cl1b-${key}-training-history.csv`,'text/csv;charset=utf-8');
  });
  $('export-records').addEventListener('click',()=>download(C.recordCSV(rows(),study.window_signature,scope()),`cl1b-${selector.value}-records.csv`,'text/csv;charset=utf-8'));
  $('export-summary').addEventListener('click',()=>{
    const selected=rows(),summary=C.summarize(selected,$('aggregate').value,study.protocol.esr_floor);
    download(C.csv([['scope','window_signature','aggregation','files','filters','method',...metrics],...C.METHODS.map(m=>[scope(),study.window_signature,$('aggregate').value,selected.length,JSON.stringify(filters()),m.key,...metrics.map(k=>summary[m.key][k])])]),`cl1b-${selector.value}-summary.csv`,'text/csv;charset=utf-8');
  });
  $('export-json').addEventListener('click',()=>download(JSON.stringify(data,null,2),'cl1b-comparison.json','application/json'));
  function setTheme(theme){document.documentElement.dataset.theme=theme;$('theme').textContent=theme==='dark'?'淺色模式':'深色模式';$('theme').setAttribute('aria-label',`切換${theme==='dark'?'淺':'深'}色模式`);}
  try{setTheme(localStorage.getItem('ht1b-report-theme')==='dark'?'dark':'light');}catch(_){setTheme('light');}
  $('theme').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';setTheme(theme);try{localStorage.setItem('ht1b-report-theme',theme);}catch(_){}});
  function revealAnchor(){
    const id=window.location.hash.slice(1),target=$(id);
    if(target&&target.tagName==='DETAILS')target.open=true;
  }
  window.addEventListener('hashchange',revealAnchor);
  render();renderHistory();renderCoverage();revealAnchor();
})();
