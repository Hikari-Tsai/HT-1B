(function(){
  'use strict';
  const C=ComparisonCore, $=id=>document.getElementById(id);
  const data=JSON.parse($('benchmark-data').textContent);
  let study=data.five_models;
  const metrics=Object.keys(C.METRICS),models=C.METHODS.filter(m=>m.key!=='baseline');
  const state={metric:'esr',page:0,historyPage:0,profile:'s6_pinn'},pageSize=20;
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
    $('scope-copy').textContent=data.long_form?'新長片段已用全部六模型與純演算法重新推論。這些資料曾參與先前研究，C／D／E 更與訓練區間重疊；不是新的獨立測試。各片段分開比較，保留歷史短視窗作為另一份評估。'+(data.full_test?'所有模型的完整測試集分數見第 02 節。':''):'這批驗證資料曾用於選模，顯示既有共同短視窗評估。'+(data.full_test?'所有模型的完整測試集分數見第 02 節。':'目前沒有涵蓋六模型的新獨立測試。');
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
  $('profile-model').innerHTML=models.map(m=>`<option value="${m.key}"${m.key===state.profile?' selected':''}>${m.name}</option>`).join('');

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
    renderDiagnostics(selected,summary);
  }
  function renderDiagnostics(selected,summary){
    const chosen=models.find(m=>m.key===state.profile)||models[0];
    if(!selected.length){$('metric-profile').innerHTML='';$('rank-profile').innerHTML='';$('profile-note').textContent='沒有符合條件的資料。';return;}
    const values=metrics.map(k=>summary[chosen.key][k]);
    const maxima=metrics.map(k=>Math.max(...models.map(m=>summary[m.key][k]))||1);
    const cx=180,cy=160,r=112,angles=metrics.map((_,i)=>-Math.PI/2+i*Math.PI*2/metrics.length);
    const point=(i,f)=>[cx+Math.cos(angles[i])*r*f,cy+Math.sin(angles[i])*r*f];
    const polygon=f=>metrics.map((_,i)=>point(i,f).join(',')).join(' ');
    const shape=values.map((v,i)=>point(i,1-Math.min(1,v/maxima[i])).join(',')).join(' ');
    $('metric-profile').innerHTML=`<svg viewBox="0 0 360 330" aria-hidden="true"><polygon class="ring" points="${polygon(.25)}"/><polygon class="ring" points="${polygon(.5)}"/><polygon class="ring" points="${polygon(.75)}"/><polygon class="ring" points="${polygon(1)}"/>${metrics.map((k,i)=>{const [x,y]=point(i,1);const [tx,ty]=point(i,1.22);return `<line class="axis" x1="${cx}" y1="${cy}" x2="${x}" y2="${y}"/><text x="${tx}" y="${ty}" text-anchor="middle">${C.METRICS[k]}</text>`}).join('')}<polygon class="shape" style="--method:${chosen.color}" points="${shape}"/>${values.map((v,i)=>{const [x,y]=point(i,1-Math.min(1,v/maxima[i]));return `<circle class="dot" style="--method:${chosen.color}" cx="${x}" cy="${y}" r="4"><title>${C.METRICS[metrics[i]]}: ${C.fmt(v,metrics[i])}</title></circle>`}).join('')}</svg>`;
    $('profile-note').textContent=`${chosen.name} · 目前範圍的相對輪廓；中心較近表示此範圍內較低誤差，圖形面積不能跨評估片段比較。`;
    const ranks=metrics.map(k=>[...models].sort((a,b)=>summary[a.key][k]-summary[b.key][k]).map(m=>m.key));
    const w=620,h=300,left=95,right=26,top=35,bottom=32,x=i=>left+i*(w-left-right)/(metrics.length-1),y=rank=>top+(rank-1)*(h-top-bottom)/(models.length-1);
    $('rank-profile').innerHTML=`<svg viewBox="0 0 ${w} ${h}" aria-hidden="true">${metrics.map((k,i)=>`<line class="axis" x1="${x(i)}" x2="${x(i)}" y1="${top}" y2="${h-bottom}"/><text x="${x(i)}" y="${h-8}" text-anchor="middle">${C.METRICS[k]}</text>`).join('')}${[1,2,3,4,5,6].map(rank=>`<text x="${left-12}" y="${y(rank)+4}" text-anchor="end">${rank}</text>`).join('')}${models.map(m=>{const pts=metrics.map((_,i)=>`${x(i)},${y(ranks[i].indexOf(m.key)+1)}`).join(' ');return `<polyline class="rank-line" style="--method:${m.color}" points="${pts}">${metrics.map((k,i)=>`<title>${m.name} · ${C.METRICS[k]} 第 ${ranks[i].indexOf(m.key)+1} 名</title>`).join('')}</polyline>${metrics.map((_,i)=>`<circle class="rank-dot" style="--method:${m.color}" cx="${x(i)}" cy="${y(ranks[i].indexOf(m.key)+1)}" r="4"/>`).join('')}`}).join('')}</svg>`;
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
    $('coverage-table').innerHTML=`<table class="coverage-table"><caption class="sr-only">訓練量、推論方式與評估覆蓋</caption><thead><tr><th scope="col">方法</th><th scope="col">本次訓練來源 / 規模</th><th scope="col">推論方式</th><th scope="col">共同短視窗驗證</th><th scope="col">歷史完整區段測試</th>${data.full_test?'<th scope="col">本次完整測試集</th>':''}</tr></thead><tbody>${C.METHODS.map(m=>`<tr><th scope="row">${m.name}${m.key==='baseline'?' · 基準':''}</th><td>${training(m.key)}</td><td>${inference(m.key)}</td><td>已完成</td><td>${data.availability[m.key].historical_full_test?'已完成（歷史）':'未執行'}</td>${data.full_test?`<td>${data.availability[m.key].new_independent_test?`已完成 · ${data.full_test.files} 檔`:'未執行'}</td>`:''}</tr>`).join('')}</tbody></table>`;
    $('provenance').innerHTML=`<p>共同資料：<code>runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json</code></p><p>視窗簽章：<code>${esc(study.window_signature)}</code></p><p>切分簽章：<code>${esc(study.manifest_signature)}</code></p><p>訓練：${data.split_hours.train.toFixed(3)} 小時；驗證：${data.split_hours.validation.toFixed(3)} 小時；歷史保留測試：${data.split_hours.test.toFixed(3)} 小時。此處為全部區段時數，總表僅計分共同短視窗。</p><p>評估環境：${esc(study.gpu)} / CUDA ${esc(study.cuda)}；純演算法使用 CPU 求解。</p>${C.METHODS.map(m=>`<p>${m.name}：<code>${esc(study.sources[m.key].path)}</code><br>SHA-256：<code>${esc(study.sources[m.key].sha256)}</code></p>`).join('')}`;
    $('built-at').textContent=`頁面整理：${new Date(data.built_utc).toLocaleString('zh-TW',{timeZone:'Asia/Taipei',hour12:false})} 台北時間 · 離線快照`;
    if(data.long_form){
      $('coverage-table').innerHTML=$('coverage-table').innerHTML.replace('<th scope="col">歷史完整區段測試</th>','<th scope="col">本次 A–H 長片段／專項</th>').replace(/<td>(?:已完成（歷史）|未執行)<\/td>/g,'<td>七方法皆完成；G 為 495 檔</td>');
      $('provenance').innerHTML=`<p>目前評估：${esc(scope())}；長片段來源 <code>runs/long-clip-evaluation-20260929/comparison.json</code>。長片段從錄音起點連續帶入 Dry，RiccardoVib 依其原生 16 樣本歷史；無重新訓練。</p><p>長片段完成時間：${esc(data.long_form.completed_utc)}；長片段簽章：<code>${esc(data.long_form.window_signature)}</code></p>`+$('provenance').innerHTML.replace('共同資料：','歷史短視窗來源：').replace('視窗簽章：','目前選擇的視窗簽章：').replace('此處為全部區段時數，總表僅計分共同短視窗。','此處為原始切分區段時數，總表依上方片段選單計分。');
    }
  }
  const fullColumns=[['macro_esr','逐檔平均 ESR'],['median_esr','ESR 中位數'],['pooled_esr','合併 ESR'],['mrstft','MR-STFT'],['rms_db_mae','RMS 包絡 · dB'],['files_esr_above_1','ESR＞1 檔數']];
  const fullColor=key=>(C.METHODS.find(m=>m.key===key)||{color:{s6_bounded:'#1d6f45',s6_v3_e40:'#5b8fa8',s6_v3:'#8aa9b8',s4_v3:'#c47e7e',s6_pinn_fixed:'#b0779f',s6_pinn_ramp:'#6f3f63'}[key]}).color;
  function renderFullTest(){
    const t=data.full_test;
    if(!t)return;
    $('full-test').hidden=false;
    const fmtCell=(k,v)=>k==='files_esr_above_1'?String(v):C.fmt(v,k==='rms_db_mae'?'esr':k);
    const ranked=t.rows.filter(r=>r.status!=='baseline'),scale=Math.max(...ranked.filter(r=>r.macro_esr<=1).map(r=>r.macro_esr));
    const best=Object.fromEntries(fullColumns.map(([k])=>[k,Math.min(...ranked.filter(r=>r.status!=='post_hoc').map(r=>r[k]))]));
    const tag={post_hoc:'<span class="badge">事後選定</span>',reference:'<span class="badge">既有權重</span>',baseline:'<span class="badge">基準</span>'};
    let rank=0;
    const row=r=>{const shown=r.status==='baseline'||r.status==='post_hoc'?'—':++rank;return `<tr><th scope="row"><span class="method-name"><span class="rank">${shown}</span><i class="dot" style="--method:${fullColor(r.key)}" aria-hidden="true"></i>${esc(r.name)}${tag[r.status]||''}</span><small>${esc(r.note)}</small></th>${fullColumns.map(([k])=>`<td class="${k==='macro_esr'?'metric-selected ':''}${r.status!=='baseline'&&r.status!=='post_hoc'&&r[k]===best[k]?'best':''}"><span class="value">${fmtCell(k,r[k])}</span>${k==='macro_esr'?(r.macro_esr<=1?`<i class="bar" aria-hidden="true" style="width:${r.macro_esr/scale*80}%;--method:${fullColor(r.key)}"></i>`:'<small>超出尺度</small>'):''}</td>`).join('')}</tr>`;};
    $('full-test-table').innerHTML=`<table class="comparison-table"><caption class="sr-only">完整測試集各模型分數，依逐檔平均 ESR 排序</caption><thead><tr><th scope="col">模型 / 選模方式</th>${fullColumns.map(([k,n])=>`<th scope="col" class="${k==='macro_esr'?'metric-selected':''}">${n} ↓</th>`).join('')}</tr></thead><tbody>${ranked.map(row).join('')}</tbody><tfoot>${t.rows.filter(r=>r.status==='baseline').map(row).join('')}</tfoot></table>`;
    $('full-test-scope').textContent=`測試切分的全部 ${t.files} 檔、${t.hours.toFixed(2)} 小時（${t.samples_scored.toLocaleString('zh-TW')} 個評分樣本），依逐檔平均 ESR 排序，全部越低越好。粗體為未事後選定的模型中最低值；名次不含事後選定列與基準。`;
    const top=ranked.find(r=>r.status!=='post_hoc'),b=t.bounded_selection,old=t.rows.find(r=>r.key==='s6_v3');
    $('full-test-insight').textContent=`${top.name} 最低：逐檔平均 ESR ${fmtCell('macro_esr',top.macro_esr)}，最差一檔 ${fmtCell('macro_esr',top.max_esr)}，沒有 ESR＞1 的檔案。它在第 ${b.epoch}／${b.epochs} 輪依驗證逐檔 macro ESR（${fmtCell('macro_esr',b.validation_macro_esr)}）選出，規則在訓練前就固定。`;
    $('full-test-spike').textContent=old?`原本的 S6 v3 第 38 輪以驗證 pooled ESR 選出，中位數 ${fmtCell('median_esr',old.median_esr)} 很好，但 ${old.files_esr_above_1} 檔 ESR＞1、${old.files_esr_above_10} 檔＞10。原因是輸出增益沒有上限：近靜音時內部狀態悄悄放大，遇到突然起音就爆衝。pooled ESR 會掩蓋少數爆音檔，因此改用逐檔 macro ESR 選模，並把增益限制在 0–2（2·sigmoid）。`:'';
    renderPinnStudy(t);
    $('full-test-protocol').textContent=`每個測試區段從起點後至少 ${t.warmup_samples.toLocaleString('zh-TW')} 樣本（約 1.02 秒）的第一個 16 對齊樣本開始計分，到最後一個完整 16 樣本區塊。MR-STFT 使用 FFT ${t.mrstft_fft_sizes.map(n=>n.toLocaleString('zh-TW')).join('／')}，與歷史短視窗的頻譜分數不能直接比較。評估 GPU：${t.gpus.join('、')}。`;
  }
  function renderPinnStudy(t){
    const p=t.pinn;
    if(!p)return;
    $('full-test-pinn').hidden=false;
    const row=key=>t.rows.find(r=>r.key===key),f=v=>C.fmt(v,'esr'),sci=v=>v.toExponential(1);
    const ref=p.reference_states,run=p.runs;
    const lines=[
      ['s6_pinn','原版 · 短視窗 16 輪','第 8／16 輪',null,ref.s6_pinn],
      ['s6_pinn_fixed','A · 物理權重固定 1',`第 ${run.s6_pinn_fixed.epoch}／${run.s6_pinn_fixed.epochs} 輪`,run.s6_pinn_fixed.physics_mse,run.s6_pinn_fixed.states],
      ['s6_pinn_ramp',`B · 物理權重 ${run.s6_pinn_ramp.ramp_epochs} 輪內由 0 升到 1`,`第 ${run.s6_pinn_ramp.epoch}／${run.s6_pinn_ramp.epochs} 輪`,run.s6_pinn_ramp.physics_mse,run.s6_pinn_ramp.states]];
    const cell=(v,best)=>`<td class="${best?'best':''}"><span class="value">${v}</span></td>`;
    const body=lines.map(([key,label,epoch,phys,st])=>{const r=row(key);return `<tr><th scope="row"><span class="method-name"><i class="dot" style="--method:${fullColor(key)}" aria-hidden="true"></i>${esc(label)}</span><small>${epoch}</small></th>${cell(f(r.macro_esr),key!=='s6_pinn'&&r.macro_esr===Math.min(row('s6_pinn_fixed').macro_esr,row('s6_pinn_ramp').macro_esr))}${cell(f(r.median_esr))}${cell(String(r.files_esr_above_1))}${cell(f(p.failing_median[key]))}${cell(f(st.e_max_median))}${cell(f(st.f_max_median))}${cell(phys?`${sci(phys.c3_kcl)} / ${sci(phys.gre_fast)} / ${sci(phys.gre_slow)}`:'—')}</tr>`;}).join('');
    const circuit=`<tr><th scope="row"><span class="method-name"><i class="dot" style="--method:${fullColor('baseline')}" aria-hidden="true"></i>參考：真實電路（原始參數）</span><small>數值求解，非神經網路</small></th>${cell(f(row('baseline').macro_esr))}${cell(f(row('baseline').median_esr))}${cell(String(row('baseline').files_esr_above_1))}${cell(f(p.failing_median.baseline))}${cell(f(ref.circuit.e_max_median))}${cell(f(ref.circuit.f_max_median))}${cell('—')}</tr>`;
    $('pinn-study-table').innerHTML=`<table class="comparison-table"><caption class="sr-only">S6＋PINN 各版本的完整測試分數、原失敗檔表現與物理狀態幅度</caption><thead><tr><th scope="col">版本 / 選中輪次</th><th scope="col">逐檔平均 ESR ↓</th><th scope="col">ESR 中位數 ↓</th><th scope="col">ESR＞1 檔數 ↓</th><th scope="col">原 ${p.failing_files} 個失敗檔 · ESR 中位數 ↓</th><th scope="col">e 每檔最大值（中位數）</th><th scope="col">f 每檔最大值（中位數）</th><th scope="col">測試物理殘差 C3 / 快 / 慢 ↓</th></tr></thead><tbody>${body}</tbody><tfoot>${circuit}</tfoot></table>`;
    $('pinn-study-intro').textContent=`原版 S6＋PINN 在完整測試集有 ${row('s6_pinn').files_esr_above_1} 檔 ESR＞1，集中在 Threshold 40 與快 Attack 的重壓縮設定。診斷發現網路為壓低物理 loss，把電路狀態幾乎固定在初始值，所以幾乎不壓縮。重新訓練改用與 S6 v3 相同的完整區段批次 TBPTT（50 輪、lr 3e-4、驗證逐檔 macro ESR 選模，規則訓練前固定），兩組只差物理 loss 權重的安排。狀態欄是原版 ${p.state_files} 個失敗檔中，每檔掃描計分區段取最大值後的中位數。`;
    const a=row('s6_pinn_fixed'),b=row('s6_pinn_ramp'),best=row('s6_bounded'),models=t.rows.filter(r=>r.status!=='baseline');
    const lowest=[['pooled_esr',`合併 ESR ${f(a.pooled_esr)}`],['rms_db_mae',`RMS 包絡 ${C.fmt(a.rms_db_mae,'esr')} dB`]].filter(([k])=>a[k]===Math.min(...models.map(r=>r[k]))).map(([,text])=>text);
    $('pinn-study-insight').textContent=`兩組都大幅改善：逐檔平均 ESR 由 ${f(row('s6_pinn').macro_esr)} 降到 ${f(a.macro_esr)}（A）與 ${f(b.macro_esr)}（B），原失敗檔的中位數由 ${f(p.failing_median.s6_pinn)} 降到 ${f(p.failing_median.s6_pinn_fixed)}／${f(p.failing_median.s6_pinn_ramp)}，接近 S6 有界 gain（${f(best.macro_esr)}）。e 已接近真實電路的幅度，表示網路確實開始壓縮。${lowest.length?`A 的${lowest.join('與')}是全部模型最低，`:''}A 最差一檔 ESR ${f(a.max_esr)}，拉高了它的逐檔平均。`;
    $('pinn-study-caveat').textContent=`仍待改善：f 的幅度只有真實電路的一部分；兩組驗證分數在各輪間大幅跳動，A 曾出現物理殘差暴增的輪次。物理 loss 仍是軟約束，只有一個 seed（7），電路參數在訓練中也明顯偏離初始值。`;
  }
  const genColor={s6:'#1d6f45',struct:'#b07a2a',pinn:'#6f3f63'};
  function renderGeneralization(){
    const g=data.full_test?.generalization;
    if(!g)return;
    $('generalization').hidden=false;
    const f=v=>C.fmt(v,'esr'),sci=v=>v.toExponential(1),arm=r=>g.arms[r.arm];
    const test=key=>g.tests.find(t=>t.key===key),done=g.runs.filter(r=>r.status!=='pending');
    const statusText={complete:'完成',replaced:'不採用',pending:'待訓練'};
    const pinnDone=g.scored.some(k=>k.startsWith('t40_pinn_'));
    $('gen-badge').textContent=pinnDone?'P1 完成 · 暫停檢討':'進行中 · 尚無結論';
    $('gen-status').textContent=pinnDone?`${done.length}／${g.runs.length} 次訓練完成，已評分 ${g.scored.length} 個模型。結果不支持「PINN 外插較好」，P2 暫停；爆衝檔已診斷，下一步改設計（見本節最後）。取代卡住 seed 的 M-STRUCT seed 9 仍在訓練。`:`${done.length}／${g.runs.length} 次訓練完成；完整測試已評分 ${g.scored.length} 個模型。`;
    $('gen-intro').textContent=`${g.description}：${g.heldout_cells} 個旋鈕組合從訓練與驗證中移除，只用其餘 ${g.training_cells} 個組合訓練。完整測試仍是全部 ${g.files} 檔，其中 ${g.heldout_files} 檔屬於留出組合。Threshold −40 dB 是壓縮最重的一層，也是原版 S6＋PINN 失敗最集中的地方。`;
    const runRow=r=>{const pending=r.status==='pending',p=r.physics_mse;return `<tr class="${r.status==='replaced'?'muted':''}"><th scope="row"><span class="method-name"><i class="dot" style="--method:${genColor[r.arm]}" aria-hidden="true"></i>${esc(arm(r).code)} · seed ${r.seed}<span class="badge">${statusText[r.status]}</span></span><small>${esc(arm(r).name)}${r.note?` · ${esc(r.note)}`:''}</small></th><td>${pending?'—':`第 ${r.best_epoch}／${r.epochs} 輪`}</td><td>${pending?'—':f(r.validation_macro_esr)}</td><td>${pending?'—':f(r.validation_median_esr)}</td><td>${p?`${sci(p.c3_kcl)} / ${sci(p.gre_fast)} / ${sci(p.gre_slow)}`:'—'}</td><td>${r.frozen_states?'是 · 每輪完全相同':pending||r.arm==='s6'?'—':'否'}</td><td>${g.scored.includes(r.key)?'已評分':pending?'—':r.status==='replaced'?'不採用':'待評分'}</td></tr>`;};
    $('gen-runs').innerHTML=`<table class="comparison-table"><caption class="sr-only">留出實驗各次訓練，依比較組與 seed</caption><thead><tr><th scope="col">比較組 / seed</th><th scope="col">選用輪次</th><th scope="col">驗證 macro ESR ↓</th><th scope="col">驗證 median ESR ↓</th><th scope="col">選用輪物理殘差 C3 / 快 / 慢</th><th scope="col">電路狀態卡住</th><th scope="col">完整測試</th></tr></thead><tbody>${g.runs.map(runRow).join('')}</tbody></table>`;
    const curves=done.filter(r=>r.curve.length),lo=.03,hi=1.5,epochs=Math.max(...curves.map(r=>r.epochs));
    const x=e=>60+(e-1)/(epochs-1)*850,y=v=>20+(Math.log(hi)-Math.log(Math.min(hi,Math.max(lo,v))))/(Math.log(hi)-Math.log(lo))*190;
    $('gen-chart').innerHTML=`<svg viewBox="0 0 960 260" role="img" aria-label="各次訓練每輪的驗證 macro ESR，對數座標"><title>驗證 macro ESR（對數座標），越低越好</title>${[.05,.1,.2,.5,1].map(v=>`<line class="grid" x1="60" x2="910" y1="${y(v)}" y2="${y(v)}"/><text x="48" y="${y(v)+4}" text-anchor="end">${v}</text>`).join('')}${curves.map(r=>`<polyline fill="none" stroke="${genColor[r.arm]}" stroke-width="2" stroke-dasharray="${r.seed===7?'':'6 4'}" opacity="${r.status==='replaced'?.35:.9}" points="${r.curve.map(c=>`${x(c.epoch).toFixed(1)},${y(c.macro_esr).toFixed(1)}`).join(' ')}"><title>${esc(arm(r).code)} seed ${r.seed}</title></polyline><circle cx="${x(r.best_epoch)}" cy="${y(r.validation_macro_esr)}" r="5" fill="${genColor[r.arm]}" stroke="var(--paper)" stroke-width="2"><title>${esc(arm(r).code)} seed ${r.seed} · 選用第 ${r.best_epoch} 輪：${f(r.validation_macro_esr)}</title></circle>`).join('')}${[1,5,10,15,20,25,30].filter(e=>e<=epochs).map(e=>`<text x="${x(e)}" y="235" text-anchor="middle">${e}</text>`).join('')}<text x="910" y="14" text-anchor="end">驗證 macro ESR（對數）/ 輪次</text></svg>`;
    $('gen-chart-note').innerHTML=`${Object.entries(g.arms).map(([k,a])=>`<span class="gen-key" style="--method:${genColor[k]}">${esc(a.code)}</span>`).join('')} 實線 seed 7、虛線 seed 8；圓點是依驗證選用的輪次。每輪分數大幅跳動，所以只看選用點，不看最後一輪。超過 ${hi} 的值畫在上緣。`;
    const ref=test('s6_bounded'),testRows=[['s6_bounded','參考：S6 有界 gain','全部 620 個組合訓練（第 02 節）','s6'],...g.runs.filter(r=>r.status!=='replaced').map(r=>[r.key,`${arm(r).code} · seed ${r.seed}`,'留出 T40 訓練',r.arm])];
    const testRow=([key,label,note,a])=>{const t=test(key);return `<tr><th scope="row"><span class="method-name"><i class="dot" style="--method:${genColor[a]}" aria-hidden="true"></i>${esc(label)}</span><small>${note}</small></th>${t?`<td class="metric-selected"><span class="value">${f(t.heldout_macro_esr)}</span></td><td>${f(t.heldout_median_esr)}</td><td>${t.heldout_files_esr_above_1}</td><td>${f(t.seen_macro_esr)}</td><td>${f(t.seen_median_esr)}</td><td>${f(t.macro_esr)}</td><td>${key==='s6_bounded'?'—':`×${Math.exp(t.heldout_median_log_ratio_vs_full).toFixed(2)}`}</td>`:'<td colspan="7" class="gen-pending">尚未評分</td>'}</tr>`;};
    $('gen-tests').innerHTML=`<table class="comparison-table"><caption class="sr-only">完整測試 ESR，留出組合與已訓練組合分開</caption><thead><tr><th scope="col">模型</th><th scope="col" class="metric-selected">留出 ${g.heldout_files} 檔 · 逐檔平均 ESR ↓</th><th scope="col">留出 · 中位數 ↓</th><th scope="col">留出 · ESR＞1 檔數 ↓</th><th scope="col">已訓練 ${g.files-g.heldout_files} 檔 · 逐檔平均 ↓</th><th scope="col">已訓練 · 中位數 ↓</th><th scope="col">全部 ${g.files} 檔 · 逐檔平均 ↓</th><th scope="col">留出檔 ÷ 參考 · 中位數</th></tr></thead><tbody>${testRows.slice(1).map(testRow).join('')}</tbody><tfoot>${testRow(testRows[0])}</tfoot></table>`;
    const s6=g.tests.filter(t=>t.key.startsWith('t40_s6_')),range=k=>`${f(Math.min(...s6.map(t=>t[k])))}–${f(Math.max(...s6.map(t=>t[k])))}`;
    const pinn=g.tests.filter(t=>t.key.startsWith('t40_pinn_')),prange=k=>`${f(Math.min(...pinn.map(t=>t[k])))}–${f(Math.max(...pinn.map(t=>t[k])))}`;
    $('gen-insight').textContent=s6.length?`留出 −40 dB 之後，M-S6 在這 ${g.heldout_files} 檔的逐檔平均 ESR 是 ${range('heldout_macro_esr')}，約是全組合參考模型（${f(ref.heldout_macro_esr)}）的 ${(Math.min(...s6.map(t=>t.heldout_macro_esr))/ref.heldout_macro_esr).toFixed(1)}–${(Math.max(...s6.map(t=>t.heldout_macro_esr))/ref.heldout_macro_esr).toFixed(1)} 倍。${pinn.length?`M-PINN 更差：${prange('heldout_macro_esr')}，ESR＞1 有 ${pinn.map(t=>t.heldout_files_esr_above_1).join('、')} 檔（M-S6 ${s6.map(t=>t.heldout_files_esr_above_1).join('、')} 檔）。`:''}已訓練組合上各模型差距小。這一層本來就難：即使全部組合都訓練過，−40 dB 的誤差也是其他組合的 ${(ref.heldout_macro_esr/ref.seen_macro_esr).toFixed(1)} 倍。`:'尚無完整測試結果。';
    const stuck=g.runs.find(r=>r.frozen_states);
    $('gen-caveat').textContent=`${pinnDone?'停止規則要求兩個 seed 的 M-PINN 都較差，seed 8 的中位數略好，字面上沒有觸發；但兩個 seed 方向相反、macro ESR 與爆衝檔數都最差，整體不支持 PINN 外插較好，因此決定先停下來檢討。':'M-PINN 與 M-STRUCT 尚未做完整測試。'}驗證分數不能拿來比，因為驗證集不含留出組合。${stuck?`M-STRUCT seed ${stuck.seed} 的電路狀態 ${stuck.epochs} 輪都沒動（物理權重 0 時 e 收不到梯度），不列入比較，改以 seed 9 重跑。`:''}`;
    renderGenCompare(g);renderGenDiagnosis(g);
    $('gen-rule').textContent=`主要指標是留出檔的逐檔 log(ESR_PINN) − log(ESR_S6)，取中位數後跨 seed 合併。停止規則：${g.stopping_rule}`;
  }
  const genName=key=>{const m=key.match(/^t40_(\w+)_s(\d+)$/);return m?`${data.full_test.generalization.arms[m[1]].code} s${m[2]}`:key;};
  function renderGenCompare(g){
    const f=v=>(v>=0?'+':'')+v.toFixed(2),rows=g.comparisons||[];
    $('gen-compare').innerHTML=rows.length?`<table class="comparison-table"><caption class="sr-only">成對 log ESR 差，負值代表前者較好</caption><thead><tr><th scope="col">比較</th><th scope="col">檔案</th><th scope="col" class="metric-selected">log ESR 差中位數</th><th scope="col">倍率</th><th scope="col">95% 區間（檔案 bootstrap）</th><th scope="col">前者較好的檔案比例</th></tr></thead><tbody>${rows.map(c=>`<tr class="${c.heldout?'':'muted'}"><th scope="row">${esc(genName(c.a))} − ${esc(genName(c.b))}</th><td>${c.heldout?'留出':'已訓練'} ${c.files}</td><td class="metric-selected ${c.heldout&&c.high<0?'best':''}"><span class="value">${f(c.median)}</span></td><td>×${Math.exp(c.median).toFixed(2)}</td><td>[${f(c.low)}, ${f(c.high)}]</td><td>${(c.better_share*100).toFixed(0)}%</td></tr>`).join('')}</tbody></table>`:'';
    $('gen-compare-note').textContent='每檔先算 log(ESR_前者) − log(ESR_後者)，再取中位數；負值代表前者較好，倍率是 exp(中位數)。區間只反映檔案間的變異，不含 seed 間的變異；兩個 seed 方向相反時，以 seed 為單位的結論仍不確定。';
  }
  function renderGenDiagnosis(g){
    const d=g.diagnosis;if(!d)return;$('gen-diagnosis').hidden=false;
    const H=d.groups.heldout,T=d.groups.trained,n=v=>v==null?'—':v.toFixed(2),db=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(1)+' dB',sci=v=>v==null?'—':v.toExponential(1);
    const label={pinn_s7:'M-PINN s7',pinn_s8:'M-PINN s8',struct_s7:'M-STRUCT s7',s6_s8:'M-S6 s8',pinn_full:'參考：全組合 PINN（看過 −40 dB）'};
    $('gen-diagnosis-intro').textContent=`${d.files} 個 PINN 爆衝檔（−40 dB 留出，任一 seed ESR＞1），與相同 Attack／Release／Ratio 的 −30 dB 檔（已訓練）對照；表中都是逐檔中位數。爆衝集中在 Attack 0、長 Release、高 Ratio，目標 RMS 只有 0.001–0.004。`;
    $('gen-diagnosis-table').innerHTML=`<table class="comparison-table"><thead><tr><th scope="col">模型</th><th scope="col">−40 dB ESR</th><th scope="col">輸出電平 − 目標</th><th scope="col">壓縮量（目標 ${db(H.pinn_s7.gain_db_target)}）</th><th scope="col">e 最大值 −30 → −40 dB</th><th scope="col">C3 殘差 −30 → −40 dB</th></tr></thead><tbody>${Object.keys(d.models).map(m=>`<tr><th scope="row">${esc(label[m]||m)}</th><td>${n(H[m].esr)}</td><td>${db(H[m].level_error_db)}</td><td>${db(H[m].gain_db_pred)}</td><td>${H[m].e_max==null?'—':`${n(T[m].e_max)} → ${n(H[m].e_max)}`}</td><td>${H[m].c3_residual==null||m.startsWith('struct')?'—':`${sci(T[m].c3_residual)} → ${sci(H[m].c3_residual)}`}</td></tr>`).join('')}</tbody></table>`;
    $('gen-diagnosis-insight').textContent=`PINN 在 −40 dB 壓縮不足：輸出比目標大 ${db(H.pinn_s7.level_error_db)}／${db(H.pinn_s8.level_error_db)}。原因是網路直接預測的 e 沒有隨 Threshold 外插（${n(T.pinn_s7.e_max)}→${n(H.pinn_s7.e_max)}），而看過 −40 dB 的同架構模型會拉到 ${n(H.pinn_full.e_max)}，接近真實電路約 0.38。C3 殘差在留出檔升高 20–30 倍，表示電路方程察覺狀態錯了，但物理只是訓練時的軟懲罰，推論時沒有作用。同一套公式在看過 −40 dB 時電平誤差只有 ${db(H.pinn_full.level_error_db)}，所以問題在架構，不在公式。`;
    const P=d.parameters,keys=Object.keys(P.pinn_s7);
    $('gen-diagnosis-params').textContent=`電路參數也不可辨識：${keys.filter(k=>Math.abs(Math.log(P.pinn_s7[k].value/P.pinn_s7[k].initial))>Math.log(2)).map(k=>`${k} ${P.pinn_s7[k].initial.toPrecision(2)}→${['pinn_s7','pinn_s8','pinn_full'].map(m=>P[m][k].value.toPrecision(2)).join('／')}`).join('；')}（初始→T40 PINN s7／s8／全組合 PINN）。參數漂移超過 2 倍而且每個 seed 不同，變成額外的擬合自由度，不再代表實體電路。`;
    $('gen-next').textContent='新設計「S6＋可微分電路」：e／f／s 不再由網路預測，而由電路方程逐步積分（硬約束），Threshold 只經由電路的 taper 影響壓縮，外插行為由物理決定；網路只做不看旋鈕的有界修正（偵測器增益約 ±4 dB、時間常數、輸出音色），電路參數加強先驗以保持可辨識。主要風險是電路本身的結構誤差：即使用全部組合擬合，純電路在 −40 dB 的 ESR 仍有 0.39，高於留出 S6 的 0.31，因此新比較組包含「純電路」作為下限。設計已寫成計畫，尚未實作或訓練。';
  }
  // Audio stays in separate short licensed WAVs; only compact waveform envelopes live in the report.
  const audition=data.listening;
  const listenAudio=$('listen-audio'), listenCanvas=$('listen-waveform'), listenCtx=listenCanvas.getContext('2d');
  let listenRecord=0, listenTrack='reference_wet', listenFrame=0;
  function trackEntries(){return Object.entries(audition.records[listenRecord].audio);}
  function auditionLabel(record){
    if(record.title){
      const span=record.start_seconds!==undefined?` · ${record.start_seconds.toFixed(1)}–${record.stop_seconds.toFixed(1)} s`:'';
      return `${record.title}${span} · ${record.id}`;
    }
    const c=record.conditioning||{};
    return `${record.id} · Attack ${c.attack ?? '—'} · Release ${c.release ?? '—'} · ${c.ratio ?? '—'}:1 · ${c.threshold ?? '—'} dB`;
  }
  function drawListeningWave(){
    const rect=listenCanvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,rect.width),h=Math.max(1,rect.height);
    if(listenCanvas.width!==Math.round(w*dpr)||listenCanvas.height!==Math.round(h*dpr)){listenCanvas.width=Math.round(w*dpr);listenCanvas.height=Math.round(h*dpr);}
    const ctx=listenCtx,css=getComputedStyle(document.documentElement),record=audition.records[listenRecord],reference=record.waveform.reference_wet,dry=record.waveform.reference_dry,selected=record.waveform[listenTrack];
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle=css.getPropertyValue('--paper');ctx.fillRect(0,0,w,h);
    const left=45,right=w-16,top=18,plotBottom=h*.61,waveMid=(top+plotBottom)/2,residualTop=h*.72,residualBottom=h-28,residualMid=(residualTop+residualBottom)/2,plotWidth=right-left;
    ctx.strokeStyle=css.getPropertyValue('--line');ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,waveMid);ctx.lineTo(right,waveMid);ctx.moveTo(left,plotBottom);ctx.lineTo(right,plotBottom);ctx.moveTo(left,residualMid);ctx.lineTo(right,residualMid);ctx.stroke();
    const traces=[dry,reference,...(listenTrack==='reference_dry'||listenTrack==='reference_wet'?[]:[selected])];
    const peak=Math.max(.001,...traces.flatMap(wave=>[...wave.low.map(Math.abs),...wave.high.map(Math.abs)]));
    const drawPeaks=(wave,color,width)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(let i=0;i<wave.low.length;i++){const x=left+(i+.5)/wave.low.length*plotWidth;ctx.moveTo(x,waveMid-wave.high[i]/peak*(plotBottom-top)*.46);ctx.lineTo(x,waveMid-wave.low[i]/peak*(plotBottom-top)*.46);}ctx.stroke();};
    drawPeaks(dry,'#8a7770',1.15);drawPeaks(reference,'#168073',1.4);if(listenTrack!=='reference_dry'&&listenTrack!=='reference_wet')drawPeaks(selected,'#4777ae',1.35);
    const residualMax=Math.max(.001,...selected.residual_low.map(Math.abs),...selected.residual_high.map(Math.abs)),difference=selected.diff_rms,diffMax=Math.max(.001,...difference);
    ctx.fillStyle='rgba(214,122,55,.20)';for(let i=0;i<difference.length;i++){const x=left+i/difference.length*plotWidth,bar=difference[i]/diffMax*(residualBottom-residualTop)*.43;ctx.fillRect(x,residualMid-bar,Math.max(1,plotWidth/difference.length),bar*2);}
    ctx.strokeStyle='#d67a37';ctx.lineWidth=1.2;ctx.beginPath();for(let i=0;i<selected.residual_low.length;i++){const x=left+(i+.5)/selected.residual_low.length*plotWidth;ctx.moveTo(x,residualMid-selected.residual_high[i]/residualMax*(residualBottom-residualTop)*.43);ctx.lineTo(x,residualMid-selected.residual_low[i]/residualMax*(residualBottom-residualTop)*.43);}ctx.stroke();
    ctx.fillStyle=css.getPropertyValue('--muted');ctx.font='11px Consolas, monospace';ctx.fillText('振幅',6,24);ctx.fillText('殘差',6,residualTop+10);ctx.fillText(`共用峰值 ±${peak.toFixed(3)}`,left+5,top+10);ctx.fillText(`殘差尺度 ±${residualMax.toFixed(4)} · RMS max ${diffMax.toFixed(4)}`,left+5,residualTop+10);
    for(let t=0;t<=4;t++){const x=left+t/4*plotWidth;ctx.fillText(`${(record.seconds*t/4).toFixed(1)}s`,Math.max(left,x-13),h-8);}
    const progress=Math.max(0,Math.min(1,(listenAudio.currentTime||0)/(listenAudio.duration||record.seconds)));ctx.strokeStyle=css.getPropertyValue('--ink');ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(left+progress*plotWidth,top);ctx.lineTo(left+progress*plotWidth,residualBottom);ctx.stroke();
    if(!listenAudio.paused)listenFrame=requestAnimationFrame(drawListeningWave);
  }
  function updateListenTime(){
    const duration=Number.isFinite(listenAudio.duration)?listenAudio.duration:0;
    $('listen-time').textContent=`${(listenAudio.currentTime||0).toFixed(3)} s`;
    $('listen-seek').max=String(Math.max(duration,.001));$('listen-seek').value=String(Math.min(listenAudio.currentTime||0,duration));
    $('listen-play').textContent=listenAudio.paused?'播放':'暫停';
  }
  function setListeningSource(preserve=true){
    const prior=preserve?listenAudio.currentTime:0,wasPlaying=!listenAudio.paused;
    const record=audition.records[listenRecord],entry=record.audio[listenTrack];listenAudio.pause();
    listenAudio.onloadedmetadata=()=>{listenAudio.currentTime=Math.min(prior,Math.max(0,listenAudio.duration-.001));updateListenTime();drawListeningWave();$('listen-status').textContent=`已就緒：${record.title?record.title+' · ':''}${entry.label} · ${record.seconds.toFixed(3)} 秒。所有音軌維持其原始振幅，未做音量正規化。`;if(wasPlaying)void playListening();};
    listenAudio.src=entry.path;listenAudio.load();$('listen-status').textContent='正在載入試聽 WAV…';updateListenTime();drawListeningWave();
  }
  async function playListening(){
    try{await listenAudio.play();$('listen-status').textContent='播放中；切換音軌將保留相同時間位置。';cancelAnimationFrame(listenFrame);drawListeningWave();}catch(error){$('listen-status').textContent=`無法播放：${error.message}`;}
  }
  function renderListening(){
    if(!audition||!audition.records?.length)return;
    $('listen-record').innerHTML=audition.records.map((r,i)=>`<option value="${i}">${esc(auditionLabel(r))}</option>`).join('');$('listen-record').value=String(listenRecord);
    $('listen-track').innerHTML=trackEntries().map(([key,v])=>`<option value="${key}">${esc(v.label)}</option>`).join('');$('listen-track').value=listenTrack;
    $('listen-download').href=audition.records[listenRecord].audio[listenTrack].path;$('listen-download').download=`${audition.records[listenRecord].id}__${listenTrack}.wav`;
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
  $('profile-model').addEventListener('change',()=>{state.profile=$('profile-model').value;const selected=rows(),summary=C.summarize(selected,$('aggregate').value,study.protocol.esr_floor);renderDiagnostics(selected,summary);});
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
  if(data.full_test)$('export-full-test').addEventListener('click',()=>download(C.csv([['scope','files','model','selection_status',...fullColumns.map(([k])=>k),'max_esr','files_esr_above_10','audible_macro_esr','source','sha256'],...data.full_test.rows.map(r=>['full_test',data.full_test.files,r.key,r.status,...fullColumns.map(([k])=>r[k]),r.max_esr,r.files_esr_above_10,r.audible_macro_esr,r.source,r.sha256])]),'cl1b-full-test-summary.csv','text/csv;charset=utf-8'));
  if(data.full_test?.generalization)$('export-generalization').addEventListener('click',()=>{
    const g=data.full_test.generalization,cols=['heldout_macro_esr','heldout_median_esr','heldout_files_esr_above_1','seen_macro_esr','seen_median_esr','macro_esr','heldout_median_log_ratio_vs_full'];
    download(C.csv([['scheme','key','arm','seed','status','best_epoch','validation_macro_esr','frozen_states',...cols],
      ...g.runs.map(r=>{const t=g.tests.find(x=>x.key===r.key);return [g.scheme,r.key,r.arm,r.seed,r.status,r.best_epoch??'',r.validation_macro_esr??'',r.frozen_states??'',...cols.map(k=>t?t[k]:'')];}),
      ...g.tests.filter(t=>t.key==='s6_bounded').map(t=>[g.scheme,t.key,'reference','','full_data','','','',...cols.map(k=>t[k])])]),'cl1b-t40-holdout.csv','text/csv;charset=utf-8');
  });
  $('export-json').addEventListener('click',()=>download(JSON.stringify(data,null,2),'cl1b-comparison.json','application/json'));
  $('listen-record').addEventListener('change',()=>{listenRecord=Number($('listen-record').value);listenTrack='reference_wet';renderListening();setListeningSource(false);});
  $('listen-track').addEventListener('change',()=>{listenTrack=$('listen-track').value;renderListening();setListeningSource(true);});
  $('listen-play').addEventListener('click',()=>{if(listenAudio.paused)void playListening();else listenAudio.pause();});
  $('listen-volume').addEventListener('input',()=>{listenAudio.volume=Number($('listen-volume').value);});
  $('listen-loop').addEventListener('change',()=>{listenAudio.loop=$('listen-loop').checked;});
  $('listen-seek').addEventListener('input',()=>{if(listenAudio.readyState>=1){listenAudio.currentTime=Number($('listen-seek').value);updateListenTime();}});
  listenCanvas.addEventListener('click',event=>{if(listenAudio.readyState<1)return;const rect=listenCanvas.getBoundingClientRect(),left=45,right=rect.width-16,fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-left)/(right-left)));listenAudio.currentTime=fraction*listenAudio.duration;updateListenTime();drawListeningWave();});
  listenCanvas.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key)||listenAudio.readyState<1)return;event.preventDefault();listenAudio.currentTime=Math.max(0,Math.min(listenAudio.duration,listenAudio.currentTime+(event.key==='ArrowRight'?.1:-.1)));updateListenTime();drawListeningWave();});
  $('listening').addEventListener('toggle',()=>{if($('listening').open)drawListeningWave();});
  window.addEventListener('resize',()=>{if($('listening').open)drawListeningWave();});
  listenAudio.addEventListener('play',()=>{cancelAnimationFrame(listenFrame);drawListeningWave();updateListenTime();});
  listenAudio.addEventListener('pause',()=>{cancelAnimationFrame(listenFrame);drawListeningWave();updateListenTime();});
  listenAudio.addEventListener('timeupdate',updateListenTime);
  listenAudio.addEventListener('error',()=>{$('listen-status').textContent='無法載入試聽 WAV。請確認網頁與試聽音檔已完整部署。';});
  function setTheme(theme){document.documentElement.dataset.theme=theme;$('theme').textContent=theme==='dark'?'淺色模式':'深色模式';$('theme').setAttribute('aria-label',`切換${theme==='dark'?'淺':'深'}色模式`);}
  try{setTheme(localStorage.getItem('ht1b-report-theme')==='dark'?'dark':'light');}catch(_){setTheme('light');}
  $('theme').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';setTheme(theme);try{localStorage.setItem('ht1b-report-theme',theme);}catch(_){}});
  function revealAnchor(){
    const id=window.location.hash.slice(1),target=$(id);
    if(target&&target.tagName==='DETAILS')target.open=true;
  }
  window.addEventListener('hashchange',revealAnchor);
  render();renderFullTest();renderGeneralization();renderHistory();renderCoverage();renderListening();setListeningSource(false);revealAnchor();
})();
