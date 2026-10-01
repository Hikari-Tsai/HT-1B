// Extended evaluation and personal listening trials. No remote services.
const storageKey=`ht1b-analysis-v1-${data.benchmark_id}`;
let canStore=true;
let annotations={};
let trials=[];
let currentTrial=null;
let blindLoadVersion=0;
let blindLoaded=null;
let blindActive=null;
let heardSeconds=0;
let lastHeardTime=0;
const blindAudio=$("blind-audio");
const letters=["A","B","C","D","E"];
const permittedModels=["pure_algorithm","author_pretrained","reference_wet","reference_dry",...steps.map(s=>`ours_${s}`)];

function modelName(key){
  if(key==="pure_algorithm")return "純演算法";
  if(key==="author_pretrained")return "RiccardoVib";
  if(key==="reference_wet")return "隱藏 Wet 參考";
  if(key==="reference_dry")return "Dry 對照";
  return `PhysicsNeMo / ${Number(key.slice(5)).toLocaleString()} 步`;
}
function validRegion(region,record){return region && Number.isInteger(region.start) && Number.isInteger(region.stop) && region.start>=0 && region.start<region.stop && region.stop<=record.frames.counts.length;}
function restoreAnalysis(){
  try{
    const saved=JSON.parse(localStorage.getItem(storageKey)||"null");
    if(saved && saved.benchmark_id===data.benchmark_id){
      for(const r of data.records){for(const kind of ["attack","release"]){const region=saved.annotations?.[r.id]?.[kind];if(validRegion(region,r)){annotations[r.id]??={};annotations[r.id][kind]=region;}}}
      trials=(Array.isArray(saved.trials)?saved.trials:[]).filter(t=>data.records.some(r=>r.id===t.record_id) && steps.includes(t.step) && typeof t.completed_at==="string" && Array.isArray(t.candidates) && t.candidates.length===5 && new Set(t.candidates.map(c=>c.model)).size===5 && t.candidates.every(c=>permittedModels.includes(c.model)&&Number.isInteger(c.score)&&c.score>=0&&c.score<=100));
    }
    localStorage.setItem(storageKey,JSON.stringify({benchmark_id:data.benchmark_id,annotations,trials}));
  }catch{canStore=false;}
}
function persistAnalysis(){
  try{localStorage.setItem(storageKey,JSON.stringify({benchmark_id:data.benchmark_id,annotations,trials}));canStore=true;}
  catch{canStore=false;}
  renderTrialSummary();
}
function renderTrialSummary(){
  $("blind-summary").textContent=`已完成 ${trials.length} 輪個人評分。${canStore?"已啟用目前瀏覽器的本機儲存；請匯出以備份。":"瀏覽器未允許本機儲存；目前記錄只在本次開啟中，請立即匯出備份。"}`;
  $("blind-start").textContent=currentTrial&&!currentTrial.completed_at?"繼續當輪盲測":"開始盲測";
  $("blind-export").disabled=trials.length===0;
  $("blind-export-dialog").disabled=trials.length===0;
}
function localMetrics(record,key,start,stop){
  const f=record.frames;
  const sum=a=>a.slice(start,stop).reduce((x,y)=>x+y,0);
  const count=sum(f.counts),targetEnergy=sum(f.target_energy),squared=sum(f.errors[key].squared_error);
  let rms=0;for(let i=start;i<stop;i++)rms+=Math.abs(f.rms_db[key][i]-f.rms_db.reference_wet[i])*f.counts[i];
  return {mse:squared/count,mae:sum(f.errors[key].absolute_error)/count,esr:targetEnergy>1e-20?squared/targetEnergy:null,rms_db_mae:rms/count,samples:count};
}
function renderRegions(){
  const r=data.records[state.record],kind=$("region-kind").value,region=annotations[r.id]?.[kind];
  $("region-start").value=region?r.frames.edges_seconds[region.start].toFixed(6):"";
  $("region-end").value=region?r.frames.edges_seconds[region.stop].toFixed(6):"";
  $("region-status").textContent=region?`${kind==="attack"?"Attack":"Release"}：${r.frames.edges_seconds[region.start].toFixed(3)}–${r.frames.edges_seconds[region.stop].toFixed(3)} s；手動標記，${canStore?"已儲存於本機瀏覽器":"僅保留於本次開啟，請匯出"}。`:"尚未標記此區段。若找不到完整的壓縮進入／恢復過程，請保持未標記。";
  const body=["attack","release"].map(type=>{
    const saved=annotations[r.id]?.[type];
    if(!saved)return `<tr><td>${type==="attack"?"Attack":"Release"}</td><td colspan="5">未標記 / 無法判定</td></tr>`;
    return models().map(m=>{const v=localMetrics(r,m.key,saved.start,saved.stop);return `<tr><td>${type==="attack"?"Attack":"Release"}<small class="region-range">${r.frames.edges_seconds[saved.start].toFixed(3)}–${r.frames.edges_seconds[saved.stop].toFixed(3)} s</small></td><td>${m.name}</td><td>${fmt(v.mse,"mse")}</td><td>${fmt(v.mae,"mae")}</td><td>${fmt(v.esr,"esr")}</td><td>${fmt(v.rms_db_mae,"rms_db_mae")}</td></tr>`;}).join("");
  }).join("");
  $("region-results").innerHTML=`<table><caption>目前設定的手動標記區段 · ${state.step.toLocaleString()} 步</caption><thead><tr><th scope="col">區段</th><th scope="col">模型</th><th scope="col">MSE</th><th scope="col">MAE</th><th scope="col">ESR</th><th scope="col">RMS MAE / dB</th></tr></thead><tbody>${body}</tbody></table>`;
}
function renderExtended(){renderRegions();renderTrialSummary();}
function applyRegion(){
  const r=data.records[state.record],startInput=$("region-start"),endInput=$("region-end"),start=Number(startInput.value),end=Number(endInput.value),edges=r.frames.edges_seconds;
  if(startInput.value===""||endInput.value===""||!Number.isFinite(start)||!Number.isFinite(end)||start<0||start>=end||end>edges.at(-1)+1e-6){$("region-status").textContent="請填入有效的起點與終點：0 ≤ 起點 < 終點 ≤ 1.999333 秒。";return;}
  const nearest=value=>edges.reduce((best,x,i)=>Math.abs(x-value)<Math.abs(edges[best]-value)?i:best,0);
  const region={start:nearest(start),stop:nearest(end)};
  if(!validRegion(region,r)){$("region-status").textContent="區間短於分析格點，請選擇至少一格（約 10 ms）。";return;}
  annotations[r.id]??={};annotations[r.id][$("region-kind").value]=region;
  persistAnalysis();state.wave="rms";renderListening();
}
function exportRegions(){
  const results=[];
  data.records.forEach(r=>["attack","release"].forEach(kind=>{
    const region=annotations[r.id]?.[kind];if(!region)return;
    results.push({record:r.id,kind,start_frame:region.start,stop_frame:region.stop,start_seconds:r.frames.edges_seconds[region.start],stop_seconds:r.frames.edges_seconds[region.stop],metrics:Object.fromEntries(Object.keys(r.metrics).map(key=>[key,localMetrics(r,key,region.start,region.stop)]))});
  }));
  download(JSON.stringify({benchmark_id:data.benchmark_id,protocol:data.metric_protocol,selection:"manual",regions:results},null,2),"cl1b-manual-regions.json","application/json");
}
function drawRms(){
  const r=data.records[state.record],f=r.frames,canvas=$("waveform"),ctx=canvas.getContext("2d"),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,left=35,top=10,bottom=h-15,duration=f.edges_seconds.at(-1);
  const options=[{key:"reference_dry",color:"reference"},{key:"reference_wet",color:"ink"},...models()];
  const upper=Math.max(0,Math.ceil(Math.max(...options.flatMap(t=>f.rms_db[t.key]))/10)*10);
  const y=v=>top+(upper-v)/(upper+100)*(bottom-top),x=time=>time/duration*w;
  for(const kind of ["attack","release"]){const a=annotations[r.id]?.[kind];if(a){ctx.globalAlpha=.12;ctx.fillStyle=cssColor(kind==="attack"?"ours":"author");ctx.fillRect(x(f.edges_seconds[a.start]),0,x(f.edges_seconds[a.stop])-x(f.edges_seconds[a.start]),h);ctx.globalAlpha=1;}}
  ctx.font="10px Consolas,monospace";ctx.fillStyle=cssColor("muted");ctx.strokeStyle=cssColor("line");
  [0,-20,-40,-60,-80,-100].forEach(db=>{ctx.beginPath();ctx.moveTo(left,y(db));ctx.lineTo(w,y(db));ctx.stroke();ctx.fillText(String(db),0,y(db)+3);});
  options.sort((a,b)=>(a.key===state.track?1:0)-(b.key===state.track?1:0));
  options.forEach(t=>{ctx.strokeStyle=cssColor(t.color);ctx.globalAlpha=t.key===state.track?1:.5;ctx.lineWidth=t.key===state.track?2.3:1;ctx.setLineDash(t.key==="reference_dry"?[3,3]:[]);ctx.beginPath();f.rms_db[t.key].forEach((db,i)=>{const px=x((f.edges_seconds[i]+f.edges_seconds[i+1])/2),py=y(db);if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);});ctx.stroke();});ctx.setLineDash([]);ctx.globalAlpha=1;
  $("wave-caption").textContent="RMS 包絡 · Dry 虛線，Wet 墨色，所選音軌加亮；色帶為手動標記";
  $("wave-scale").textContent=`−100 至 ${upper} dBFS / 10 ms`;
  canvas.setAttribute("aria-label","10 毫秒 RMS 包絡，dBFS 刻度；局部誤差在下方表格提供文字數值");
}

function shuffle(items){
  const shuffled=[...items];for(let i=shuffled.length-1;i>0;i--){
    // Rejection sampling avoids modulo bias and does not reuse a fixed mapping.
    const limit=Math.floor(2**32/(i+1))*(i+1);let n;
    do{n=crypto.getRandomValues(new Uint32Array(1))[0];}while(n>=limit);
    const j=n%(i+1);[shuffled[i],shuffled[j]]=[shuffled[j],shuffled[i]];
  }return shuffled;
}
function createTrial(){
  currentTrial={id:crypto.randomUUID?crypto.randomUUID():`${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0]}`,record_id:data.records[state.record].id,step:state.step,started_at:new Date().toISOString(),reference_heard:false,
    candidates:shuffle(["pure_algorithm",`ours_${state.step}`,"author_pretrained","reference_wet","reference_dry"]).map((model,i)=>({label:letters[i],model,score:null,heard:false})),volume:blindAudio.volume};
  stopBlind();blindActive=null;blindAudio.removeAttribute("src");blindAudio.load();
  renderBlind();renderTrialSummary();
}
function openBlind(){
  audio.pause();if(!currentTrial||currentTrial.completed_at)createTrial();
  renderBlind();$("blind-dialog").showModal();
}
function renderBlind(){
  const r=data.records.find(r=>r.id===currentTrial.record_id),done=Boolean(currentTrial.completed_at);
  $("blind-context").textContent=`設定 ${data.records.indexOf(r)+1} / ${knobsText(r)} · 比較 ${currentTrial.step.toLocaleString()} 步 · ${done?"已完成，可查看身份":"本輪設定已固定"}`;
  $("blind-candidates").innerHTML=currentTrial.candidates.map((c,i)=>`<div class="blind-candidate"><button data-blind-index="${i}" aria-pressed="false">播放 ${c.label}</button><span id="blind-identity-${i}">${done?modelName(c.model):`匿名音軌 ${c.label}`}</span><label>相似度 <input data-score-index="${i}" type="number" min="0" max="100" step="1" inputmode="numeric" placeholder="未評分" value="${c.score??""}" aria-label="音軌 ${c.label} 的相似度分數" ${done?"disabled":""}> / 100</label><span id="heard-${i}" class="muted">${c.heard?"已試聽":"尚未試聽"}</span></div>`).join("");
  $("blind-submit").hidden=done;$("blind-new").hidden=!done;
  $("blind-status").textContent=done?"評分已保存；身份已揭示。可匯出記錄，或開始新的隨機排列。":"請先聽參考與每個代號至少 0.5 秒，再填寫所有分數；不提供預設分數。";
  updateBlindEligibility();syncBlindPlayback();
}
function updateBlindEligibility(){
  $("blind-submit").disabled=!currentTrial||Boolean(currentTrial.completed_at)||!currentTrial.reference_heard||!currentTrial.candidates.every(c=>c.heard&&Number.isInteger(c.score)&&c.score>=0&&c.score<=100);
}
function stopBlind(){
  blindLoadVersion++;blindAudio.pause();
  if(blindLoaded){blindAudio.removeEventListener("loadedmetadata",blindLoaded);blindLoaded=null;}
}
async function playBlindTrack(index){
  const time=blindAudio.currentTime||0;stopBlind();const version=blindLoadVersion;
  blindActive=index;heardSeconds=0;lastHeardTime=time;
  const r=data.records.find(r=>r.id===currentTrial.record_id);
  const key=index==="reference"?"reference_wet":currentTrial.candidates[index].model;
  blindLoaded=async()=>{
    if(version!==blindLoadVersion)return;
    blindAudio.currentTime=Math.min(time,Math.max(0,blindAudio.duration-.001));lastHeardTime=blindAudio.currentTime;
    try{await blindAudio.play();if(version===blindLoadVersion)$("blind-status").textContent=`正在播放${index==="reference"?"參考 Wet":`匿名音軌 ${letters[index]}`}；所有音軌使用相同播放音量。`;}
    catch{if(version===blindLoadVersion)$("blind-status").textContent="無法播放音訊，請再按一次播放；若仍失敗，請以支援 WAV 的 Chrome 或 Edge 開啟此頁。";}
    syncBlindPlayback();
  };
  blindAudio.addEventListener("loadedmetadata",blindLoaded,{once:true});
  blindAudio.src=`data:audio/wav;base64,${r.audio[key]}`;blindAudio.load();
  $("blind-status").textContent="載入匿名音軌…";syncBlindPlayback();
}
function syncBlindPlayback(){
  $("blind-time").textContent=`${(blindAudio.currentTime||0).toFixed(3)} s`;
  $("blind-seek").value=blindAudio.currentTime||0;
  document.querySelectorAll("[data-blind-index]").forEach(b=>b.setAttribute("aria-pressed",Number(b.dataset.blindIndex)===blindActive&&!blindAudio.paused));
  $("blind-reference").setAttribute("aria-pressed",blindActive==="reference"&&!blindAudio.paused);
}
function submitTrial(){
  updateBlindEligibility();if($("blind-submit").disabled)return;
  stopBlind();currentTrial.completed_at=new Date().toISOString();currentTrial.volume=blindAudio.volume;
  trials.push(JSON.parse(JSON.stringify(currentTrial)));persistAnalysis();renderBlind();
}
function exportTrials(){download(JSON.stringify({protocol:"ht1b-personal-blind-v1",benchmark_id:data.benchmark_id,scale:"0-100 similarity to wet, higher is more similar",formal_mushra:false,normalization:"none; shared playback volume",trials},null,2),"cl1b-personal-blind-ratings.json","application/json");}

$("region-kind").addEventListener("change",renderRegions);
$("region-start-current").addEventListener("click",()=>{$("region-start").value=(audio.currentTime||0).toFixed(6);});
$("region-end-current").addEventListener("click",()=>{$("region-end").value=(audio.currentTime||0).toFixed(6);});
$("region-apply").addEventListener("click",applyRegion);
$("region-clear").addEventListener("click",()=>{const r=data.records[state.record];if(annotations[r.id])delete annotations[r.id][$("region-kind").value];persistAnalysis();renderListening();});
$("export-regions").addEventListener("click",exportRegions);
$("blind-start").addEventListener("click",openBlind);
$("blind-close").addEventListener("click",()=>{$("blind-dialog").close();});
$("blind-dialog").addEventListener("close",()=>{stopBlind();renderTrialSummary();});
$("blind-reference").addEventListener("click",()=>{void playBlindTrack("reference");});
$("blind-pause").addEventListener("click",()=>{stopBlind();syncBlindPlayback();});
$("blind-volume").addEventListener("input",e=>{blindAudio.volume=Number(e.target.value);});
$("blind-loop").addEventListener("change",e=>{blindAudio.loop=e.target.checked;});
$("blind-seek").addEventListener("input",e=>{if(blindAudio.readyState>=1){blindAudio.currentTime=Number(e.target.value);lastHeardTime=blindAudio.currentTime;syncBlindPlayback();}});
$("blind-submit").addEventListener("click",submitTrial);
$("blind-new").addEventListener("click",createTrial);
$("blind-export").addEventListener("click",exportTrials);
$("blind-export-dialog").addEventListener("click",exportTrials);
$("blind-candidates").addEventListener("click",e=>{const b=e.target.closest("[data-blind-index]");if(b)void playBlindTrack(Number(b.dataset.blindIndex));});
$("blind-candidates").addEventListener("input",e=>{const input=e.target.closest("[data-score-index]");if(!input||currentTrial.completed_at)return;const score=Number(input.value);currentTrial.candidates[Number(input.dataset.scoreIndex)].score=input.value!==""&&Number.isInteger(score)&&score>=0&&score<=100?score:null;updateBlindEligibility();});
blindAudio.addEventListener("timeupdate",()=>{
  const time=blindAudio.currentTime||0,delta=time-lastHeardTime;
  if(!blindAudio.paused&&!blindAudio.seeking&&delta>0&&delta<.6)heardSeconds+=delta;
  lastHeardTime=time;
  if(currentTrial&&!currentTrial.completed_at&&heardSeconds>=.5){
    if(blindActive==="reference")currentTrial.reference_heard=true;
    else if(Number.isInteger(blindActive)){currentTrial.candidates[blindActive].heard=true;$("heard-"+blindActive).textContent="已試聽";}
    updateBlindEligibility();
  }syncBlindPlayback();
});
blindAudio.addEventListener("pause",syncBlindPlayback);
blindAudio.addEventListener("error",()=>{$("blind-status").textContent="此瀏覽器無法載入音軌；本輪不會自動評分。請嘗試 Chrome 或 Edge。";});
restoreAnalysis();window.ht1bExtendedReady=true;renderExtended();
