"use strict";
const data = JSON.parse(document.getElementById("benchmark-data").textContent);
const $ = id => document.getElementById(id);
const state = {metric:"esr", step:3000, record:0, track:"reference_wet", wave:"signal"};
const steps = [2000,3000,4000,5000];
const definitions = {
  esr:{name:"ESR",help:"能量正規化誤差 · 無單位 · 越低越好", formula:"ESR"},
  mse:{name:"MSE",help:"平均平方誤差 · 振幅² · 越低越好",formula:"MSE"},
  mae:{name:"MAE",help:"平均絕對誤差 · 振幅 · 越低越好",formula:"MAE"},
  mrstft:{name:"MR-STFT",help:"多解析度頻譜誤差 · SC + log magnitude L1 · 越低越好"},
  rms_db_mae:{name:"RMS 包絡 MAE",help:"10 ms RMS 包絡的平均絕對差 · dB · 越低越好"}
};
const models = () => [
  {key:"pure_algorithm",name:"純演算法",color:"baseline",tag:"CIRCUIT",description:"原始物理參數 · 數值求解"},
  {key:`ours_${state.step}`,name:"PhysicsNeMo",color:"ours",tag:`${state.step.toLocaleString()} STEPS`,description:"學習物理參數 · 數值求解"},
  {key:"author_pretrained",name:"RiccardoVib",color:"author",tag:"ENCODER–DECODER",description:"條件式神經網路 · 固定權重"}
];
const fmt = (v,metric=state.metric) => v === null ? "無法定義" : metric === "mse" ? v.toExponential(3) : v.toFixed(metric === "rms_db_mae" ? 3 : 5);
const knobsText = r => `A${r.knobs.attack} · R${r.knobs.release} · ${r.knobs.ratio}:1 · ${r.knobs.threshold} dB`;
const cssColor = name => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
const methodStyle = m => `--method:var(--${m.color})`;
const audio = $("audio");
let loadVersion = 0;
let frame = 0;
let activeSeek = null;

function renderMetrics(){
  const methods = models(), metric = state.metric, metricName = definitions[metric].name;
  const values = methods.map(m=>data.macro_mean[m.key][metric]);
  const best = Math.min(...values);
  $("metric-help").textContent = `${definitions[metric].help} / 五段音訊等權平均`;
  document.querySelectorAll("[data-metric]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.metric===metric));
  $("summary").innerHTML = methods.map((m,i)=>`<article class="model" style="${methodStyle(m)}"><div class="model-top"><span class="model-name">${m.name}</span><span class="model-tag">${values[i]===best?"本組最低":m.tag}</span></div><p class="value">${fmt(values[i])}</p><p class="description">${m.description}</p></article>`).join("");
  $("legend").innerHTML = methods.map(m=>`<span style="${methodStyle(m)}"><i class="dot"></i>${m.name}</span>`).join("");
  const max = Math.max(...data.records.flatMap(r=>methods.map(m=>r.metrics[m.key][metric])))*1.03;
  $("condition-chart").innerHTML = data.records.map((r,i)=>`<button class="condition-row" data-record="${i}" aria-pressed="${state.record===i}" aria-label="設定 ${i+1}，${knobsText(r)}"><span class="condition-name"><b>0${i+1}</b> &nbsp; ${r.knobs.ratio}:1 / ${r.knobs.threshold} dB<small>Attack ${r.knobs.attack} · Release ${r.knobs.release}</small></span><span class="bars">${methods.map(m=>`<span class="bar-row" style="${methodStyle(m)}" title="${m.name}: ${r.metrics[m.key][metric]}"><span class="bar-track"><span class="bar" style="display:block;width:${r.metrics[m.key][metric]/max*100}%"></span></span><span>${fmt(r.metrics[m.key][metric])}</span></span>`).join("")}</span></button>`).join("");
  const winner=methods[values.indexOf(best)];
  const won = data.records.filter(r=>r.metrics.author_pretrained[metric]===Math.min(...methods.map(m=>r.metrics[m.key][metric]))).length;
  $("insight-title").textContent = `${winner.name}的平均 ${metricName} 最低。`;
  $("insight-body").textContent = `RiccardoVib 模型在 ${won} / 5 組設定取得最低 ${metricName}。目前比較我們的 ${state.step.toLocaleString()} 步版本，各設定的勝負仍有差異。`;
  const improvement=(1-values[2]/values[1])*100;
  $("delta-value").textContent = `${Math.abs(improvement).toFixed(1)}%`;
  $("delta-label").textContent = `RiccardoVib 相較我們的平均 ${metricName} ${improvement>=0?"降低":"增加"}`;
  $("table-caption").textContent = `${metricName} · ${definitions[metric].help} · PhysicsNeMo ${state.step.toLocaleString()} 步 · 粗體表示該列最低值`;
  $("metric-table").innerHTML = data.records.map((r,i)=>{
    const row=methods.map(m=>r.metrics[m.key][metric]);
    return `<tr><td>0${i+1} / ${knobsText(r)}</td>${row.map(v=>`<td class="${v===Math.min(...row)?"best":""}">${fmt(v)}</td>`).join("")}</tr>`;
  }).join("")+`<tr class="average"><td>平均 / Macro mean</td>${values.map(v=>`<td class="${v===best?"best":""}">${fmt(v)}</td>`).join("")}</tr>`;
}

function trainingChart(container,values,color,label){
  const w=520,h=175,left=52,right=30,top=24,bottom=25,max=Math.max(...values)*1.2;
  const x=i=>left+i*(w-left-right)/3, y=v=>h-bottom-v/max*(h-top-bottom);
  const ticks=[0,max/2,max];
  $(container).innerHTML=`<svg class="linechart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}：${steps.map((s,i)=>`${s} 步 ${values[i].toFixed(5)}`).join('，')}">${ticks.map(t=>`<line x1="${left}" x2="${w-right}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)"/><text x="${left-8}" y="${y(t)+4}" text-anchor="end" fill="currentColor" font-family="Consolas,monospace" font-size="11">${t.toFixed(3)}</text>`).join("")}<path d="${values.map((v,i)=>`${i?'L':'M'} ${x(i)} ${y(v)}`).join(' ')}" fill="none" stroke="var(--${color})" stroke-width="2"/>${values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="${steps[i]===state.step?6:4}" fill="var(--${color})"/><text x="${x(i)}" y="${y(v)-13}" text-anchor="middle" fill="var(--ink)" font-family="Consolas,monospace" font-size="12">${v.toFixed(5)}</text>`).join("")}</svg><div class="step-buttons">${steps.map(s=>`<button data-step="${s}" aria-pressed="${s===state.step}" aria-label="切換至 ${s} 步">${s.toLocaleString()}</button>`).join("")}</div>`;
}
function renderTraining(){
  trainingChart("loss-chart",data.training.map(s=>s.fixed_loss),"ours","固定 Loss");
  trainingChart("validation-chart",steps.map(s=>data.macro_mean[`ours_${s}`].esr),"author","驗證 ESR");
}
function trackOptions(){return [{key:"reference_dry",name:"Dry 輸入",color:"reference"},{key:"reference_wet",name:"Wet 目標",color:"ink"},...models()];}
function renderListening(){
  const r=data.records[state.record];
  $("record").value=state.record;
  $("knobs").innerHTML = [["ATTACK",r.knobs.attack],["RELEASE",r.knobs.release],["RATIO",r.knobs.ratio+":1"],["THRESHOLD",r.knobs.threshold+" dB"]].map(([label,value])=>`<div class="knob">${label}<strong>${value}</strong></div>`).join("");
  $("tracks").innerHTML=trackOptions().map(t=>`<button class="track" data-track="${t.key}" style="${methodStyle(t)}" aria-pressed="${state.track===t.key}"><i class="dot"></i>${t.name}</button>`).join("");
  document.querySelectorAll("[data-wave]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.wave===state.wave));
  drawWaveform();
  if(window.ht1bExtendedReady) renderExtended();
}
function drawWaveform(){
  if(state.wave==="rms" && window.ht1bExtendedReady){drawRms();return;}
  const r=data.records[state.record], canvas=$("waveform"), ctx=canvas.getContext("2d");
  const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;
  const options=state.wave==="residual"?models():[{key:"reference_wet",color:"reference",name:"Wet"},...models()];
  if(state.wave==="signal"&&state.track==="reference_dry") options.push(trackOptions()[0]);
  const waves=state.wave==="residual"?r.residuals:r.waveforms;
  const peak=Math.max(.001,...options.flatMap(t=>waves[t.key].flat().map(Math.abs)));
  ctx.strokeStyle=cssColor("line");ctx.lineWidth=1;
  [.25,.5,.75].forEach(f=>{ctx.beginPath();ctx.moveTo(0,h*f);ctx.lineTo(w,h*f);ctx.stroke();});
  for(let j=0;j<=4;j++){ctx.beginPath();ctx.moveTo(w*j/4,0);ctx.lineTo(w*j/4,h);ctx.stroke();}
  options.sort((a,b)=>(a.key===state.track?1:0)-(b.key===state.track?1:0));
  options.forEach(t=>{const wave=waves[t.key];ctx.strokeStyle=cssColor(t.color);ctx.globalAlpha=t.key===state.track?.9:.24;ctx.lineWidth=Math.max(1,w/wave.length*.75);ctx.beginPath();wave.forEach(([lo,hi],i)=>{const x=(i+.5)/wave.length*w;ctx.moveTo(x,h/2-hi/peak*h*.45);ctx.lineTo(x,h/2-lo/peak*h*.45);});ctx.stroke();});
  ctx.globalAlpha=1;
  const selected=trackOptions().find(t=>t.key===state.track);
  $("wave-caption").textContent=state.wave==="signal"?`輸出包絡 · ${selected.name} 加亮，其他模型淡色疊圖`:`預測 − Wet · 共用振幅刻度${state.track.startsWith("reference")?"（目標與輸入無殘差軌）":"，所選模型加亮"}`;
  $("wave-scale").textContent=`±${peak.toFixed(3)} amplitude`;
  canvas.setAttribute("aria-label",`${$("wave-caption").textContent}；振幅範圍正負 ${peak.toFixed(3)}`);
}
function updateTime(){
  const duration=data.records[state.record].samples_scored/48000;
  const time=Math.min(audio.currentTime||0,duration);
  $("seek").value=time;$("seek").max=duration;
  $("time").textContent=`${time.toFixed(3)} / ${duration.toFixed(3)} s`;
  $("playhead").style.left=`${Math.min(100,time/duration*100)}%`;
  $("play").textContent=audio.paused?"播放":"暫停";
  $("play").setAttribute("aria-label",audio.paused?"播放音訊":"暫停音訊");
}
async function play(){
  try{await audio.play();$("audio-status").textContent="播放中 · 所有模型使用相同播放音量。";}
  catch(error){$("audio-status").textContent=`無法播放：${error.message}。請再次按播放，或下載音軌試聽。`;}
  updateTime();
}
function setAudio(preserve=true){
  const time=preserve?audio.currentTime:0, wasPlaying=!audio.paused, version=++loadVersion;
  audio.pause();
  if(activeSeek) audio.removeEventListener("loadedmetadata",activeSeek);
  activeSeek=()=>{if(version!==loadVersion)return;audio.currentTime=Math.min(time||0,Math.max(0,audio.duration-.001));updateTime();$("audio-status").textContent="音訊已就緒 · 同位置切換，未做音量正規化。";if(wasPlaying)void play();};
  audio.addEventListener("loadedmetadata",activeSeek,{once:true});
  audio.src=`data:audio/wav;base64,${data.records[state.record].audio[state.track]}`;
  audio.load();updateTime();
  $("audio-status").textContent="載入內嵌音訊…";
}
function chooseRecord(index){state.record=index;renderMetrics();renderListening();setAudio(false);}
function chooseStep(step){const wasOurs=state.track.startsWith("ours_");state.step=step;$("step").value=step;if(wasOurs)state.track=`ours_${step}`;renderMetrics();renderTraining();renderListening();if(wasOurs)setAudio(true);}
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(){
  const rows=[["record","model","step","attack_knob","release_knob","ratio","threshold_db","mse","mae","esr","mrstft","rms_db_mae","samples","sample_rate","metric_protocol"]];
  data.records.forEach(r=>Object.entries(r.metrics).forEach(([model,v])=>rows.push([r.id,model,model.startsWith("ours_")?model.slice(5):"",r.knobs.attack,r.knobs.release,r.knobs.ratio,r.knobs.threshold,v.mse,v.mae,v.esr,v.mrstft,v.rms_db_mae,r.samples_scored,48000,data.metric_protocol.version])));
  Object.entries(data.macro_mean).forEach(([model,v])=>rows.push(["MACRO_MEAN",model,model.startsWith("ours_")?model.slice(5):"","","","","",v.mse,v.mae,v.esr,v.mrstft,v.rms_db_mae,"",48000,data.metric_protocol.version]));
  download("\uFEFF"+rows.map(row=>row.join(",")).join("\r\n"),"cl1b-comparison-all-checkpoints.csv","text/csv;charset=utf-8");
}
document.addEventListener("click",e=>{
  const b=e.target.closest("button");if(!b)return;
  if(b.dataset.metric){state.metric=b.dataset.metric;renderMetrics();}
  if(b.dataset.record!==undefined)chooseRecord(Number(b.dataset.record));
  if(b.dataset.step)chooseStep(Number(b.dataset.step));
  if(b.dataset.wave){state.wave=b.dataset.wave;renderListening();}
  if(b.dataset.track){state.track=b.dataset.track;renderListening();setAudio(true);}
});
$("step").addEventListener("change",e=>chooseStep(Number(e.target.value)));
$("record").addEventListener("change",e=>chooseRecord(Number(e.target.value)));
$("play").addEventListener("click",()=>{if(audio.paused)void play();else audio.pause();});
$("seek").addEventListener("input",e=>{if(audio.readyState>=1){audio.currentTime=Number(e.target.value);updateTime();}});
$("volume").addEventListener("input",e=>{audio.volume=Number(e.target.value);});
$("loop").addEventListener("change",e=>{audio.loop=e.target.checked;});
$("csv").addEventListener("click",csv);
$("download-audio").addEventListener("click",()=>{const link=document.createElement("a");link.href=audio.src;link.download=`${data.records[state.record].id}__${state.track}.wav`;link.click();});
$("theme").addEventListener("click",()=>{const dark=document.documentElement.dataset.theme!=="dark";document.documentElement.dataset.theme=dark?"dark":"light";$("theme").textContent=dark?"淺色模式":"深色模式";$("theme").setAttribute("aria-label",`切換${dark?"淺":"深"}色模式`);drawWaveform();});
audio.addEventListener("play",()=>{cancelAnimationFrame(frame);function tick(){updateTime();if(!audio.paused)frame=requestAnimationFrame(tick);}tick();});
audio.addEventListener("pause",()=>{cancelAnimationFrame(frame);updateTime();});
audio.addEventListener("ended",updateTime);
audio.addEventListener("timeupdate",updateTime);
audio.addEventListener("error",()=>{$("audio-status").textContent="瀏覽器無法載入此 WAV，請下載音軌以音訊軟體開啟。";});
new ResizeObserver(drawWaveform).observe($("waveform"));
$("record").innerHTML=data.records.map((r,i)=>`<option value="${i}">0${i+1} / ${knobsText(r)}</option>`).join("");
$("source-link").href=data.source_repository;
$("commit").textContent=data.source_commit;$("hash").textContent=data.weights_sha256;
$("parity").textContent=data.tensorflow_parity.max_absolute_error.toExponential(2);
renderMetrics();renderTraining();renderListening();setAudio(false);
