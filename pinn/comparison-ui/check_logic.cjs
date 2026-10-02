/* Node data/state checks; no browser rendering claim. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'../..'),ui=path.join(__dirname,'unified'),C=require('./unified/core.js');
const reportPath=process.env.HT1B_REPORT_PATH||path.join(root,'output/cl1b-comparison.html');
const html=fs.readFileSync(reportPath,'utf8');
const data=JSON.parse(fs.readFileSync(reportPath.replace(/\.html$/,'.metrics.json'),'utf8'));
const source=JSON.parse(fs.readFileSync(path.join(root,'runs/s6-tfilm-pinn-gpu-20260929/common-validation/comparison.json'),'utf8'));
assert.deepEqual(data.five_models,source);
assert.deepEqual(JSON.parse(html.match(/<script id="benchmark-data" type="application\/json">([\s\S]*?)<\/script>/)[1]),data);
assert.equal(data.model_keys.length,6);assert.equal(C.METHODS.length,7);
assert.deepEqual(new Set(C.METHODS.map(m=>m.key)),new Set(Object.keys(source.sources)));
const close=(a,b)=>assert.ok(Math.abs(a-b)<=1e-12+Math.abs(b)*1e-9,`${a} != ${b}`);
for(const mode of ['macro','pooled'])for(const m of C.METHODS)for(const k of Object.keys(C.METRICS))close(C.aggregate(source.records,m.key,k,mode,source.protocol.esr_floor),source[mode==='macro'?'macro_mean':'pooled'][m.key][k]);
const example=source.records[180],filter=Object.fromEntries(Object.entries(C.FIELDS).map(([key,field])=>[key,String(example.labels[field])]));
const filtered=C.filterRows(source.records,filter);
assert.ok(filtered.length>0&&filtered.length<620);
assert.ok(filtered.every(r=>Object.entries(C.FIELDS).every(([key,field])=>r.labels[field]===Number(filter[key]))));
assert.equal(C.filterRows(source.records,{attack:'999'}).length,0);
for(const mode of ['macro','pooled'])assert.equal(C.aggregate([],'mlp','esr',mode),null);
const synthetic=[{samples:1,target_energy:1,metrics:{x:{mse:4,esr:4,mae:2}}},{samples:3,target_energy:12,metrics:{x:{mse:1,esr:.25,mae:1}}}];
close(C.aggregate(synthetic,'x','esr','macro'),2.125);close(C.aggregate(synthetic,'x','esr','pooled'),7/13);close(C.aggregate(synthetic,'x','mae','pooled'),1.25);
for(const [key,h] of Object.entries(data.histories)){
 if(h.kind==='step_training'){
  assert.equal(h.epochs,1);assert.equal(h.selected_epoch,null);assert.ok(h.rows.length>100);
  assert.equal(h.final_step,key==='mlp'?316198:335417);assert.equal(h.frames_covered,5100828000);
  assert.ok(h.rows.every((r,i)=>r.step<=h.final_step&&(!i||r.step>h.rows[i-1].step)));
  const lossRows=fs.readFileSync(path.join(root,h.source),'utf8').replace(/^\uFEFF/,'').split(/\r?\n/).filter(Boolean).flatMap(line=>{try{return [JSON.parse(line)];}catch{return [];}});
  for(const r of h.rows){
   const original=lossRows.find(v=>v.step===r.step&&(key==='mlp'||v.latest_loss?.total!==undefined));
   assert.ok(original);const loss=key==='mlp'?original:original.latest_loss;
   assert.equal(r.training_loss,loss.total);assert.equal(r.audio_loss,loss.data);assert.equal(r.physics_loss,loss.physics);
  }
  if(key==='mlp'){assert.equal(h.rows.length,lossRows.length);assert.equal(h.rows.at(-1).step,316150);}
  continue;
 }
 assert.equal(h.rows.length,16);close(h.rows.find(r=>r.epoch===h.selected_epoch).esr,Math.min(...h.rows.map(r=>r.esr)));
 assert.ok(Math.abs(h.rows.find(r=>r.epoch===h.selected_epoch).esr-source.macro_mean[key].esr)<2e-4*source.macro_mean[key].esr);
}
assert.deepEqual(Object.keys(data.availability).filter(k=>data.availability[k].historical_full_test).sort(),['gru','mlp','riccardovib']);
assert.ok(Object.values(data.availability).every(v=>v.common_validation&&!v.new_independent_test));
assert.equal(data.coverage.mlp,5100828000);assert.equal(data.coverage.gru,5100828000);
const csv=C.recordCSV(filtered,source.window_signature);
assert.equal(csv.split('\r\n').length,filtered.length*5+1);assert.ok(csv.includes('"s6_pinn"'));assert.ok(csv.includes(source.window_signature));
assert.equal(C.recordCSV([],source.window_signature).split('\r\n').length,1);
const markup=html.slice(0,html.indexOf('<script id="benchmark-data"'));
const ids=[...markup.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size,'Duplicate IDs');
for(const obsolete of ['blind-test','five-cards','five-bars','architecture-cards','s6-pinn-study'])assert.ok(!ids.includes(obsolete));
assert.ok(!/data:audio|<script[^>]+src=/i.test(html),'Report must not embed audio or third-party scripts');
assert.equal(data.listening.scope,'historical_five_clip_only');assert.equal(data.listening.records.length,5);
const auditionTracks=['reference_dry','reference_wet','mlp_full','gru_full','s4_tfilm','s6_tfilm','s6_tfilm_pinn','author_pretrained','pure_algorithm','ours_5000'];
for(const record of data.listening.records){
 assert.deepEqual(Object.keys(record.audio),auditionTracks);
 assert.deepEqual(Object.keys(record.waveform),Object.keys(record.audio));
 for(const [key,wave] of Object.entries(record.waveform)){
  for(const field of ['low','high','residual_low','residual_high','diff_rms'])assert.equal(wave[field].length,640);
  assert.ok(wave.low.every((value,i)=>value<=wave.high[i]));
  assert.ok(wave.residual_low.every((value,i)=>value<=wave.residual_high[i]));
  assert.ok(wave.diff_rms.every(value=>Number.isFinite(value)&&value>=0));
  if(key==='reference_wet')assert.ok(wave.diff_rms.every(value=>value===0)&&wave.residual_low.every(value=>value===0)&&wave.residual_high.every(value=>value===0));
 }
 assert.ok(record.waveform.reference_dry.diff_rms.some(value=>value>0));
}
for(const record of data.listening.records)for(const track of Object.values(record.audio)){
 const wav=path.join(root,'output',track.path);
 assert.ok(fs.existsSync(wav),`Missing local audition track: ${track.path}`);
 const bytes=fs.readFileSync(wav);
 assert.equal(bytes.toString('ascii',0,4),'RIFF');assert.equal(bytes.toString('ascii',8,12),'WAVE');
 assert.equal(bytes.readUInt32LE(bytes.indexOf(Buffer.from('data'))+4)/4,record.samples,`Audio length mismatch: ${track.path}`);
}
for(const match of markup.matchAll(/href="([^"#]+)"/g))if(!/^https?:/.test(match[1]))assert.ok(fs.existsSync(path.join(root,'output',match[1])),`Missing ${match[1]}`);
const elements=new Map(),downloads=[];
function node(id){return {id,value:'',textContent:'',innerHTML:'',dataset:{},attributes:{},listeners:{},disabled:false,tagName:'DIV',open:false,paused:true,currentTime:0,duration:0,readyState:0,volume:1,loop:false,width:0,height:0,setAttribute(k,v){this.attributes[k]=v;},addEventListener(k,fn){this.listeners[k]=fn;},pause(){this.paused=true;},load(){},play(){this.paused=false;return Promise.resolve();},getBoundingClientRect(){return {width:800,height:248};},getContext(){return {setTransform(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){}};}};}
for(const id of ids)elements.set(id,node(id));
const benchmark=node('benchmark-data');benchmark.textContent=JSON.stringify(data);elements.set('benchmark-data',benchmark);
const get=id=>{if(!id)return null;assert.ok(elements.has(id),`Unknown element ${id}`);return elements.get(id);};
for(const k of Object.keys(C.FIELDS))get('filter-'+k).value='all';
get('aggregate').value='macro';get('record-sort').value='s6_pinn';get('history-model').value='s6_pinn';get('records').tagName='DETAILS';
const controls=Object.keys(C.METRICS).map(k=>{const n=node(k);n.dataset.metric=k;return n;});
let currentBlob;
const context=vm.createContext({ComparisonCore:C,console,Blob,URL:{createObjectURL(b){currentBlob=b;return 'blob:test';},revokeObjectURL(){}},setTimeout(fn){fn();},
 getComputedStyle(){return {getPropertyValue(){return '#888';}}},
 localStorage:{getItem(){throw new Error('storage disabled');},setItem(){throw new Error('storage disabled');}},window:{location:{hash:'#records'},addEventListener(){}},
 document:{getElementById:get,querySelectorAll(s){assert.equal(s,'[data-metric]');return controls;},documentElement:{dataset:{}},body:{appendChild(){}},createElement(tag){assert.equal(tag,'a');return {click(){downloads.push({filename:this.download,blob:currentBlob});},remove(){}};}}
});
new vm.Script(html.match(/<\/script><script>([\s\S]*)<\/script>/)[1]);
vm.runInContext(fs.readFileSync(path.join(ui,'app.js'),'utf8'),context);
if(data.long_form){
 assert.equal(get('evaluation').value,'B');
 assert.deepEqual(Object.keys(data.long_form.sources).sort(),C.METHODS.map(m=>m.key).sort());
 for(const [id,study] of Object.entries(data.long_form.studies)){
  assert.equal(study.records.length,id==='G'?495:620);
  for(const mode of ['macro','pooled'])for(const m of C.METHODS)for(const k of Object.keys(C.METRICS))close(C.aggregate(study.records,m.key,k,mode),study[mode==='macro'?'macro_mean':'pooled'][m.key][k]);
  get('evaluation').value=id;get('evaluation').listeners.change();
  assert.ok(get('filter-count').textContent.startsWith(`${study.files}／${study.files}`));
  assert.ok(get('fact-length').textContent.includes(study.candidate.seconds.toFixed(1)));
  for(const m of C.METHODS)assert.ok(get('comparison-table').innerHTML.includes(m.name));
  assert.ok(!get('ranking-insight').textContent.includes('這是短視窗'));
 }
 get('evaluation').value='short';get('evaluation').listeners.change();
 assert.equal(get('fact-length').textContent,'21.33 ms');
}
assert.equal(get('records').open,true);
for(const m of C.METHODS)assert.ok(get('comparison-table').innerHTML.includes(m.name));
assert.equal((get('comparison-table').innerHTML.match(/<tr>/g)||[]).length,8);
assert.ok(get('ranking-insight').textContent.includes('MLP＋PINN'));
assert.ok(get('filter-count').textContent.startsWith('620／620'));assert.ok(get('history-summary').textContent.includes('第 15 輪'));
assert.ok(get('history-chart').innerHTML.includes('selected-point'));
assert.ok(get('prev').disabled);get('next').listeners.click();assert.ok(get('page-label').textContent.includes('第 2／31 頁'));
get('filter-attack').value='999';get('filter-attack').listeners.change();assert.ok(get('comparison-table').innerHTML.includes('沒有符合'));
assert.equal(get('page-label').textContent,'0 檔');assert.ok(!get('ranking-insight').textContent.includes('最低'));
get('reset').listeners.click();assert.equal(get('filter-attack').value,'all');assert.ok(get('page-label').textContent.includes('第 1／31 頁'));
controls.find(b=>b.dataset.metric==='mse').listeners.click();assert.ok(get('pinn-insight').textContent.includes('較低'));
assert.equal(controls.find(b=>b.dataset.metric==='mse').attributes['aria-pressed'],'true');
get('aggregate').value='pooled';get('aggregate').listeners.change();assert.ok(get('metric-help').textContent.includes('合併樣本'));
for(const key of ['s4','s6','s6_pinn']){get('history-model').value=key;get('history-model').listeners.change();assert.ok(get('history-summary').textContent.includes(`第 ${data.histories[key].selected_epoch} 輪`));}
for(const key of ['mlp','gru']){
 get('history-model').value=key;get('history-model').listeners.change();
 assert.ok(get('history-summary').textContent.includes('最佳輪次：不適用'));
 assert.ok(get('history-note').textContent.includes('單批'));
 assert.ok(!get('history-table').innerHTML.includes('驗證 macro ESR'));
 assert.ok(!get('history-chart').innerHTML.includes('selected-point'));
 assert.ok(get('history-table').innerHTML.includes(data.histories[key].rows.at(-1).step.toLocaleString('zh-TW')));
 assert.equal((get('history-table').innerHTML.match(/<tr>/g)||[]).length,21);
 get('history-next').listeners.click();assert.ok(get('history-page').textContent.includes('第 2／'));
 get('history-prev').listeners.click();assert.ok(get('history-page').textContent.includes('第 1／'));
 get('export-history').listeners.click();
}
get('history-model').value='s6_pinn';get('history-model').listeners.change();assert.equal(get('history-page').textContent,'全部 16 輪');assert.ok(get('history-next').disabled);
for(const m of C.METHODS)assert.ok(get('comparison-table').innerHTML.includes(m.name));
assert.ok(!get('comparison-table').innerHTML.includes('第 null'));
get('theme').listeners.click();assert.equal(context.document.documentElement.dataset.theme,'dark');get('theme').listeners.click();assert.equal(context.document.documentElement.dataset.theme,'light');
for(const id of ['export-summary','export-records','export-json'])get(id).listeners.click();
(async()=>{
 assert.equal(downloads.length,5);
 assert.equal((await downloads[0].blob.text()).split('\r\n').length,data.histories.mlp.rows.length+1);
 assert.equal((await downloads[1].blob.text()).split('\r\n').length,data.histories.gru.rows.length+1);
 assert.equal((await downloads[2].blob.text()).split('\r\n').length,8);
 assert.equal((await downloads[3].blob.text()).split('\r\n').length,3101);assert.deepEqual(JSON.parse(await downloads[4].blob.text()),data);
 console.log('PASS: 7 methods x 5 metrics x 2 aggregations; source parity; filters, pagination, histories, theme, CSV/JSON, unique sections and local links.');
 console.log('Node data/state checks only; no browser rendering claim.');
})().catch(e=>{console.error(e);process.exitCode=1;});
