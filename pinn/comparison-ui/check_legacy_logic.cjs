/* Node-only state/metric checks. This is not a browser rendering or audio test. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {webcrypto} = require("node:crypto");
const root = path.resolve(__dirname, "../..");
const report = JSON.parse(fs.readFileSync(path.join(root,"output/cl1b-comparison.metrics.json"),"utf8"));
const elements = new Map();
const controls = new Map();
function node(){return {value:"",textContent:"",innerHTML:"",disabled:false,hidden:false,paused:true,currentTime:0,volume:1,loop:true,style:{},dataset:{},listeners:{},setAttribute(){},removeAttribute(){},addEventListener(type,handler){this.listeners[type]=handler;},removeEventListener(){},pause(){this.paused=true;},load(){},getBoundingClientRect(){return {width:800,height:180};}};}
function $(id){if(!elements.has(id))elements.set(id,node());return elements.get(id);}
$("region-kind").value="attack";
const stored=new Map();
const context = vm.createContext({data:report,$,steps:[2000,3000,4000,5000],state:{record:0,step:3000,track:"reference_wet"},window:{},crypto:webcrypto,console,
  audio:node(),document:{querySelectorAll(selector){return controls.get(selector)||[];}},models:()=>[{key:"pure_algorithm",name:"純演算法"},{key:"ours_3000",name:"PhysicsNeMo"},{key:"author_pretrained",name:"RiccardoVib"}],fmt:v=>v===null?"無法定義":String(v),knobsText:()=>"test",
  localStorage:{getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v)},renderListening(){},download(){},
});
vm.runInContext(fs.readFileSync(path.join(__dirname,"analysis.js"),"utf8"),context);
context.assert=assert;
vm.runInContext(`
// No subjective scores or manual annotations are fabricated on first load.
assert.equal(trials.length,0);
assert.equal(Object.keys(annotations).length,0);
// Region statistics agree with the Python-generated full-clip benchmark.
for(const r of data.records)for(const key of Object.keys(r.metrics)){
  const v=localMetrics(r,key,0,r.frames.counts.length);
  for(const m of ['mse','mae','esr','rms_db_mae'])assert.ok(Math.abs(v[m]-r.metrics[key][m]) < 1e-9 + Math.abs(r.metrics[key][m])*1e-5);
}
createTrial();
assert.equal(new Set(currentTrial.candidates.map(c=>c.model)).size,5);
assert.ok(currentTrial.candidates.some(c=>c.model==='reference_wet'));
assert.ok(currentTrial.candidates.some(c=>c.model==='reference_dry'));
assert.ok(currentTrial.candidates.every(c=>c.score===null));
assert.equal($('blind-submit').disabled,true);
assert.ok(!$('blind-candidates').innerHTML.includes('RiccardoVib'));
assert.ok(!$('blind-candidates').innerHTML.includes('PhysicsNeMo'));
submitTrial();assert.equal(trials.length,0);
currentTrial.candidates.forEach(c=>{c.score=0;c.heard=true;});
updateBlindEligibility();assert.equal($('blind-submit').disabled,true);
currentTrial.reference_heard=true;
updateBlindEligibility();assert.equal($('blind-submit').disabled,false);
const frozen=currentTrial.step;state.step=5000;assert.equal(currentTrial.step,frozen);
submitTrial();assert.equal(trials.length,1);
assert.ok($('blind-candidates').innerHTML.includes('PhysicsNeMo'));
submitTrial();assert.equal(trials.length,1);
assert.ok(trials[0].candidates.every(c=>c.score===0));
// Local storage round-trip; then an unavailable storage backend is surfaced.
trials=[];restoreAnalysis();assert.equal(trials.length,1);
localStorage.setItem=()=>{throw new Error('storage disabled');};
persistAnalysis();assert.equal(canStore,false);
assert.ok($('blind-summary').textContent.includes('只在本次開啟'));
// Empty, inverted and too-small annotations cannot produce plausible-looking scores.
$('region-start').value='';$('region-end').value='0.1';applyRegion();
assert.equal(Object.keys(annotations).length,0);
$('region-start').value='0.2';$('region-end').value='0.1';applyRegion();
assert.equal(Object.keys(annotations).length,0);
$('region-start').value='0.001';$('region-end').value='0.002';applyRegion();
assert.equal(Object.keys(annotations).length,0);
$('region-start').value='0.12';$('region-end').value='0.34';applyRegion();
assert.equal(annotations[data.records[0].id].attack.start,12);
assert.equal(annotations[data.records[0].id].attack.stop,34);
`,context);
console.log("PASS: region parity, annotation validation, anonymous initial state, blank/zero scores, listening gate, frozen checkpoint, one-time reveal, storage fallback.");

// Full-corpus model study: controls must render the externally scored model in
// both splits while keeping baseline absent from the reused test split.
assert.equal(report.architecture.records.length,620);
$("arch-ratio").value="all";
$("arch-record-sort").value="gru";
const splitButtons=["validation","test"].map(value=>{const button=node();button.dataset.archSplit=value;return button;});
const metricButtons=["esr","mse","mae","mrstft","rms_db_mae"].map(value=>{const button=node();button.dataset.archMetric=value;return button;});
controls.set("[data-arch-split]",splitButtons);
controls.set("[data-arch-metric]",metricButtons);
context.definitions=Object.fromEntries(["esr","mse","mae","mrstft","rms_db_mae"].map(key=>[key,{name:key,help:"lower is better"}]));
vm.runInContext(fs.readFileSync(path.join(__dirname,"architecture.js"),"utf8"),context);
assert.ok($("arch-cards").innerHTML.includes("RiccardoVib"));
assert.ok($("arch-cards").innerHTML.includes("純演算法"));
assert.ok($("arch-record-table").innerHTML.includes("RiccardoVib"));
splitButtons[1].listeners.click();
assert.ok($("arch-scope").textContent.includes("探索性重用"));
assert.ok($("arch-cards").innerHTML.includes("RiccardoVib"));
assert.ok(!$("arch-cards").innerHTML.includes("純演算法"));
$("arch-ratio").value="2";
$("arch-record-sort").value="riccardovib";
metricButtons[3].listeners.click();
assert.ok($("arch-record-table").innerHTML.includes("RiccardoVib − MLP"));
assert.ok($("arch-scope").textContent.includes(`${report.architecture.records.filter(r=>r.ratio===2).length}／620`));
console.log("PASS: RiccardoVib, MLP and GRU model study renders across splits, metrics, ratio filter and record sorting.");

// Common five-model data must remain comparable through filters, pooling and CSV.
assert.equal(report.five_models.records.length,620);
for (const key of ["attack","release","ratio","threshold"]) $("five-"+key).value="all";
$("five-aggregate").value="macro";
$("five-sort").value="s6";
const fiveButtons=["esr","mse","mae","mrstft","rms_db_mae"].map(value=>{const button=node();button.dataset.fiveMetric=value;return button;});
controls.set("[data-five-metric]",fiveButtons);
context.window.location={hash:""};
context.window.addEventListener=()=>{};
let csvCapture="";
context.download=text=>{csvCapture=text;};
vm.runInContext(fs.readFileSync(path.join(__dirname,"five-models.js"),"utf8"),context);
const displayedNumber=value=>value.toLocaleString("zh-TW",{minimumFractionDigits:3,maximumFractionDigits:5});
assert.equal($("five-models").hidden,false);
for(const name of ["純演算法","MLP＋PINN","GRU＋PINN","S4＋TFiLM","S6＋TFiLM","RiccardoVib"])
  assert.ok($("five-cards").innerHTML.includes(name));
for(const values of Object.values(report.five_models.macro_mean))
  assert.ok($("five-cards").innerHTML.includes(displayedNumber(values.esr)));
assert.ok($("five-page").textContent.includes("1–20／620"));
$("five-next").listeners.click();
assert.ok($("five-page").textContent.includes("21–40／620"));
$("five-aggregate").value="pooled";
$("five-aggregate").listeners.change();
for(const values of Object.values(report.five_models.pooled))
  assert.ok($("five-cards").innerHTML.includes(displayedNumber(values.esr)));
$("five-ratio").value="2";
$("five-ratio").listeners.change();
const commonRows=report.five_models.records.filter(r=>r.labels.ratio===2);
for(const key of ["baseline","mlp","gru","s4","s6","riccardovib"]){
  const squared=commonRows.reduce((sum,r)=>sum+r.metrics[key].mse*r.samples,0);
  const energy=commonRows.reduce((sum,r)=>sum+r.target_energy,0);
  const count=commonRows.reduce((sum,r)=>sum+r.samples,0);
  assert.ok($("five-cards").innerHTML.includes(displayedNumber(squared/Math.max(energy,count*1e-8))));
}
$("five-csv").listeners.click();
assert.equal(csvCapture.split("\r\n").length,commonRows.length*5+1);
assert.ok(csvCapture.includes(report.five_models.window_signature));
assert.ok(csvCapture.split("\r\n")[0].includes(",baseline,"));
$("five-sort").value="baseline";
$("five-sort").listeners.change();
const worstBaseline=[...commonRows].sort((a,b)=>b.metrics.baseline.esr-a.metrics.baseline.esr)[0];
assert.ok($("five-records").innerHTML.includes(worstBaseline.id));
fiveButtons[3].listeners.click();
assert.ok($("five-help").textContent.includes("256／512／1,024"));
$("five-threshold").value="123";
$("five-threshold").listeners.change();
assert.ok($("five-records").innerHTML.includes("沒有符合條件"));
assert.ok(!$("five-cards").innerHTML.includes("NaN"));
assert.equal($("five-next").disabled,true);
$("five-reset").listeners.click();
assert.ok($("five-count").textContent.includes("620／620"));
console.log("PASS: five-model means, pooled ESR, knob filtering, pagination, empty state and complete filtered CSV.");

// Current pending/completed PINN panel must reflect the saved experiment state.
vm.runInContext(fs.readFileSync(path.join(__dirname,"s6-pinn-study.js"),"utf8"),context);
if (report.s6_pinn_study) {
  assert.equal($("s6-pinn-study").hidden,false);
  assert.ok($("s6-pinn-time").textContent.includes("頁面快照"));
}
// In-memory fixture only: exercise the seventh-method UI before real results
// exist. No fixture scores are ever written to reports or the delivered page.
const fixture = structuredClone(report);
fixture.five_models.sources.s6_pinn ||= {selected_epoch:1};
fixture.five_models.macro_mean.s6_pinn ||= fixture.five_models.macro_mean.s6;
fixture.five_models.pooled.s6_pinn ||= fixture.five_models.pooled.s6;
for (const record of fixture.five_models.records) record.metrics.s6_pinn ||= record.metrics.s6;
context.data=fixture;
for (const key of ["attack","release","ratio","threshold"]) $("five-"+key).value="all";
$("five-aggregate").value="macro";
$("five-sort").value="s6_pinn";
vm.runInContext(fs.readFileSync(path.join(__dirname,"five-models.js"),"utf8"),context);
assert.ok($("five-cards").innerHTML.includes("S6＋TFiLM＋PINN"));
assert.ok($("five-records").innerHTML.includes("S6＋TFiLM＋PINN"));
assert.ok($("five-sort").innerHTML.includes('value="s6_pinn"'));
$("five-csv").listeners.click();
assert.ok(csvCapture.split("\r\n")[0].includes(",s6_pinn,"));
$("five-aggregate").value="pooled";
$("five-aggregate").listeners.change();
assert.ok($("five-cards").innerHTML.includes(displayedNumber(fixture.five_models.pooled.s6_pinn.esr)));
console.log("PASS: PINN snapshot panel and seventh-method fixture cards, sorting, pooled ESR and CSV.");
