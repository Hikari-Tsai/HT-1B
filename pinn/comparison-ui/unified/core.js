/* Shared, independently testable report calculations. */
(function(root){
  'use strict';
  const METRICS={esr:'ESR',mse:'MSE',mae:'MAE',mrstft:'MR-STFT',rms_db_mae:'RMS 包絡 · dB'};
  const METHODS=[
    {key:'mlp',name:'MLP＋PINN',color:'#a46215'},
    {key:'gru',name:'GRU＋PINN',color:'#7663a3'},
    {key:'s4',name:'S4＋TFiLM',color:'#b45353'},
    {key:'s6',name:'S6＋TFiLM',color:'#327697'},
    {key:'s6_pinn',name:'S6＋TFiLM＋PINN',color:'#925d87'},
    {key:'riccardovib',name:'RiccardoVib',color:'#177a66'},
    {key:'baseline',name:'純演算法',color:'#667481'}
  ];
  const FIELDS={attack:'attack_index',release:'release_index',ratio:'ratio',threshold:'threshold_db'};
  function filterRows(rows,filters){return rows.filter(r=>Object.entries(FIELDS).every(([k,f])=>filters[k]==='all'||filters[k]===undefined||r.labels[f]===Number(filters[k])));}
  function aggregate(rows,key,metric,mode,floor=1e-8){
    if(!rows.length)return null;
    if(mode==='macro')return rows.reduce((s,r)=>s+r.metrics[key][metric],0)/rows.length;
    const count=rows.reduce((s,r)=>s+r.samples,0);
    if(metric==='esr')return rows.reduce((s,r)=>s+r.metrics[key].mse*r.samples,0)/Math.max(rows.reduce((s,r)=>s+r.target_energy,0),count*floor);
    return rows.reduce((s,r)=>s+r.metrics[key][metric]*r.samples,0)/count;
  }
  function summarize(rows,mode,floor){return Object.fromEntries(METHODS.map(m=>[m.key,Object.fromEntries(Object.keys(METRICS).map(k=>[k,aggregate(rows,m.key,k,mode,floor)]))]));}
  function fmt(v,metric='esr'){
    if(v===null||!Number.isFinite(v))return '—';
    if(metric==='mse'||(v!==0&&Math.abs(v)<1e-4))return v.toExponential(3);
    return v.toLocaleString('zh-TW',{minimumFractionDigits:metric==='mae'?6:3,maximumFractionDigits:metric==='mae'?6:4});
  }
  function csv(rows){return '\uFEFF'+rows.map(row=>row.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');}
  function recordCSV(rows,signature,scope='reused_validation'){return csv([
    ['scope','window_signature','record','attack','release','ratio','threshold_db','metric',...METHODS.map(m=>m.key)],
    ...rows.flatMap(r=>Object.keys(METRICS).map(k=>[scope,signature,r.id,r.labels.attack_index,r.labels.release_index,r.labels.ratio,r.labels.threshold_db,k,...METHODS.map(m=>r.metrics[m.key][k])]))
  ]);}
  const api={METRICS,METHODS,FIELDS,filterRows,aggregate,summarize,fmt,csv,recordCSV};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.ComparisonCore=api;
})(globalThis);
