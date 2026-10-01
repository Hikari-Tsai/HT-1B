"""Create a reviewable audio-selection proposal without changing any experiment."""
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/test-candidates'
RATE=48000
CANDIDATES=[
 dict(id='A',start=90.,stop=107.7,title='掃頻 → 鼓組 → Bass',category='heldout',
      content='約 90–95 秒為掃頻，95–100 秒為鼓組，100 秒後為 Bass；這是一段連續多素材序列。',
      reason='保留完整有聲段及自然停頓，觀察切換素材時的狀態延續；分素材另列成績。',checks=[90,95,100]),
 dict(id='B',start=199.7,stop=210.,title='鼓組／打擊樂',category='heldout',
      content='模型在 200–205 與 205–210 秒皆辨識為鼓組，包含瞬態和衰減。',
      reason='現有 test 區段，所有 620 檔都有；適合檢查起音、增益恢復與節奏。',checks=[200,205]),
 dict(id='C',start=70.,stop=80.,title='歌唱人聲',category='diagnostic',
      content='五個代表錄音皆支持歌唱人聲。前半段的人聲類別有分歧，因此不判定歌手性別或身分。',
      reason='補足人聲持續音與音節動態；曾用於訓練，只能當診斷或未來重切分的候選。',checks=[70,75]),
 dict(id='D',start=60.,stop=70.,title='Bass／低音撥弦',category='diagnostic',
      content='60–65 與 65–70 秒在五個代表錄音皆以 Bass 為首選。',
      reason='觀察低頻音符的起音、延音與音符之間的恢復；曾用於訓練。',checks=[60,65]),
 dict(id='E',start=35.,stop=40.,title='吉他撥弦（偏木吉他）',category='diagnostic',
      content='五個代表錄音首選皆為木吉他，次選電吉他；樂器細分類仍待試聽确认。',
      reason='原始單一吉他素材約 5 秒；保留連續原片段，不拼貼其他時間湊長度。曾用於訓練。',checks=[35]),
 dict(id='F',start=149.4,stop=160.,title='掃頻 → 停頓 → 短打擊與轉場',category='special',
      content='150–155 秒的有聲部分為掃頻；154.7 秒落在停頓。155 秒後為短打擊，末端混有轉場素材待試聽。',
      reason='將舊的 21.33 ms 靜音取樣延伸為完整前後脈絡。與 A 的掃頻有重複證據，專項另列、不重複加權。',checks=[150,155]),
 dict(id='G',start=210.,stop=219.8,title='尾端 Bass（部分錄音才有）',category='special',
      content='四個有尾段的代表錄音均辨識為 Bass；Release index 0 的錄音在 210 秒結束。',
      reason='可補充保留資料中的 Bass；只有 495 檔，不能與 620 檔主表直接合併平均。',checks=[210,215]),
 dict(id='H',start=33.,stop=35.,title='近靜音／底噪',category='special',
      content='用原始 Dry 的 100 ms RMS 判斷低能量，不以樂器分類器替靜音命名。',
      reason='單獨檢查零輸入殘留與底噪。僅 2 秒、不放入主音樂排名，也不放大試聽音量。',checks=[]),
]

def features(x):
    x=x.astype(np.float64)
    a=x[:len(x)//4800*4800].reshape(-1,4800)
    db=10*np.log10(np.maximum(np.mean(a*a,axis=1),1e-20))
    return dict(rms_dbfs=float(10*np.log10(max(np.mean(x*x),1e-20))),
                peak_dbfs=float(20*np.log10(max(np.max(np.abs(x)),1e-10))),
                quiet_fraction=float(np.mean(db<-65)))

def wave(x):
    parts=np.array_split(x,240)
    vals=np.array([max(abs(p.min()),abs(p.max())) for p in parts])
    vals=vals/max(vals.max(),1e-10)*22
    return '<svg viewBox="0 0 720 52" class="wave" aria-hidden="true">'+''.join(f'<line x1="{i*3}" x2="{i*3}" y1="{26-v:.2f}" y2="{26+v:.2f}"/>' for i,v in enumerate(vals))+'</svg>'

def build():
    official=ROOT/'output/cl1b-comparison.html'
    before=hashlib.sha256(official.read_bytes()).hexdigest()
    manifest=json.loads((ROOT/'runs/full-corpus-gpu-20260925/manifest.json').read_text())
    audit=json.loads((OUT/'content-audit.json').read_text())
    cross=json.loads((OUT/'content-crosscheck.json').read_text())
    reference=next(r for r in manifest['records'] if r['path']==audit['reference'])
    pair,sr=sf.read(ROOT/'data/tubetech-cl-1b-v1'/reference['path'],dtype='float32',always_2d=True)
    assert sr==RATE
    dry=pair[:,0]
    (OUT/'audio').mkdir(exist_ok=True)
    proposals=[];windows=[];cards={};esc=html.escape
    for original in CANDIDATES:
        c=dict(original);c['content']=c['content'].replace('确认','確認')
        x=dry[round(c['start']*sr):round(c['stop']*sr)]
        c['seconds']=len(x)/sr;c['reference_features']=features(x)
        c['reference_path']=reference['path'];c['label_status']='machine_inference_pending_listening'
        c['crosschecks']=[r for r in cross['checks'] if r['start'] in c['checks']]
        if c['id'] in ('A','F'):
            tone_start,tone_stop=(90,95) if c['id']=='A' else (150,155)
            f,t,z=stft(dry[tone_start*sr:tone_stop*sr],fs=sr,nperseg=4096,noverlap=3616,boundary=None)
            power=np.abs(z)**2;active=power.sum(axis=0)>1e-7
            dominant=f[np.argmax(power,axis=0)][active]
            c['spectral_check']=dict(start=tone_start,stop=tone_stop,fft_size=4096,
                active_power_threshold=1e-7,dominant_hz_quantiles=np.quantile(dominant,[.1,.5,.9]).tolist(),
                frequency_time_rank_correlation=float(spearmanr(t[active],dominant).statistic))
            if c['spectral_check']['frequency_time_rank_correlation']>.9:
                c['content']+=' 時頻峰值隨時間上升，另以訊號分析支持由低至高掃頻的判讀。'
        if c['id']=='H':c['label_status']='measured_low_energy_not_source_identity'
        c['preview_gain_db']=0. if c['id']=='H' else min(24.,-6.-c['reference_features']['peak_dbfs'])
        c['raw_audio']=f'audio/{c["id"]}-dry-original.wav'
        c['preview_audio']=f'audio/{c["id"]}-dry-preview.wav'
        sf.write(OUT/c['raw_audio'],x,sr,subtype='FLOAT')
        sf.write(OUT/c['preview_audio'],x*10**(c['preview_gain_db']/20),sr,subtype='FLOAT')
        back,_=sf.read(OUT/c['raw_audio'],dtype='float32');assert np.array_equal(back,x)
        applicable=[];splits=set();groups=set()
        representative_stats=[]
        for r in manifest['records']:
            a=round((c['start']+r['content_offset_seconds'])*sr)
            b=round((c['stop']+r['content_offset_seconds'])*sr)
            if a<0 or b>r['frames']:continue
            overlaps=[dict(split=i['split'],group=i['group'],samples=max(0,min(b,i['stop_sample'])-max(a,i['start_sample'])))
                      for i in r['intervals'] if min(b,i['stop_sample'])>max(a,i['start_sample'])]
            assert sum(i['samples'] for i in overlaps)==b-a
            splits.update(i['split'] for i in overlaps);groups.update(i['group'] for i in overlaps)
            applicable.append(r['id'])
            windows.append(dict(candidate=c['id'],record=r['id'],path=r['path'],dry_context_start_sample=0,
                score_start_sample=a,score_stop_sample=b,original_split='+'.join(sorted(set(i['split'] for i in overlaps))),
                overlap_details=overlaps,content_offset_seconds=r['content_offset_seconds']))
            if r['id'] in manifest['representatives']:
                with sf.SoundFile(ROOT/'data/tubetech-cl-1b-v1'/r['path']) as f:
                    f.seek(a);block=f.read(b-a,dtype='float32',always_2d=True)
                representative_stats.append(dict(reference=r['path'],**features(block[:,0])))
        c['files']=len(applicable);c['original_splits']=sorted(splits);c['original_groups']=sorted(groups)
        c['representative_features']=representative_stats
        c['full_training_overlap']='train' in splits;c['independent_test']=False
        assert c['files']==(495 if c['id']=='G' else 620)
        c['repeated_source_note']='掃頻約 92–94 與 152–154 秒具高相關，A/F 屬同一素材群。' if c['id'] in ('A','F') else ''
        proposals.append(c)
        split_label='/'.join(c['original_splits'])
        detail=f'{len(c["crosschecks"])} 個 5 秒左右子窗／跨代表錄音分類' if c['checks'] else '5 個代表錄音的原始音量檢查'
        cards[c['id']]=f'''<article class="candidate" id="candidate-{c['id']}"><div class="candidate-title"><h3>{c['id']} · {esc(c['title'])}</h3><span>{c['start']:g}–{c['stop']:g} 秒 · {c['seconds']:g} 秒</span></div>
<p>{esc(c['content'])}</p><p class="muted">{esc(c['reason'])}</p>
<div class="candidate-meta"><span>原切分：<b>{split_label}</b></span><span>可用 {c['files']}／620 檔</span><span>代表 Dry RMS {c['reference_features']['rms_dbfs']:.1f} dBFS</span><span>低於 −65 dBFS 的 100 ms 格點：{c['reference_features']['quiet_fraction']:.1%}</span></div>
{wave(x)}<div class="listen"><label>試聽版本<select data-audio="audio-{c['id']}"><option value="test-candidates/{c['preview_audio']}">辨識內容用（{c['preview_gain_db']:+.1f} dB）</option><option value="test-candidates/{c['raw_audio']}">原始音量 Dry</option></select></label><audio id="audio-{c['id']}" controls preload="none" src="test-candidates/{c['preview_audio']}"></audio><a download href="test-candidates/{c['raw_audio']}">下載原始片段</a></div>
<details class="method-detail"><summary>辨識依據與覆蓋</summary><p>{esc(detail)}。這是模型推定，尚未人工試聽定案；相似度不是正確率。</p><p>舊素材群：{', '.join(map(str,c['original_groups']))}。{esc(c['repeated_source_note'])}</p></details></article>'''
    proposed=dict(status='draft_for_discussion_not_executed',created_utc=datetime.now(timezone.utc).isoformat(),
       reference_path=reference['path'],sample_rate=sr,source_manifest_signature=manifest['signature'],
       content_classifier=dict(model=audit['model_id'],revision=audit['revision'],device='cpu'),
       candidates=proposals,windows=windows,
       protocol_proposal=dict(context='Replay original Dry continuously from recording start; score candidates only. Do not concatenate disjoint clips or reset at clip start.',
          comparison='Same candidate, controls, raw levels, target and metric protocol across all seven methods; no test-based gain/delay fitting.',
          categories='Report active music, transitions, and quiet separately; do not weight quiet equally with music or double-weight repeated sweeps.',
          independence='All existing holdouts have already been inspected. Training-overlap candidates require retraining with source-group exclusion for future held-out evaluation; upstream model overlap remains unknown.',
          target_length='10–17.7 s where coherent content permits; 5 s guitar and 2 s quiet exceptions. G uses 495 files separately.',
          scores='No new compressor inference or scores. Existing comparison page and historical split untouched.'))
    (OUT/'proposal.json').write_text(json.dumps(proposed,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'candidate-windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['candidate','record','path','dry_context_start_sample','score_start_sample','score_stop_sample','original_split','content_offset_seconds']
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(windows)
    table=''.join(f'<tr><th><a href="#candidate-{c["id"]}">{c["id"]} · {esc(c["title"])}</a></th><td>{c["start"]:g}–{c["stop"]:g}</td><td>{c["seconds"]:g} s</td><td>{"/".join(c["original_splits"])}</td><td>{c["files"]}</td></tr>' for c in proposals)
    envelope=dry[:len(dry)//4800*4800].reshape(-1,4800)
    db=10*np.log10(np.maximum(np.mean(envelope.astype(float)**2,axis=1),1e-12))
    points=' '.join(f'{50+i/10/220*1000:.2f},{25-np.clip(v,-120,0)/120*100:.2f}' for i,v in enumerate(db))
    colors={'train':'#c9b7a0','validation':'#80ae95','test':'#779bb9'}
    timeline='<svg viewBox="0 0 1100 175" role="img" aria-label="完整 220 秒 Dry 的 100 ms RMS 與舊切分">'
    for value in (0,-40,-80,-120):
        y=25-value/120*100;timeline+=f'<line class="grid" x1="50" x2="1050" y1="{y}" y2="{y}"/><text x="43" y="{y+4}" text-anchor="end">{value}</text>'
    timeline+=f'<polyline class="curve" points="{points}"/>'
    for t in manifest['timeline']:
        timeline+=f'<rect x="{50+t["start_seconds"]/220*1000}" y="137" width="{(t["stop_seconds"]-t["start_seconds"])/220*1000}" height="9" fill="{colors[t["split"]]}"><title>{t["start_seconds"]}–{t["stop_seconds"]} s / {t["split"]}</title></rect>'
    for t in range(0,221,20):timeline+=f'<text x="{50+t/220*1000}" y="165" text-anchor="middle">{t}s</text>'
    timeline+='</svg>'
    css=(ROOT/'pinn/comparison-ui/unified/style.css').read_text(encoding='utf-8')
    extra='.candidate{padding:25px 0;border-top:1px solid var(--line)}.candidate-title{display:flex;justify-content:space-between;gap:16px}.candidate-title h3{font-size:20px}.candidate-title span{font:14px Consolas,monospace;white-space:nowrap}.candidate p{font-size:14px;max-width:980px}.candidate-meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:12px}.wave{width:100%;height:56px;margin:14px 0}.wave line{stroke:var(--green);stroke-width:1.4}.listen{display:flex;align-items:end;gap:20px}.listen label{min-width:235px}.listen a{font-size:12px;padding-bottom:14px}audio{width:320px;max-width:100%}.chart svg{max-height:200px}.proposal-note{font-size:14px;line-height:1.9}.candidate .method-detail{margin-top:18px}.intro{padding-bottom:26px}h1{font-size:40px}@media(max-width:680px){.candidate-title,.listen{flex-direction:column;align-items:stretch}.listen label{min-width:0}.listen a{padding:0}.candidate-title span{white-space:normal}audio{width:100%}}'
    doc=f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HT-1B｜長片段測試候選・討論稿</title><style>{css}{extra}</style></head><body>
<header class="topbar"><a class="brand" href="cl1b-comparison.html">HT—1B <small>TEST SELECTION</small></a><nav><a href="#overview">候選總表</a><a href="#listen">試聽</a><a href="#decision">討論事項</a></nav></header><main>
<section class="intro"><div><p class="eyebrow">DRAFT / NO NEW MODEL SCORES</p><h1>先確認聲音，<br>再決定怎麼測。</h1><p class="lede">8 組候選。主要片段延長至 10–17.7 秒。<br>保留自然停頓，靜音與重複素材另外評估。</p></div><div class="proposal-note"><strong>這份是素材選擇討論稿。</strong><p>原訓練切分、模型權重與結果頁均未變動，尚未執行新一輪模型比較。內容標籤由本機 CLAP 與聲學檢查推定，附原始片段，待你試聽確認。</p></div></section>
<aside class="scope"><strong>測試範圍</strong><p>A/B 沒有參與本方訓練，但已做過歷史驗證／測試；C/D/E 曾用於訓練，只能做診斷。若要把它們變成正式保留測試，需先重切素材群並重新訓練本方模型。RiccardoVib 的原訓練重疊仍未知。</p></aside>
<section class="section" id="overview"><div class="section-head"><div><p class="eyebrow">01 / CANDIDATE MAP</p><h2>完整脈絡與候選時間</h2></div><a href="test-candidates/candidate-windows.csv" download>逐檔候選位置 CSV ↗</a></div><div class="chart">{timeline}</div><p class="helper">上：Dry RMS（dBFS）；下：舊切分，棕＝train、綠＝validation、藍＝test。參考檔長 220 秒；125 檔只到 210 秒。620 檔共享大致素材順序，並非 620 首不同歌曲。</p><div class="table-wrap"><table><thead><tr><th>內容候選</th><th>時間（秒）</th><th>長度</th><th>舊切分</th><th>可用檔數</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section class="section" id="listen"><p class="eyebrow">02 / LISTEN & IDENTIFY</p><h2>逐段試聽</h2><p class="helper">播放的是參考檔 Dry。預設只為辨識內容調整試聽增益（上限 +24 dB），不改變任何評分資料；可切換原始音量。波形各自縮放，不能用圖高比較片段音量。每次只播放一段。</p><h3>既有保留資料：可先做較長片段的探索性比較</h3>{cards['A']}{cards['B']}<h3>補充素材：現有模型已看過，需與保留資料分開</h3>{cards['C']}{cards['D']}{cards['E']}<h3>專項／可選片段：不直接併入主排名</h3>{cards['F']}{cards['G']}{cards['H']}</section>
<section class="section" id="decision"><p class="eyebrow">03 / BEFORE EVALUATION</p><h2>選好後要一起決定的事</h2><ol class="proposal-note"><li>先確認 C 的人聲、D 的 Bass、E 的吉他，以及 F 的混合轉場標籤是否符合試聽。</li><li>若先檢查現有模型：A/B 作保留資料探索性比較，C/D/E 作已見素材診斷，F/G/H 專項另列。不要把這些混成一個新的獨立測試排名。</li><li>若要重新建立正式測試集：凍結內容清單後，整個素材群及其重複位置都排除於訓練；本方模型需重新訓練。挑選長度或素材不能依哪個模型得分較好。</li><li>延長計分片段之外，還要保留記憶：建議從原錄音開頭連續輸入 Dry，只在候選範圍計分，避免 85 ms 暖機不足。不可把不同時間的音訊拼接成假連續片段。</li><li>五項誤差用同一套長片段規格重算七方法；有聲音樂、停頓／底噪、重複掃頻分開報告，並補充分布與試聽。A 應另列掃頻、鼓、Bass，避免單一類別掩蓋差異。</li></ol><p class="helper">本次只產生候選位置；內容分類與時間邊界分析使用 Dry。既有模型結果先前已看過，因此這是探索性重選，不是盲選的新獨立測試。內容自動辨識以五個 Release 家族的代表錄音交叉檢查；它無法證明原歌曲或演奏者獨立，也不能取代人工試聽。</p></section>
<details class="section method-detail"><summary>資料、方法與下載</summary><p class="small">參考錄音：{esc(reference['path'])}；48 kHz、左 Dry／右 Wet。只讀 Dry 做辨識。原樣本時間依既有 offset 映射；一檔提前 0.75 秒。全部候選已檢查長度、舊切分重疊與可用檔數。</p><p class="small">分類器：<a href="https://huggingface.co/laion/clap-htsat-unfused">LAION CLAP</a>，CPU，revision {audit['revision']}。先以全長 5 秒格點盤點，再用五個代表錄音交叉檢查；cosine 分數不是辨識正確率。未宣稱已人工聽辨。</p><p class="small">音訊來源：Riccardo Simionato，<a href="https://www.kaggle.com/datasets/riccardosimionato/tubetech-cl-1b/versions/1">Kaggle TubeTech CL 1B v1</a>，依下載時標示 <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>。本頁試聽檔為裁切衍生片段，preview 另調整增益，原始音量檔保留浮點樣本。</p><p><a href="test-candidates/proposal.md">Markdown 討論稿</a> · <a href="test-candidates/proposal.json">候選與映射 JSON</a> · <a href="test-candidates/content-audit.json">全長內容盤點</a> · <a href="test-candidates/content-crosscheck.json">跨錄音辨識紀錄</a></p></details><footer>HT—1B / DISCUSSION COPY · 尚未執行新模型評估</footer></main>
<script>document.querySelectorAll('select[data-audio]').forEach(s=>s.addEventListener('change',()=>{{const a=document.getElementById(s.dataset.audio);a.pause();a.src=s.value;a.load();}}));document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>{{document.querySelectorAll('audio').forEach(b=>{{if(b!==a)b.pause();}});}}));</script></body></html>'''
    (ROOT/'output/test-candidates.html').write_text(doc,encoding='utf-8')
    lines=['# 長片段測試候選：討論稿','','尚未變更切分、重新訓練或評分。內容為 CLAP 推定並經五個代表錄音交叉檢查，尚待人工試聽確認。','',
           '[開啟含試聽的候選頁](../test-candidates.html)','','| 候選 | 時間 | 長度 | 內容 | 舊切分 | 檔數 |','|---|---|---:|---|---|---:|']
    for c in proposals:lines.append(f'| {c["id"]} | {c["start"]:g}–{c["stop"]:g}s | {c["seconds"]:g}s | {c["title"]} | {"/".join(c["original_splits"])} | {c["files"]} |')
    lines+=['','## 建議討論方案','','- A/B：既有保留資料上的長片段探索性比較，並非新的獨立測試。',
      '- C/D/E：補充人聲、Bass、吉他，但原本是 train；現有模型只能作已見素材診斷。要成為未見資料評估，需先排除整個素材群及重複音訊並重訓本方模型。',
      '- F：掃頻和 A 重複，單列轉場／衰減測試。G：尾端 Bass 只有 495 檔，單獨固定條件比較。H：底噪單列，不與有聲素材等權混排。',
      '- 計分前從原錄音起點連續送入 Dry；不在片段起點重設狀態、不拼接音訊。分類時的音量正規化及試聽增益絕不進入正式評分。',
      '- 七方法固定相同來源、旋鈕、原始電平與長片段指標規格。RiccardoVib 原訓練重疊仍未知。',
      '- 決定候選後才執行，現在沒有新增成績。','','## 各段依據','']
    for c in proposals:
        lines += [f'### {c["id"]} · {c["title"]}', '',c['content'],c['reason'],
                  f'原始 Dry RMS：{c["reference_features"]["rms_dbfs"]:.1f} dBFS；低於 −65 dBFS 格點占比：{c["reference_features"]["quiet_fraction"]:.1%}。',
                  f'[原音量試聽]({c["raw_audio"]}) · [內容辨識用試聽]({c["preview_audio"]})','']
    lines+=['## 來源','','音訊：Riccardo Simionato / Kaggle TubeTech CL 1B v1（CC BY-SA 4.0），此處為裁切片段。',
            'https://www.kaggle.com/datasets/riccardosimionato/tubetech-cl-1b/versions/1',
            '分類器：https://huggingface.co/laion/clap-htsat-unfused',
            'CLAP cosine 是候選描述相似度，不是可靠度百分比，也不是原始素材名稱。']
    (OUT/'proposal.md').write_text('\n'.join(lines),encoding='utf-8')
    assert hashlib.sha256(official.read_bytes()).hexdigest()==before
    print(json.dumps(dict(candidates=[dict(id=c['id'],seconds=c['seconds'],files=c['files'],splits=c['original_splits'],quiet=c['reference_features']['quiet_fraction']) for c in proposals],window_count=len(windows),official_page_unchanged=True),indent=2))

if __name__=='__main__':build()
