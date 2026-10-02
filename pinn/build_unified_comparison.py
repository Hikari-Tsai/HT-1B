"""Consolidated six-model report, one shared scoring protocol, one baseline."""
from datetime import datetime, timezone
import json
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT/'pinn/comparison-ui/unified'
RUN = ROOT/'runs/s6-tfilm-pinn-gpu-20260929'
MODEL_KEYS = ('mlp','gru','s4','s6','s6_pinn','riccardovib')


def wav_samples(path: Path) -> np.ndarray:
    """Read the retained mono float32 WAVs without adding an audio dependency."""
    raw = path.read_bytes()
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError(f'Invalid WAV: {path}')
    position, fmt, audio = 12, None, None
    while position + 8 <= len(raw):
        kind, size = struct.unpack_from('<4sI', raw, position)
        start = position + 8
        if kind == b'fmt ':
            fmt = struct.unpack_from('<HHIIHH', raw, start)
        elif kind == b'data':
            audio = raw[start:start+size]
        position = start + size + (size & 1)
    if fmt is None or audio is None or fmt[0] != 3 or fmt[1] != 1 or fmt[2] != 48000 or fmt[5] != 32:
        raise ValueError(f'Expected 48 kHz mono float32 WAV: {path}')
    return np.frombuffer(audio, dtype='<f4')


def waveform_summary(samples: np.ndarray, reference: np.ndarray, bins: int = 640) -> dict:
    """Non-reconstructable min/max display envelopes and RMS residual per bin."""
    if len(samples) != len(reference):
        raise ValueError('Audition tracks have different sample counts')
    edges = np.linspace(0, len(samples), bins + 1, dtype=np.int64)
    lows, highs, residual_lows, residual_highs, difference = [], [], [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        segment = samples[start:stop].astype(np.float64)
        residual = segment - reference[start:stop].astype(np.float64)
        lows.append(round(float(segment.min()), 5))
        highs.append(round(float(segment.max()), 5))
        residual_lows.append(round(float(residual.min()), 5))
        residual_highs.append(round(float(residual.max()), 5))
        difference.append(round(float(np.sqrt(np.mean(residual * residual))), 5))
    return dict(low=lows, high=highs, residual_low=residual_lows,
                residual_high=residual_highs, diff_rms=difference)


def historical_listening(audio_base: str = '../runs/pretrained-comparison/audio',
                         instrument_audio_base: str = '../site/audio/instrument-excerpts') -> dict:
    """Expose the retained local-only five-clip audition without mixing it into rankings."""
    source = ROOT/'runs/pretrained-comparison'
    report = read(source/'comparison.json')
    rendered = read(source/'audio/current-model-provenance.json')
    common_sources = read(RUN/'common-validation/comparison.json')['sources']
    for key in ('mlp','gru','s4','s6','s6_pinn'):
        if rendered['models'][key]['source']['sha256'] != common_sources[key]['sha256']:
            raise ValueError(f'Stale audition audio for {key}')
    tracks = (
        ('reference_dry', 'Dry input'),
        ('reference_wet', '實測 Wet'),
        ('mlp_full', 'MLP＋PINN · 完整訓練'),
        ('gru_full', 'GRU＋PINN · 完整訓練'),
        ('s4_tfilm', 'S4＋TFiLM · 最佳第 14 輪'),
        ('s6_tfilm', 'S6＋TFiLM · 最佳第 8 輪'),
        ('s6_tfilm_pinn', 'S6＋TFiLM＋PINN · 最佳第 15 輪'),
        ('author_pretrained', 'RiccardoVib'),
        ('pure_algorithm', '純演算法'),
        ('ours_5000', '歷史 MLP · 5,000 更新（額外版本）'),
    )
    records = []
    for row in report['records']:
        audio = {}
        waveforms = {}
        reference = wav_samples(source/'audio'/'reference_wet'/f"{row['id']}.wav")
        for key, label in tracks:
            path = source/'audio'/key/f"{row['id']}.wav"
            if not path.exists():
                raise ValueError(f'Missing historical audition WAV: {path}')
            audio[key] = dict(label=label, path=f'{audio_base}/{key}/{row["id"]}.wav')
            waveforms[key] = waveform_summary(wav_samples(path), reference)
        records.append(dict(id=row['id'], samples=row['samples_scored'], seconds=row['samples_scored']/48000,
                            conditioning=row['conditioning'], audio=audio, waveform=waveforms))
    excerpt_manifest = ROOT/'site/audio/instrument-excerpts/manifest.json'
    if excerpt_manifest.exists():
        excerpts = read(excerpt_manifest)
        for item in excerpts['records']:
            audio = {key:dict(value, path=f'{instrument_audio_base}/{item["asset_id"]}/{key}.wav')
                     for key,value in item['audio'].items()}
            records.append(dict(item, audio=audio))
    return dict(scope='historical_five_clip_plus_instrument_excerpts', records=records,
                note='Five historical 2-second clips plus matched instrument excerpts. These reuse existing recordings and are for listening only, not independent test results.')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_report(output, common, *, audio_base: str = '../runs/pretrained-comparison/audio',
                 instrument_audio_base: str = '../site/audio/instrument-excerpts', write_metrics: bool = True):
    if common is None or set(common['sources']) != {*MODEL_KEYS,'baseline'}:
        raise ValueError('Consolidated report requires all six models and the baseline')
    manifest = read(ROOT/'runs/full-corpus-gpu-20260925/manifest.json')
    histories = {}
    sequence = ROOT/'runs/s4-s6-representative-gpu-20260925'
    for key in ('s4','s6','s6_pinn'):
        folder = RUN if key == 's6_pinn' else sequence
        history = read(folder/('history.json' if key == 's6_pinn' else f'{key}_tfilm-history.json'))['history']
        protocol = read(folder/'protocol.json')
        epoch = common['sources'][key]['selected_epoch']
        if protocol['manifest_signature'] != common['manifest_signature'] or len(history) != protocol['epochs']:
            raise ValueError(f'Incomplete/mismatched history: {key}')
        selected = next(h for h in history if h['epoch'] == epoch)
        if not np.isclose(selected['validation_macro_mean']['esr'], common['macro_mean'][key]['esr'],rtol=2e-4):
            raise ValueError(f'History/checkpoint mismatch: {key}')
        histories[key] = dict(kind='epoch_validation',selected_epoch=epoch, epochs=len(history), rows=[dict(
            epoch=h['epoch'],esr=h['validation_macro_mean']['esr'],pooled_esr=h['validation_pooled']['esr'],
            training_loss=h.get('train_mean_batch_loss',h.get('train_mean_losses',{}).get('total')),
            audio_loss=h.get('train_mean_losses',{}).get('audio'),physics_loss=h.get('train_mean_losses',{}).get('physics'))
            for h in history])
    coverage = {}
    for key, folder in (('mlp','full-corpus-gpu-20260925'),('gru','gru-pinn-gpu-20260925')):
        c = read(ROOT/'runs'/folder/'training/coverage.json')
        if c['frames_covered'] != manifest['frames_by_split']['train'] or c['holdout_wet_training_samples'] != 0:
            raise ValueError(f'Invalid training coverage: {key}')
        coverage[key] = c['frames_covered']
        folder = ROOT/'runs'/folder
        config = read(folder/'training/run-config.json')
        status = read(folder/'training/status.json')
        fitted = read(ROOT/common['sources'][key]['path'])['metadata']
        if (config['manifest_signature'] != common['manifest_signature'] or
            config['options']['epochs'] != 1 or status['phase'] != 'training_complete' or
            status['frames_covered'] != c['frames_covered'] or
            fitted['step'] != status['step'] or not fitted['complete_training_pass']):
            raise ValueError(f'Incomplete/mismatched training history: {key}')
        if key == 'mlp':
            source = folder/'training/history.jsonl'
            raw = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
            rows = [dict(step=r['step'],training_loss=r['total'],audio_loss=r['data'],physics_loss=r['physics']) for r in raw]
            selection = read(folder/'selection.json')
            if selection['selected'] != 'fitted' or not selection['test_not_used']:
                raise ValueError('MLP final-parameter selection differs from displayed source')
            selection_note = '以完整區段驗證 ESR 比較原始參數與訓練後參數，採用訓練後參數；未逐步或逐輪搜尋最佳 checkpoint。'
            sampling_note = '每 50 步記錄當時的單批 loss；最後一筆紀錄早於訓練完成步數，不將舊 loss 補成最終步數的值。'
        else:
            source = folder.with_suffix('.stdout.log')
            raw = []
            for line in source.read_text(encoding='utf-8-sig').splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Non-JSON launcher/evaluator messages are not training observations.
                if row.get('phase') in ('training','training_complete') and row.get('latest_loss'):
                    raw.append(row)
            raw.append(status)
            # Repeated status reports at one optimizer step are one observation.
            snapshots = {r['step']:r for r in raw}
            rows = [dict(step=step,training_loss=r['latest_loss']['total'],
                         audio_loss=r['latest_loss']['data'],physics_loss=r['latest_loss']['physics'])
                    for step,r in sorted(snapshots.items())]
            selection_note = '採用完成一輪後的最終參數。歷史驗證比較架構表現，未在 GRU 多個輪次或 checkpoint 間選最佳。'
            sampling_note = '從 stdout 約每分鐘的狀態快照及完成狀態還原；每筆是當時的單批 loss，不是該分鐘平均，也未保存所有步數。'
        if (not rows or rows[-1]['step'] > status['step'] or
            any(a['step'] >= b['step'] for a,b in zip(rows,rows[1:])) or
            not all(np.isfinite(r[k]) for r in rows for k in ('training_loss','audio_loss','physics_loss'))):
            raise ValueError(f'Invalid loss observations: {key}')
        histories[key] = dict(kind='step_training',epochs=1,selected_epoch=None,
            final_step=status['step'],frames_covered=c['frames_covered'],
            source=str(source.relative_to(ROOT)).replace('\\','/'),
            sampling_note=sampling_note,selection_note=selection_note,rows=rows)
    # Verify the historical test coverage, but do not publish another ranking
    # of an incomplete subset on this common-window comparison page.
    old_tests = {'mlp':'full-corpus-gpu-20260925/selected-test.json',
                 'gru':'gru-pinn-gpu-20260925/gru-test.json',
                 'riccardovib':'riccardovib-full-corpus-20260925/test.json'}
    availability = {key:dict(common_validation=True,historical_full_test=False,new_independent_test=False)
                    for key in (*MODEL_KEYS,'baseline')}
    for key, rel in old_tests.items():
        path = ROOT/'runs'/rel
        if path.exists():
            test = read(path)
            if (test['split'] != 'test' or test['manifest_signature'] != common['manifest_signature'] or
                test['samples_scored'] != manifest['frames_by_split']['test'] or len(test['records']) != 620):
                raise ValueError(f'Invalid historical test metadata: {key}')
            availability[key]['historical_full_test'] = True
    report = dict(schema_version=3,built_utc=datetime.now(timezone.utc).isoformat(),
        model_keys=list(MODEL_KEYS),baseline_key='baseline',five_models=common,
        histories=histories,coverage=coverage,availability=availability,
        split_hours=manifest['hours_by_split'],split_samples=manifest['frames_by_split'],
        listening=historical_listening(audio_base, instrument_audio_base),
        cleanup=dict(removed_sections=['blind listening and partial full-corpus rankings',
            'standalone S4/S6 comparison','standalone S6/PINN summary','duplicate metric definitions'],
            scope='Presentation cleanup only. No new inference, training or test scores. Source artifacts retained.'))
    long_path=ROOT/'runs/long-clip-evaluation-20260929/comparison.json'
    if long_path.exists():
        long=read(long_path)
        if long['manifest_signature']!=common['manifest_signature'] or long['protocol']['smoke']:
            raise ValueError('Invalid long-clip evaluation provenance')
        if set(long['studies'])!=set('ABCDEFGH')|{'A1','A2','A3'}:
            raise ValueError('Incomplete long-clip studies')
        for key in (*MODEL_KEYS,'baseline'):
            if long['sources'][key]['sha256']!=common['sources'][key]['sha256']:
                raise ValueError('Long-clip checkpoint differs: '+key)
        for study in long['studies'].values():
            if len(study['records'])!=study['candidate']['files']:
                raise ValueError('Incomplete candidate coverage')
            for row in study['records']:
                if set(row['metrics'])!={*MODEL_KEYS,'baseline'} or not all(
                    np.isfinite(value) for scores in row['metrics'].values() for value in scores.values()):
                    raise ValueError('Invalid long-clip metrics')
        report['long_form']=long
        report['schema_version']=4
    payload=json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    html=(UI/'index.html').read_text(encoding='utf-8')
    attribution_href = 'AUDIO-ATTRIBUTION.md' if output.parent == ROOT/'site' else '../site/AUDIO-ATTRIBUTION.md'
    html=html.replace('href="AUDIO-ATTRIBUTION.md"',f'href="{attribution_href}"')
    html=html.replace('/*__DATA__*/',payload).replace('/*__STYLE__*/',(UI/'style.css').read_text(encoding='utf-8'))
    html=html.replace('/*__SCRIPT__*/',(UI/'core.js').read_text(encoding='utf-8')+'\n'+(UI/'app.js').read_text(encoding='utf-8'))
    if any(marker in html for marker in ('/*__DATA__*/','/*__STYLE__*/','/*__SCRIPT__*/')):
        raise ValueError('Unfilled HTML template')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.html.tmp');temp.write_text(html,encoding='utf-8');temp.replace(output)
    if write_metrics:
        metrics=output.with_suffix('.metrics.json')
        temp=metrics.with_suffix('.json.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(metrics)
    print('Verified all six models + baseline, fixed artifact hashes, window signature, histories and historical test coverage.')
    print(f'Built consolidated report: {output} ({output.stat().st_size/1024**2:.2f} MiB)')
