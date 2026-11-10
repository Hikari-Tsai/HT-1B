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


FULL_TEST = ROOT/'runs/full-test-20261103/report.json'
FULL_TEST_EXTRA = (ROOT/'runs/full-test-s6-bounded-20261104/report.json',
                   ROOT/'runs/full-test-s6-pinn-20261106/report.json')
BOUNDED_RUN = ROOT/'runs/s6-bounded-macro-20261103-1045'
PINN_RUNS = {'s6_pinn_fixed': ROOT/'runs/s6-pinn-fixed-20261104-1434',
             's6_pinn_ramp': ROOT/'runs/s6-pinn-ramp-20261104-1442'}
# e/f/s reach on the original S6+PINN's failing test files: new checkpoints,
# the original model and the reduced circuit (original parameters), same files.
PINN_STATES = ROOT/'runs/full-test-s6-pinn-20261106/state-check.json'
# name, training note, selection status. 'clean': chosen on validation only, with a
# rule fixed before training; 'post_hoc': chosen after test scores were seen.
FULL_TEST_MODELS = {
    's6_bounded': ('S6＋TFiLM 有界 gain', '批次 TBPTT · 第 22／50 輪 · 驗證逐檔 macro ESR 選模', 'clean'),
    's6_pinn_ramp': ('S6＋TFiLM＋PINN · 物理權重漸增', '批次 TBPTT · 第 20／50 輪 · 驗證 macro ESR 選模', 'clean'),
    's6_v3_e40': ('S6＋TFiLM v3 · 第 40 輪', '批次 TBPTT · 看過測試結果後改選', 'post_hoc'),
    's6_pinn_fixed': ('S6＋TFiLM＋PINN · 物理權重固定', '批次 TBPTT · 第 37／50 輪 · 驗證 macro ESR 選模', 'clean'),
    'riccardovib': ('RiccardoVib', '固定權重 · 未重新訓練', 'reference'),
    's4_v3': ('S4＋TFiLM v3', '批次 TBPTT · 第 39／50 輪 · 驗證 pooled ESR 選模', 'clean'),
    's6': ('S6＋TFiLM', '代表視窗 · 16 輪最佳', 'clean'),
    'gru': ('GRU＋PINN', '完整訓練區間 · 學得電路參數', 'clean'),
    'mlp': ('MLP＋PINN', '完整訓練區間 · 學得電路參數', 'clean'),
    's6_pinn': ('S6＋TFiLM＋PINN（原版）', '代表視窗 · 16 輪最佳', 'clean'),
    's6_v3': ('S6＋TFiLM v3 · 第 38 輪', '批次 TBPTT · 驗證 pooled ESR 選模（原定）', 'clean'),
    's4': ('S4＋TFiLM', '代表視窗 · 16 輪最佳', 'clean'),
    'baseline': ('純演算法', '原始電路參數 · 未擬合', 'baseline'),
}


def full_test(common, manifest):
    """Complete test-interval scores, published only when every source checks out."""
    if not FULL_TEST.exists() or not all(path.exists() for path in FULL_TEST_EXTRA):
        return None
    main, extras = read(FULL_TEST), [read(path) for path in FULL_TEST_EXTRA]
    reports = [main, *extras]
    files = manifest['records']
    for report in reports:
        if (report['split'] != 'test' or report['manifest_signature'] != common['manifest_signature'] or
            report['files'] != len(files) or report['samples_scored'] != main['samples_scored'] or
            report['warmup_samples'] != main['warmup_samples'] or
            report['metric_protocol'] != main['metric_protocol']):
            raise ValueError('Full-test reports differ in split, files or protocol')
    records = {k: v for report in reports for k, v in report['records'].items()}
    sources = {k: v for report in reports for k, v in report['sources'].items()}
    summaries = {k: v for report in reports for k, v in report['summary'].items()}
    if set(records) != set(FULL_TEST_MODELS):
        raise ValueError('Full-test models differ from the published list')
    for key in (*MODEL_KEYS, 'baseline'):
        if sources[key]['sha256'] != common['sources'][key]['sha256']:
            raise ValueError(f'Full-test checkpoint differs from the common comparison: {key}')
    for key, source in sources.items():
        path = ROOT/source['path']
        if path.exists() and sha(path) != source['sha256']:
            raise ValueError(f'Full-test source changed after scoring: {key}')
    best, config = read(BOUNDED_RUN/'best.json'), read(BOUNDED_RUN/'config.json')
    if (config['training']['selection'] != 'macro_esr' or config['model']['output_gain'] != 'bounded' or
        sources['s6_bounded']['path'] != str((BOUNDED_RUN/best['checkpoint']/'model.mdlus').relative_to(ROOT)).replace('\\', '/')):
        raise ValueError('Bounded S6 source is not the validation-selected checkpoint')
    names = {r['id'] for r in files}
    rows = []
    for key, (name, note, status) in FULL_TEST_MODELS.items():
        if set(records[key]) != names:
            raise ValueError(f'Incomplete full-test records: {key}')
        esr = np.array([r['esr'] for r in records[key].values()])
        summary = summaries[key]
        if not np.isclose(summary['macro_mean']['esr'], esr.mean(), rtol=1e-9):
            raise ValueError(f'Full-test summary mismatch: {key}')
        rows.append(dict(key=key, name=name, note=note, status=status,
                         macro_esr=summary['macro_mean']['esr'], median_esr=summary['median_esr'],
                         pooled_esr=summary['pooled_esr'], mrstft=summary['macro_mean']['mrstft'],
                         rms_db_mae=summary['macro_mean']['rms_db_mae'],
                         audible_macro_esr=summary['audible']['macro_esr'], max_esr=float(esr.max()),
                         files_esr_above_1=int((esr > 1).sum()), files_esr_above_10=int((esr > 10).sum()),
                         source=sources[key]['path'], sha256=sources[key]['sha256']))
    rows.sort(key=lambda r: (r['status'] == 'baseline', r['macro_esr']))
    return dict(files=main['files'], samples_scored=main['samples_scored'], hours=main['samples_scored']/48000/3600,
                warmup_samples=main['warmup_samples'], audible_threshold_dbfs=main['audible_threshold_dbfs'],
                mrstft_fft_sizes=main['metric_protocol']['mrstft']['fft_sizes'],
                gpus=sorted({r['gpu'] for r in reports}), completed_utc=[r['completed_utc'] for r in reports],
                bounded_selection=dict(epoch=best['epoch'], epochs=config['training']['epochs'],
                                       validation_macro_esr=best['validation_score'],
                                       validation_pooled_esr=best['validation_esr']),
                rows=rows, pinn=pinn_study(records, summaries, sources),
                generalization=generalization(main, records))


def pinn_study(records, summaries, sources):
    """Original vs batched S6+PINN on the full test: scores, residuals, state reach."""
    failing = sorted(k for k, v in records['s6_pinn'].items() if v['esr'] > 1)
    states = read(PINN_STATES)
    runs = {}
    for key, run in PINN_RUNS.items():
        best, config = read(run/'best.json'), read(run/'config.json')
        path = str((run/best['checkpoint']/'model.mdlus').relative_to(ROOT)).replace('\\', '/')
        if (config['model']['architecture'] != 's6_tfilm_pinn' or config['training']['selection'] != 'macro_esr' or
            sources[key]['path'] != path or states['checkpoints'][key.removeprefix('s6_pinn_')] != path):
            raise ValueError(f'S6+PINN source is not the validation-selected checkpoint: {key}')
        runs[key] = dict(epoch=best['epoch'], epochs=config['training']['epochs'],
                         ramp_epochs=config['training']['physics_ramp_epochs'],
                         physics_weight=config['training']['physics_weight'],
                         validation_macro_esr=best['validation_score'],
                         physics_mse=summaries[key]['physics_mse'],
                         states=states['states'][key.removeprefix('s6_pinn_')])
    def failing_median(key):
        return float(np.median([records[key][k]['esr'] for k in failing]))
    if states['files'] != len(failing) or set(states['reference']) != {'circuit', 's6_pinn'}:
        raise ValueError('State check does not cover the failing files with both references')
    return dict(failing_files=len(failing), state_files=states['files'], reference_states=states['reference'],
                failing_median={key: failing_median(key) for key in ('s6_pinn', *PINN_RUNS, 'baseline', 's6_bounded')},
                runs=runs)


# P1 of s6/docs/plans/2026-11-06-pinn-generalization-experiments.md: threshold
# -40 dB cells held out of training and validation, every test interval scored.
T40_RUNS = {('s6', 7): ROOT/'runs/t40-s6-s7-20261106-1436', ('s6', 8): ROOT/'runs/t40-s6-s8-20261106-1436',
            ('struct', 7): ROOT/'runs/t40-struct-s7-20261106-1436', ('struct', 8): ROOT/'runs/t40-struct-s8-20261106-1436',
            ('pinn', 7): ROOT/'runs/t40-pinn-s7-20261106-1436', ('pinn', 8): ROOT/'runs/t40-pinn-s8-20261106-1436'}
# Replacement for the stuck STRUCT seed 8 (user decision 2026-11-10); not trained yet.
T40_PENDING = (('struct', 9),)
T40_PENDING_NOTE = '2026-11-10 起於 GPU 5 訓練中'
T40_REPLACED = {('struct', 8): '狀態卡住，以 seed 9 重跑取代'}
T40_TESTS = (ROOT/'runs/full-test-t40-s6-20261107/report.json',
             ROOT/'runs/full-test-t40-pinn-struct-20261110/report.json')
# Blow-up diagnosis of the held-out PINN failures (s6/docs/plans/2026-11-06-..., tools/blowup_diag.py).
T40_DIAGNOSIS = ROOT/'runs/blowup-diagnosis-20261110.json'
T40_INTERVALS = ROOT/'runs/v3/s6-v3-intervals.json'
T40_ARMS = {'s6': ('M-S6', 'S6＋TFiLM 有界 gain'),
            'struct': ('M-STRUCT', 'PINN 架構 · 物理權重 0'),
            'pinn': ('M-PINN', 'PINN · 物理權重 10 輪內由 0 升到 1')}


def generalization(main, full_records):
    """Held-out knob-cell pilot (T40): training status and split test scores."""
    if not T40_INTERVALS.exists() or not all(p.exists() for p in (*T40_RUNS.values(), *T40_TESTS)):
        return None
    from s6.holdout import SEED, cell, schemes
    intervals = read(T40_INTERVALS)
    cells = {r['id'].rsplit('-i', 1)[0]: cell(r['parameters']) for r in intervals['records'] if r['split'] == 'test'}
    heldout = schemes(set(cells.values()), SEED)['T40'][0]
    if len(cells) != main['files'] or len(heldout) != 124 or set(cells) != set(full_records['s6_bounded']):
        raise ValueError('T40 cells do not match the full-test files')
    runs, sources = [], {}
    for (arm, seed), run in T40_RUNS.items():
        config, best = read(run/'config.json'), read(run/'best.json')
        history = [json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        t, m = config['training'], config['model']
        expected = dict(s6=('s6_tfilm', None), struct=('s6_tfilm_pinn', (0., 0)), pinn=('s6_tfilm_pinn', (1., 10)))[arm]
        if (m['architecture'] != expected[0] or t['seed'] != seed or t['selection'] != 'macro_esr' or
            (expected[1] and (t['physics_weight'], t['physics_ramp_epochs']) != expected[1]) or
            (arm == 's6' and m['output_gain'] != 'bounded') or
            [h['epoch'] for h in history] != list(range(1, t['epochs']+1))):
            raise ValueError(f'T40 run differs from the planned arm: {arm} seed {seed}')
        scores = [h['validation']['macro_esr'] for h in history]
        chosen = history[best['epoch']-1]
        if best['epoch'] != int(np.argmin(scores))+1 or not np.isclose(best['validation_score'], min(scores), rtol=1e-12):
            raise ValueError(f'T40 best checkpoint is not the validation minimum: {arm} seed {seed}')
        physics = [h['validation'].get('physics_mse') for h in history]
        # Identical up to float summation order: the e/f/s trajectories never moved.
        frozen = arm == 'struct' and all(np.isclose(p[k], physics[0][k], rtol=1e-9) for p in physics for k in p)
        key = f't40_{arm}_s{seed}'
        sources[key] = str((run/best['checkpoint']/'model.mdlus').relative_to(ROOT)).replace('\\', '/')
        runs.append(dict(key=key, arm=arm, seed=seed, status='replaced' if (arm, seed) in T40_REPLACED else 'complete',
                         note=T40_REPLACED.get((arm, seed), ''), epochs=t['epochs'], best_epoch=best['epoch'],
                         validation_macro_esr=best['validation_score'], validation_median_esr=chosen['validation']['median_esr'],
                         physics_mse=chosen['validation'].get('physics_mse'), frozen_states=frozen,
                         curve=[dict(epoch=h['epoch'], macro_esr=h['validation']['macro_esr'],
                                     physics_weight=h.get('physics_weight')) for h in history]))
    if [r['key'] for r in runs if r['frozen_states']] != ['t40_struct_s8']:
        raise ValueError('Unexpected STRUCT state check result')
    for arm, seed in T40_PENDING:
        runs.append(dict(key=f't40_{arm}_s{seed}', arm=arm, seed=seed, status='pending', note=T40_PENDING_NOTE,
                         epochs=30, curve=[]))
    reports = [read(p) for p in T40_TESTS]
    records = {'s6_bounded': full_records['s6_bounded']}
    for report in reports:
        if (report['split'] != 'test' or report['manifest_signature'] != main['manifest_signature'] or
            report['files'] != main['files'] or report['samples_scored'] != main['samples_scored'] or
            report['warmup_samples'] != main['warmup_samples'] or report['metric_protocol'] != main['metric_protocol']):
            raise ValueError('T40 test report differs in split, files or protocol')
        for key, source in report['sources'].items():
            if source['path'] != sources[key] or (ROOT/source['path']).exists() and sha(ROOT/source['path']) != source['sha256']:
                raise ValueError(f'T40 test source is not the validation-selected checkpoint: {key}')
        records.update(report['records'])
    tests = []
    for key, rec in records.items():
        if set(rec) != set(cells):
            raise ValueError(f'Incomplete T40 test records: {key}')
        part = lambda inside: np.array([rec[f]['esr'] for f in cells if (cells[f] in heldout) == inside])
        held, seen = part(True), part(False)
        ratio = [np.log(rec[f]['esr'])-np.log(records['s6_bounded'][f]['esr']) for f in cells if cells[f] in heldout]
        tests.append(dict(key=key, heldout_macro_esr=float(held.mean()), heldout_median_esr=float(np.median(held)),
                          seen_macro_esr=float(seen.mean()), seen_median_esr=float(np.median(seen)),
                          macro_esr=float(np.mean([r['esr'] for r in rec.values()])),
                          heldout_files_esr_above_1=int((held > 1).sum()),
                          heldout_median_log_ratio_vs_full=float(np.median(ratio))))
    def paired(a, b, inside):
        x = np.array([np.log(records[a][f]['esr'])-np.log(records[b][f]['esr']) for f in cells if (cells[f] in heldout) == inside])
        boot = np.median(np.random.default_rng(20261110).choice(x, (5000, len(x))), axis=1)
        return dict(a=a, b=b, heldout=inside, median=float(np.median(x)), low=float(np.percentile(boot, 2.5)),
                    high=float(np.percentile(boot, 97.5)), better_share=float((x < 0).mean()), files=len(x))
    comparisons = [paired(a, b, inside) for a, b in (('t40_pinn_s7', 't40_s6_s7'), ('t40_pinn_s8', 't40_s6_s8'),
                                                       ('t40_struct_s7', 't40_s6_s7'), ('t40_pinn_s7', 't40_struct_s7'))
                   if a in records and b in records for inside in (True, False)]
    return dict(scheme='T40', description='Threshold −40 dB 整層', heldout_cells=len(heldout),
                training_cells=len(set(cells.values()))-len(heldout), heldout_files=int(sum(c in heldout for c in cells.values())),
                files=len(cells), arms={k: dict(code=c, name=n) for k, (c, n) in T40_ARMS.items()},
                runs=runs, tests=tests, scored=sorted(k for k in records if k != 's6_bounded'),
                comparisons=comparisons, diagnosis=blowup_diagnosis(records, cells, heldout),
                stopping_rule='兩個 seed 的 M-PINN 在 T40 留出檔上都比 M-S6 差（log ESR 差中位數 > 0）時先停下檢討，不進 P2。')


def blowup_diagnosis(records, cells, heldout):
    """Medians over the PINN blow-up files (held out) and their trained -30 dB twins."""
    if not T40_DIAGNOSIS.exists():
        return None
    d = read(T40_DIAGNOSIS)
    bad = sorted({f for k in ('t40_pinn_s7', 't40_pinn_s8') for f in cells if cells[f] in heldout and records[k][f]['esr'] > 1})
    twins = [f.replace('_t_40_', '_t_30_') for f in bad]
    if d['heldout'] != bad or d['files'] != bad+twins:
        raise ValueError('Blow-up diagnosis does not cover the current PINN blow-up files')
    models = list(d['models'])
    def med(files, model, key):
        values = [d['records'][f][model][key] for f in files if d['records'][f][model].get(key) is not None]
        return float(np.median(values)) if values else None
    groups = {}
    for name, files in (('heldout', bad), ('trained', twins)):
        groups[name] = {m: dict(**{k: med(files, m, k) for k in ('esr', 'level_error_db', 'gain_db_pred', 'gain_db_target',
                                                                   'worst_near_onset', 'e_max', 'f_p50')},
                                c3_residual=float(np.median([d['records'][f][m]['residual']['c3_kcl'] for f in files]))
                                if 'residual' in d['records'][files[0]][m] else None)
                        for m in models}
    return dict(files=len(bad), models=d['models'], groups=groups, parameters=d['parameters'],
                frames_near_onset=med(bad, models[0], 'frames_near_onset'))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    tested = full_test(common, manifest)
    if tested:
        report['full_test'] = tested
        for key in availability:
            availability[key]['new_independent_test'] = True
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
