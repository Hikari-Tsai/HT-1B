# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare the author's CL1B ED weights with our circuit outputs on shared samples.

Architecture follows Riccardo Simionato's Models.py (2023, LGPL-3.0-or-later):
https://github.com/RiccardoVib/CONDITIONED-MODELING-OF-OPTICAL-COMPRESSOR
Paper: Simionato and Fasciani, DAFx 2023, Fully Conditioned and Low-Latency
Black-Box Modeling of Analog Compression. See downloaded LICENSE.txt.
The CUDA adapter must pass native TensorFlow parity before any scoring.
"""
import hashlib
import itertools
import json
from pathlib import Path
import time

import h5py
import numpy as np
import soundfile as sf
import torch
from torch import nn

from ht1b.audio import metrics


class AuthorCL1B(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.conv_h=nn.Linear(16,64)
        self.conv_c=nn.Linear(16,64)
        self.cond_h=nn.Linear(4,64)
        self.cond_c=nn.Linear(4,64)
        self.decoder=nn.LSTM(1,64,batch_first=True)
        self.dense=nn.Linear(64,64)
        self.output=nn.Linear(64,16)
        with h5py.File(weights,'r') as f, torch.no_grad():
            def array(path): return torch.from_numpy(np.array(f[path]))
            for layer,key in [(self.conv_h,'Conv_h'),(self.conv_c,'Conv_c'),
                              (self.cond_h,'Dense_cond_h'),(self.cond_c,'Dense_cond_c'),
                              (self.dense,'DenseLay'),(self.output,'OutLay')]:
                kernel=array(f'{key}/{key}/kernel:0')
                layer.weight.copy_(kernel.reshape(-1,kernel.shape[-1]).T)
                layer.bias.copy_(array(f'{key}/{key}/bias:0'))
            base='LSTM_De/LSTM_De/lstm_cell/'
            self.decoder.weight_ih_l0.copy_(array(base+'kernel:0').T)
            self.decoder.weight_hh_l0.copy_(array(base+'recurrent_kernel:0').T)
            self.decoder.bias_ih_l0.copy_(array(base+'bias:0'))
            self.decoder.bias_hh_l0.zero_()

    def forward(self, windows, conditioning):
        h=self.conv_h(windows[:,:16])+self.cond_h(conditioning)
        c=self.conv_c(windows[:,:16])+self.cond_c(conditioning)
        _,(h,_)=self.decoder(windows[:,16:,None],(h[None],c[None]))
        return self.output(torch.sigmoid(self.dense(h[0])))


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for the adapter')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(7)
    root=Path('runs/pretrained-comparison').resolve()
    repo=Path('data/external/optical-compressor').resolve()
    weights=repo/'Fully_Conditioned_Black_Box_Model_for_Compression/Models/CL1BModel/Checkpoints/best/weights.h5'
    model=AuthorCL1B(weights).double().cuda().eval()
    def infer(windows,conditions):
        with torch.no_grad():
            return np.concatenate([model(torch.as_tensor(windows[i:i+1024],device='cuda',dtype=torch.float64),
                torch.as_tensor(conditions[i:i+1024],device='cuda',dtype=torch.float64)).cpu().numpy()
                for i in range(0,len(windows),1024)])
    native=np.load(root/'tensorflow-reference.npz')
    check=infer(native['windows'],native['conditioning'])
    np.testing.assert_allclose(check,native['predictions'],atol=2e-10,rtol=2e-9)
    parity=dict(max_absolute_error=float(np.max(np.abs(check-native['predictions']))),
                reference_blocks=len(check),parameters=sum(p.numel() for p in model.parameters()),
                note='Torch LSTM has an extra all-zero recurrent bias; effective learned parameters match TensorFlow.')
    print('TENSORFLOW PARITY',parity,flush=True)
    pilot=Path('runs/gpu-pilot-20260925-0300').resolve()
    continuation=Path('runs/gpu-continuation-2000-to-5000').resolve()
    rows=json.loads((pilot/'manifest.json').read_text())['records']
    audio={}
    for row in rows:
        x,sr=sf.read(row['stereo_pair'],frames=row['stop_sample'],dtype='float32',always_2d=True)
        if sr!=48000 or x.shape[1]!=2: raise ValueError('Expected stereo 48kHz data')
        audio[row['id']]=x
    # Missing in upstream: named order of the four normalized condition columns.
    # Resolve only on OUR TRAIN split; do not use held-out recordings for selection.
    fields=('attack','release','ratio','threshold')
    def values(row,flip):
        p=row['metadata']['filename_parameters']
        value=[p['attack_index']/4,p['release_index']/4,(p['ratio']-2)/8,abs(p['threshold_db'])/40]
        if flip: value[3]=1-value[3]
        return np.array(value,dtype=np.float32)
    calibration=[]
    for row in rows:
        if row['split']!='train': continue
        x=audio[row['id']]
        starts=np.array(list(range(0,len(x)-32,16)))
        starts=starts[np.linspace(0,len(starts)-1,128,dtype=int)]
        windows=np.stack([x[t:t+32,0] for t in starts])
        targets=np.stack([x[t+16:t+32,1] for t in starts])
        calibration.append((row,windows,targets))
    all_windows=np.concatenate([x[1] for x in calibration])
    candidates=[]
    for flip in (False,True):
        for permutation in itertools.permutations(range(4)):
            conditions=np.concatenate([np.tile(values(row,flip)[list(permutation)],(len(w),1)) for row,w,y in calibration])
            prediction=infer(all_windows,conditions)
            cursor=0; errors=[]
            for row,w,y in calibration:
                p=prediction[cursor:cursor+len(w)];cursor+=len(w)
                errors.append(metrics(p.ravel(),y.ravel())['esr'])
            candidates.append(dict(order=[fields[i] for i in permutation],permutation=list(permutation),
                threshold_flipped=flip,training_macro_esr=float(np.mean(errors))))
    candidates.sort(key=lambda x:x['training_macro_esr'])
    selected=candidates[0]
    (root/'conditioning-calibration.json').write_text(json.dumps(dict(selected=selected,candidates=candidates,
        calibration_records=[x[0]['id'] for x in calibration],blocks_per_record=128,
        selection='20 training recordings only; no validation targets used',
        caveat='Order and threshold direction empirically inferred because the upstream serialized-data preprocessing is missing.'),indent=2))
    print('CONDITIONING',selected,'runner_up',candidates[1],flush=True)
    progression=json.loads((continuation/'progression.json').read_text())
    best_step=progression['best_step']
    results=[]
    for row in rows:
        if row['split']!='validation': continue
        name=row['id']; x=audio[name]
        starts=np.array(list(range(0,len(x)-32,16)))
        windows=np.stack([x[t:t+32,0] for t in starts])
        condition=values(row,selected['threshold_flipped'])[selected['permutation']]
        conditions=np.tile(condition,(len(windows),1))
        # Warm up on representative blocks before recording batch inference time.
        infer(windows[:32],conditions[:32]);torch.cuda.synchronize()
        started=time.perf_counter()
        predicted=infer(windows,conditions).ravel()
        seconds=time.perf_counter()-started
        begin=16;end=int(starts[-1]+32)
        assert end-begin==len(predicted)
        outputs={'author_pretrained':predicted}
        paths={
            'pure_algorithm':pilot/'audio-baseline'/f'{name}.wav',
            'ours_2000':pilot/'audio-fitted'/f'{name}.wav',
            'ours_3000':continuation/'step-3000/audio-validation'/f'{name}.wav',
            'ours_4000':continuation/'step-4000/audio-validation'/f'{name}.wav',
            'ours_5000':continuation/'step-5000/audio-validation'/f'{name}.wav'}
        for label,path in paths.items():
            samples,sr=sf.read(path)
            assert sr==48000 and len(samples)==len(x)
            outputs[label]=samples[begin:end]
        scores={label:metrics(p.astype('float64'),x[begin:end,1].astype('float64')) for label,p in outputs.items()}
        for label,samples in dict(outputs,reference_wet=x[begin:end,1],reference_dry=x[begin:end,0]).items():
            folder=root/'audio'/label;folder.mkdir(parents=True,exist_ok=True)
            sf.write(folder/f'{name}.wav',samples,48000,subtype='FLOAT')
        results.append(dict(id=name,conditioning=condition.tolist(),start_sample=begin,stop_sample=end,
            samples_scored=end-begin,metrics=scores,author_gpu_batch_inference_seconds=seconds))
        print('SCORED',name,{k:round(v['esr'],6) for k,v in scores.items()},flush=True)
    means={model:{metric:float(np.mean([r['metrics'][model][metric] for r in results])) for metric in ('mse','mae','esr')}
           for model in results[0]['metrics']}
    report=dict(source_repository='https://github.com/RiccardoVib/CONDITIONED-MODELING-OF-OPTICAL-COMPRESSOR',
        source_commit='3146ca7bd9a5df09e220d4784911d4db656ad775',weights=str(weights),
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),tensorflow_parity=parity,
        device=torch.cuda.get_device_name(),conditioning=selected,records=results,macro_mean=means,
        our_selected_best_step=best_step,protocol='Same five held-out knob settings, original 48kHz amplitude, common samples [16,95984). No fitting of output gain or time shift.',
        limitations=['Conditioning order/direction inferred using our training split; not supplied by original preprocessing.',
            'Author training data may include these recordings; this is not an independent-source generalization test.',
            'Our model trained only 20 settings and two-second prefixes; training budgets differ.',
            'Author inference uses non-overlapping 16-sample output blocks and 16-sample lookback; no artificial prefix padding or edge taper.',
            'GPU batch inference timing is not a measured real-time plugin latency.'])
    (root/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# CL-1B 預訓練模型比較','',
        '相同 5 組保留設定、48 kHz、原始振幅；共同評分區間為前 2 秒中的 sample 16 至 95,983，共 95,968 samples。','',
        '| 模型 | 平均 MSE ↓ | 平均 MAE ↓ | 平均 ESR ↓ |','|---|---:|---:|---:|']
    for label,v in means.items():lines.append(f'| {label} | {v["mse"]:.8g} | {v["mae"]:.8g} | {v["esr"]:.8g} |')
    lines.extend(['','## 方法與限制','',
        '作者模型：CL1BModel/Checkpoints/best/weights.h5，原生 TensorFlow 載入後產生參考輸出；等價 PyTorch CUDA 推論通過數值比對。',
        f'128 個隨機輸入區塊的最大絕對差：{parity["max_absolute_error"]:.3g}。',
        f'四個條件輸入順序：{selected["order"]}；threshold 方向反轉：{selected["threshold_flipped"]}。',
        '上游未提供具名條件欄位的資料前處理，因此以本方 20 組 train 錄音比較 48 種排列／threshold 方向。5 組保留設定未參與選擇；細節見 conditioning-calibration.json。',
        '作者模型可能已看過這批音訊；兩邊訓練資料量與預算不同。這是相同輸入下的實測比較，不是獨立測試集的泛化能力排名。',
        '我方分別列出 2,000、3,000、4,000、5,000 步參數經同一數值求解器的結果；pure_algorithm 為未訓練的預設參數。',
        '未調整輸出增益、未以目標搜尋延遲、未重新訓練作者權重。試聽 audio/ 中各版本時請保持同一播放音量。','',
        '## 逐檔 ESR','', '| 錄音 | 作者預訓練 | 我方 3,000 步 | 我方 5,000 步 |','|---|---:|---:|---:|'])
    for r in results:
        m=r['metrics'];lines.append(f'| {r["id"]} | {m["author_pretrained"]["esr"]:.6g} | {m["ours_3000"]["esr"]:.6g} | {m["ours_5000"]["esr"]:.6g} |')
    lines.extend(['','來源：'+report['source_repository'],
        '論文：https://www.dafx.de/paper-archive/2023/DAFx23_paper_10.pdf',
        '作者：Riccardo Simionato、Stefano Fasciani（DAFx 2023）。下載的程式與模型保留原授權檔。'])
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE',means,flush=True)


if __name__=='__main__':
    main()
