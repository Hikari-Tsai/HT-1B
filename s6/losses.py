"""Supervised waveform + multi-resolution STFT loss; no invented physics term."""
import torch
from torch import nn
from torch.nn import functional as F


class AudioLoss(nn.Module):
    def __init__(self, fft_sizes=(256, 512, 1024), l1_weight=1., esr_weight=.1,
                 spectral_weight=.1):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.weights = l1_weight, esr_weight, spectral_weight
        if not self.fft_sizes or any(not isinstance(n, int) or n < 4 for n in self.fft_sizes):
            raise ValueError('fft_sizes must contain integers >= 4')

    def forward(self, prediction, target):
        if prediction.shape != target.shape or target.ndim != 2 or target.numel() == 0:
            raise ValueError('Loss expects identical nonempty [batch,time] shapes')
        error = prediction-target
        l1 = error.abs().mean()
        # A fixed floor makes silence finite without removing DC or normalizing channels.
        esr = (error.square().mean(-1)/target.square().mean(-1).clamp_min(1e-8)).mean()
        spectral = prediction.new_zeros(())
        if self.weights[2] > 0:
            for size in self.fft_sizes:
                pad = max(0, size-target.shape[-1])
                window = torch.hann_window(size, device=target.device, dtype=target.dtype)
                def magnitude(signal):
                    transformed = torch.stft(F.pad(signal, (0, pad)), n_fft=size,
                                             hop_length=size//4, window=window,
                                             center=False, return_complex=True)
                    return transformed.abs().clamp_min(1e-7)
                actual, reference = magnitude(prediction), magnitude(target)
                convergence = (torch.linalg.vector_norm(actual-reference, dim=(-2, -1))/
                               torch.linalg.vector_norm(reference, dim=(-2, -1)).clamp_min(1e-5)).mean()
                log_error = (actual.log()-reference.log()).abs().mean()
                spectral = spectral + convergence + log_error
            spectral = spectral/len(self.fft_sizes)
        parts = dict(l1=l1, esr=esr, spectral=spectral)
        total = sum(weight*value for weight, value in zip(self.weights, parts.values()))
        return total, parts
