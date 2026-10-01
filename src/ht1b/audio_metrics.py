"""Explicit, reproducible spectral and envelope metrics for paired audio.

This protocol is not a reproduction of either paper's numerical benchmark.
MR-STFT uses spectral convergence plus log-magnitude L1, as in auraloss,
with the explicit resolutions below. No phase/gain alignment is applied.
"""
from __future__ import annotations

import numpy as np

FFT_SIZES = (512, 1024, 2048)
POWER_FLOOR = 1e-8
RMS_FLOOR_DB = -100.0
FRAME_SAMPLES = 480  # 10 ms at the benchmark's 48 kHz.

METRIC_PROTOCOL = {
    "version": "ht1b-audio-metrics-v1",
    "sample_rate": 48000,
    "mrstft": {
        "fft_sizes": list(FFT_SIZES),
        "hop_sizes": [n // 4 for n in FFT_SIZES],
        "window_lengths": list(FFT_SIZES),
        "window": "periodic Hann", "center": True, "padding": "reflect",
        "fft_normalization": "none", "power_floor": POWER_FLOOR,
        "formula": "mean_resolution(||P-T||_F / ||T||_F + mean(abs(ln(P)-ln(T))))",
        "magnitude": "sqrt(max(real(STFT)^2+imag(STFT)^2, power_floor))",
        "gain_normalization": False,
        "reference": "https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py",
        "note": "Explicit local protocol; not numerically comparable to paper tables without matching their configuration and data.",
    },
    "rms": {"window_samples": FRAME_SAMPLES, "hop_samples": FRAME_SAMPLES,
            "floor_dbfs": RMS_FLOOR_DB, "final_frame": "partial, no padding",
            "aggregation": "sample-count-weighted absolute dB difference"},
    "regions": {"selection": "manual, never inferred from analog threshold",
                "grid_seconds": .01, "interval": "start inclusive, stop exclusive",
                "note": "Annotated-window errors, not measured hardware attack/release time constants."},
}


def _signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite, nonempty mono audio")
    return x


def stft_magnitude(x: np.ndarray, size: int) -> np.ndarray:
    x = _signal(x)
    if len(x) <= size // 2:
        raise ValueError("Audio too short for centered reflect-padding STFT")
    padded = np.pad(x, (size // 2, size // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::size // 4]
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(size) / size)
    spectrum = np.fft.rfft(frames * window, axis=-1)
    return np.sqrt(np.maximum(spectrum.real**2 + spectrum.imag**2, POWER_FLOOR))


def mrstft(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction, target = _signal(prediction), _signal(target)
    if prediction.shape != target.shape:
        raise ValueError("Paired signals must have identical lengths")
    resolutions = []
    for size in FFT_SIZES:
        p, t = stft_magnitude(prediction, size), stft_magnitude(target, size)
        convergence = float(np.linalg.norm(p - t) / np.linalg.norm(t))
        log_magnitude = float(np.mean(np.abs(np.log(p) - np.log(t))))
        resolutions.append({"fft_size": size, "spectral_convergence": convergence,
                            "log_magnitude_l1": log_magnitude,
                            "total": convergence + log_magnitude})
    return {"total": float(np.mean([r["total"] for r in resolutions])),
            "resolutions": resolutions}


def frame_analysis(signals: dict[str, np.ndarray], target_key: str = "reference_wet") -> dict:
    """Store sufficient per-frame statistics for exact aligned-region time metrics."""
    signals = {key: _signal(value) for key, value in signals.items()}
    target = signals[target_key]
    if any(x.shape != target.shape for x in signals.values()):
        raise ValueError("Paired signals must have identical lengths")
    starts = np.arange(0, len(target), FRAME_SAMPLES)
    counts = np.minimum(FRAME_SAMPLES, len(target) - starts)
    def sums(x: np.ndarray) -> np.ndarray:
        return np.add.reduceat(x, starts)
    rms = {key: np.maximum(10 * np.log10(np.maximum(sums(x*x) / counts, 1e-10)),
                           RMS_FLOOR_DB).tolist() for key, x in signals.items()}
    stats = {}
    for key, x in signals.items():
        if key.startswith("reference_"):
            continue
        error = x - target
        stats[key] = {"squared_error": sums(error*error).tolist(),
                      "absolute_error": sums(np.abs(error)).tolist()}
    return {"edges_seconds": [*(starts / 48000).tolist(), len(target) / 48000],
            "counts": counts.tolist(), "target_energy": sums(target*target).tolist(),
            "rms_db": rms, "errors": stats}


def region_metrics(frames: dict, model: str, start: int = 0, stop: int | None = None) -> dict:
    stop = len(frames["counts"]) if stop is None else stop
    if not 0 <= start < stop <= len(frames["counts"]):
        raise ValueError("Region must cover at least one complete analysis frame")
    sl = slice(start, stop)
    counts = np.asarray(frames["counts"])[sl]
    samples = int(counts.sum())
    squared_error = float(np.sum(frames["errors"][model]["squared_error"][sl]))
    target_energy = float(np.sum(frames["target_energy"][sl]))
    db_error = np.abs(np.asarray(frames["rms_db"][model])[sl]
                      - np.asarray(frames["rms_db"]["reference_wet"])[sl])
    return {"mse": squared_error / samples,
            "mae": float(np.sum(frames["errors"][model]["absolute_error"][sl])) / samples,
            "esr": squared_error / target_energy if target_energy > 1e-20 else None,
            "rms_db_mae": float(np.dot(db_error, counts) / samples), "samples": samples}
