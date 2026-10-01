import numpy as np
import pytest

from ht1b.audio_metrics import FFT_SIZES, frame_analysis, mrstft, region_metrics, stft_magnitude


def test_spectral_identity_and_silence_are_finite():
    rng = np.random.default_rng(23)
    for x in (rng.normal(size=4096), np.zeros(4096)):
        assert mrstft(x, x)["total"] == 0
    assert np.isfinite(mrstft(rng.normal(size=4096), np.zeros(4096))["total"])


def test_spectral_gain_error_matches_closed_form_for_broadband_signal():
    x = np.random.default_rng(8).normal(size=8192)
    result = mrstft(2*x, x)
    assert result["total"] == pytest.approx(1 + np.log(2), abs=1e-7)


def test_rms_gain_and_partial_region_match_raw_samples():
    x = np.random.default_rng(42).normal(size=1999) * .1
    p = 2*x
    frames = frame_analysis({"reference_wet": x, "model": p})
    full = region_metrics(frames, "model")
    assert full["rms_db_mae"] == pytest.approx(20*np.log10(2))
    local = region_metrics(frames, "model", 1, 5)
    assert local["samples"] == 1519
    assert local["mse"] == pytest.approx(np.mean((p[480:] - x[480:])**2))
    assert local["mae"] == pytest.approx(np.mean(np.abs(p[480:] - x[480:])))
    assert local["esr"] == pytest.approx(1)


def test_silence_does_not_invent_esr_or_infinite_envelope_error():
    frames = frame_analysis({"reference_wet": np.zeros(500), "model": np.zeros(500)})
    result = region_metrics(frames, "model")
    assert result["esr"] is None
    assert result["rms_db_mae"] == 0
    with pytest.raises(ValueError):
        region_metrics(frames, "model", 1, 1)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        mrstft(np.zeros(8192), np.zeros(8191))
    with pytest.raises(ValueError):
        mrstft(np.zeros(100), np.zeros(100))


def test_numpy_stft_matches_torch_reference():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(73).normal(size=8191)
    for size in FFT_SIZES:
        spectrum = torch.stft(torch.from_numpy(x), n_fft=size, hop_length=size//4,
                              win_length=size, window=torch.hann_window(size, dtype=torch.float64),
                              center=True, pad_mode="reflect", return_complex=True)
        expected = torch.sqrt(torch.clamp(spectrum.real**2 + spectrum.imag**2, min=1e-8))
        np.testing.assert_allclose(stft_magnitude(x, size), expected.numpy().T,
                                   rtol=1e-11, atol=1e-11)
