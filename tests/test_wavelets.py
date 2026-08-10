import numpy as np
from m1_defect_anomaly_2026.wavelets import dwt_features, beat_dwt_matrix, cwt_features, beat_cwt_matrix


def test_energy_ratios_sum_to_one():
    beat = np.random.default_rng(0).normal(size=234)
    features = dwt_features(beat)
    total = sum(v for k, v in features.items() if k.endswith("_energy"))
    assert np.isclose(total, 1.0, atol=1e-6)


def test_flat_beat_does_not_produce_nan():
    features = dwt_features(np.zeros(234))
    assert all(np.isfinite(v) for v in features.values())


def test_dwt_matrix_has_one_row_per_beat():
    beats = np.random.default_rng(1).normal(size=(6, 234))
    df = beat_dwt_matrix(beats)
    assert df.shape[0] == 6
    assert not df.isna().any().any()


# a slow 2 Hz background, plus a short 40 Hz burst placed somewhere in the window
def make_burst(start, n=234, fs=360, length=20, freq=40.0):
    t = np.arange(n) / fs
    signal = 0.1 * np.sin(2 * np.pi * 2.0 * t)
    burst_t = np.arange(length) / fs
    signal[start:start + length] += np.sin(2 * np.pi * freq * burst_t)
    return signal


def test_cwt_finds_when_the_burst_happens():
    early = cwt_features(make_burst(10))["cwt_30_50_time"]
    late = cwt_features(make_burst(200))["cwt_30_50_time"]
    assert early < 0.4
    assert late > 0.6


def test_cwt_matrix_has_one_row_per_beat():
    beats = np.random.default_rng(3).normal(size=(5, 234))
    df = beat_cwt_matrix(beats)
    assert df.shape[0] == 5
    assert not df.isna().any().any()