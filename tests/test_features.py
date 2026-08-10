import numpy as np
from m1_defect_anomaly_2026.features import time_domain_features, beat_time_feature_matrix, zero_crossings, rr_intervals, hrv_features



def test_rms_of_constant_signal():
    beat = np.full(100, 2.0)
    features = time_domain_features(beat)
    assert np.isclose(features["rms"], 2.0)
    assert np.isclose(features["variance"], 0.0)



def test_flat_window_does_not_produce_nan():
    features = time_domain_features(np.zeros(50))
    assert all(np.isfinite(v) for v in features.values())



def test_symmetric_signal_has_zero_skewness():
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    features = time_domain_features(np.sin(t))
    assert abs(features["skewness"]) < 1e-6



def test_gaussian_has_near_zero_excess_kurtosis():
    rng = np.random.default_rng(0)
    features = time_domain_features(rng.normal(size=20000))
    assert abs(features["kurtosis"]) < 0.2



def test_zero_crossings_counts_sign_changes():
    assert zero_crossings(np.array([1.0, -1.0, 1.0, -1.0])) == 3




def test_matrix_has_one_row_per_beat():
    rng = np.random.default_rng(1)
    beats = rng.normal(size=(7, 234))
    df = beat_time_feature_matrix(beats)
    assert df.shape[0] == 7
    assert not df.isna().any().any()





def test_rr_intervals_of_regular_rhythm():
    peaks = np.arange(0, 3600, 360)          
    assert np.allclose(rr_intervals(peaks, fs=360), 1.0) # we use allclose() instead of "==" because it's not fragile



