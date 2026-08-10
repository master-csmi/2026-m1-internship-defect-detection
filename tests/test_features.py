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



def test_hrv_has_one_row_per_beat():
    peaks = np.arange(0, 3600, 360)
    df = hrv_features(peaks, fs=360)
    assert len(df) == len(peaks)
    assert not df.isna().any().any()


def test_regular_rhythm_gives_flat_hrv():
    peaks = np.arange(0, 7200, 360)
    df = hrv_features(peaks, fs=360)
    assert np.allclose(df["rr_ratio"], 1.0)
    assert np.allclose(df["heart_rate"], 60.0)
    assert np.allclose(df["local_rmssd"], 0.0, atol=1e-9)


def test_premature_beat_lowers_rr_ratio():
    # regular 1 s rhythm, but beat 5 arrives early and is followed by a longer pause
    peaks = [0, 360, 720, 1080, 1440, 1620, 2160, 2520, 2880, 3240]
    df = hrv_features(np.array(peaks), fs=360)
    assert df["rr_ratio"].iloc[5] < 0.6
    assert df["rr_deviation"].iloc[5] < 0.8


def test_single_beat_record_does_not_crash():
    df = hrv_features(np.array([100]), fs=360)
    assert len(df) == 1
    assert np.isfinite(df.to_numpy()).all()



