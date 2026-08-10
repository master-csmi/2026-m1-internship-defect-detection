import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from .preprocessing import FS


# count how many times the signal crosses zero
# a noisy or fragmented beat crosses much more often than a clean one
def zero_crossings(beat):
    signs = np.sign(beat)
    signs[signs==0]=1
    return int(np.sum(signs[1:]!=signs[:-1]))# count where there is consecutive elements have different signs



#describe one beat window with a small set of statistics
def time_domain_features(beat):
    beat=np.asarray(beat,dtype=float)
    std=float(beat.std())
    # if the window is flat, skew and kurtosis are undefinied
    if std<1e-12:
        beat_skew=0.0
        beat_kurtosis=0.0
    else:
        beat_skew=float(skew(beat))
        beat_kurtosis=float(kurtosis(beat))

    return {"rms": float(np.sqrt(np.mean(beat ** 2))),
            "variance": float(beat.var()),
            "std": std,
            "skewness": beat_skew,
            "kurtosis": beat_kurtosis,
            "peak_to_peak": float(beat.max() - beat.min()),
            "mad": float(np.mean(np.abs(beat - beat.mean()))),
            "energy": float(np.sum(beat ** 2)),
            "zero_crossings": float(zero_crossings(beat)),
            "max_abs": float(np.max(np.abs(beat)))}



# apply time_domain_features to every beat of a record
def beat_time_feature_matrix(beat_matrix):
    return pd.DataFrame([time_domain_features(b) for b in beat_matrix])



# intervals between successive R-peaks, in seconds
def rr_intervals(r_peaks, fs=FS):
    return np.diff(np.asarray(r_peaks, dtype=float)) / fs



# rhythm features: one row per beat, aligned with r_peaks
# local_window = how many surrounding beats define "the patient's current rhythm"
def hrv_features(r_peaks, fs=FS, local_window=10):
    n = len(r_peaks)
    # a record with 0 or 1 beat has no rhythm to describe
    if n < 2:
        return pd.DataFrame(index=range(n), columns=_HRV_COLUMNS, dtype=float).fillna(0.0)

    rr = rr_intervals(r_peaks, fs)
    fallback = float(np.median(rr))

    #the first beat has no interval before it, the last has none after it
    pre_rr = np.concatenate([[fallback], rr])
    post_rr = np.concatenate([rr, [fallback]])

    # successive difference centred on each beat: this is what RMSSD and pNN50 are built from
    succ_diff = post_rr - pre_rr

    pre_series = pd.Series(pre_rr)
    diff_series = pd.Series(succ_diff)
    roll = dict(window=local_window, center=True, min_periods=1)

    local_rr_mean = pre_series.rolling(**roll).mean().to_numpy() # average beat duartion
    local_sdnn = pre_series.rolling(**roll).std().fillna(0.0).to_numpy() # standard deviation
    local_rmssd = np.sqrt((diff_series ** 2).rolling(**roll).mean().to_numpy()) # root mean squared of successive differences
    local_pnn50 = (diff_series.abs() > 0.050).rolling(**roll).mean().to_numpy() # proportion of successive differences greater than 50 milliseconds

    return pd.DataFrame({"pre_rr": pre_rr,
                         "post_rr": post_rr,
                         "rr_ratio": pre_rr / (post_rr + 1e-12),
                         "local_rr_mean": local_rr_mean,
                         "rr_deviation": pre_rr / (local_rr_mean + 1e-12),
                         "local_sdnn": local_sdnn,
                         "local_rmssd": local_rmssd,
                         "local_pnn50": local_pnn50,
                         "heart_rate": 60.0 / (pre_rr + 1e-12)})


_HRV_COLUMNS = ["pre_rr", "post_rr", "rr_ratio", "local_rr_mean", "rr_deviation",
                "local_sdnn", "local_rmssd", "local_pnn50", "heart_rate"]




