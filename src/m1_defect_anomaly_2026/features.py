import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


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



