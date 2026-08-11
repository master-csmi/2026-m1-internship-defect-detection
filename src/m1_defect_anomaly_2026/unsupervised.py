import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .beats import extract_beat_matrix
from .features import beat_time_feature_matrix, hrv_features
from .spectral import beat_feature_matrix
from .wavelets import beat_dwt_matrix

# columns that describe the beat, not the features themselves
META_COLUMNS = ["record", "label", "y"]


# put every family of features side by side: one row per beat
# shape (time domain) + rhythm (HRV) + frequency (FFT) + time-frequency (wavelets)
def build_feature_matrix(rec):
    beats = extract_beat_matrix(rec)
    features = pd.concat([beat_time_feature_matrix(beats),hrv_features(rec.r_peaks),
                          beat_feature_matrix(beats),  beat_dwt_matrix(beats)], axis=1)
    # a division by an almost zero value can leave an inf, the models can't read those
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# same thing for a whole split, keeping track of which record each beat comes from
def build_split_matrix(records):
    frames = []
    for rec in records:
        features = build_feature_matrix(rec)
        features.insert(0, "record", rec.name)
        features["label"] = rec.labels
        features["y"] = rec.y
        frames.append(features)
    return pd.concat(frames, ignore_index=True)


# the feature columns only, without record / label / y
def feature_columns(df):
    return [c for c in df.columns if c not in META_COLUMNS]




# Isolation Forest isolates a point by splitting the feature space at random
# an anomaly is different from the rest so it gets separated after only a few splits
# the StandardScaler is not needed by the trees themselves, but it keeps the
# same preparation for the One-Class SVM later, which does need it
def fit_isolation_forest(X, contamination=0.1, seed=42):
    model = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200,contamination=contamination,
                                          random_state=seed))
    model.fit(X)
    return model


# score_samples gives a high value to normal points, we flip the sign so that
# a high score means anomalous, like every other detector in the project
def anomaly_score(model, X):
    return -model.score_samples(X) # "-" to flip the axis