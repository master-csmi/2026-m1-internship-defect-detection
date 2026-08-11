import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM

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



# the SVM cost grows quadratically with the number of beats and DS1 has around 50 000,
# so we fit on a random sample of them instead of the whole split
def subsample(X, max_samples, seed=42):
    if max_samples is None or len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=max_samples, replace=False)
    return X.iloc[rows] if hasattr(X, "iloc") else X[rows]


# One-Class SVM draws a boundary around the region where the data lives
# and flags everything that falls outside of it
# unlike the trees, it really needs the scaling: the RBF kernel measures distances,
# so without it a feature like "energy" would cover all the others
# nu is roughly the share of the training beats we accept to leave outside the boundary
def fit_one_class_svm(X, nu=0.1, gamma="scale", max_samples=20000, seed=42):
    model = make_pipeline(StandardScaler(),
                          OneClassSVM(kernel="rbf", nu=nu, gamma=gamma))
    model.fit(subsample(X, max_samples, seed))
    return model




