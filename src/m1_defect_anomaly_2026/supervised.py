import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from xgboost import XGBClassifier
import shap
from .unsupervised import subsample
from sklearn.base import clone


# small grid: enough to see whether tuning actually helps, without taking all day to run
RF_PARAM_GRID = {"n_estimators": [200, 500],
                 "max_depth": [None, 12],
                 "min_samples_leaf": [1, 4],
                 "class_weight": [None, "balanced"]}




# GroupKFold on "record" keeps every beat of one patient on the same side of a fold,
# so the model can't partly learn a patient's own rhythm instead of the arrhythmia -
# same idea as the DS1/DS2 split
def fit_random_forest(X, y, groups, param_grid=RF_PARAM_GRID, cv_splits=5, seed=42):
    model = RandomForestClassifier(random_state=seed)
    cv = GroupKFold(n_splits=cv_splits)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1")
    grid.fit(X, y, groups=groups)
    return grid




# P(anomaly), same "high score = anomalous" idea as anomaly_score() for the unsupervised models
def predict_scores(grid, X):
    return grid.predict_proba(X)[:, 1]


# scale_pos_weight depends on how imbalanced y is, so it's built from y instead of a fixed number
def fit_xgboost(X, y, groups, param_grid=None, cv_splits=5, seed=42):
    if param_grid is None:
        ratio = (y == 0).sum() / (y == 1).sum()
        param_grid = {
            "n_estimators": [200, 500],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "scale_pos_weight": [1, ratio],
        }
    model = XGBClassifier(random_state=seed, eval_metric="logloss")
    cv = GroupKFold(n_splits=cv_splits)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1")
    grid.fit(X, y, groups=groups)
    return grid




# P(anomaly), same "high score = anomalous" idea as anomaly_score() for the unsupervised models
# works for both fit_random_forest() and fit_xgboost() results
def predict_scores(grid, X):
    return grid.predict_proba(X)[:, 1]




# SHAP says how much each feature pushed one beat towards "anomaly" or away from it
# we explain a sample of the beats: 50 000 would be slow and the picture does not change
def shap_values(model, X, sample_size=1000, seed=42):
    X_sample = subsample(X, sample_size, seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)

    # Random Forest returns one array per class, XGBoost returns a single one
    # in both cases we keep the anomaly class
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    return X_sample, values


# average over all the beats: which features matter the most in general
def shap_importance(values, feature_names):
    importance = np.abs(values).mean(axis=0)
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)




# the threshold is picked on DS1, but a model that already saw a beat while training is too
# sure about it, so we score each beat with the folds that did not train on it
def oof_scores(grid, X, y, groups, cv_splits=5):
    model = clone(grid.best_estimator_)
    cv = GroupKFold(n_splits=cv_splits)
    proba = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")
    return proba[:, 1]





