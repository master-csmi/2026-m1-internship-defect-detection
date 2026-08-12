import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold


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





