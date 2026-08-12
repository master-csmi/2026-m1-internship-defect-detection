import numpy as np
from m1_defect_anomaly_2026.supervised import fit_random_forest, fit_xgboost, predict_scores




# 6 fake patients, alternating class, one informative feature - just enough
# signal to check the model actually learns something and doesn't leak across patients
def make_fake_data(n_records=6, beats_per_record=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for r in range(n_records):
        label = r % 2
        beats = rng.normal(size=(beats_per_record, 4))
        beats[:, 0] += 3 * label
        X.append(beats)
        y += [label] * beats_per_record
        groups += [f"rec{r}"] * beats_per_record
    return np.vstack(X), np.array(y), np.array(groups)





def test_random_forest_separates_the_two_classes():
    X, y, groups = make_fake_data()
    grid = fit_random_forest(X, y, groups, cv_splits=3)
    scores = predict_scores(grid, X)
    assert scores[y == 1].mean() > scores[y == 0].mean()




def test_predict_scores_returns_one_value_per_row():
    X, y, groups = make_fake_data()
    grid = fit_random_forest(X, y, groups, cv_splits=3)
    scores = predict_scores(grid, X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()




def test_xgboost_separates_the_two_classes():
    X, y, groups = make_fake_data()
    grid = fit_xgboost(X, y, groups, cv_splits=3)
    scores = predict_scores(grid, X)
    assert scores[y == 1].mean() > scores[y == 0].mean()




def test_xgboost_scale_pos_weight_matches_the_class_imbalance():
    X, y, groups = make_fake_data()
    grid = fit_xgboost(X, y, groups, cv_splits=3)
    ratio = (y == 0).sum() / (y == 1).sum()
    assert ratio in grid.param_grid["scale_pos_weight"]



