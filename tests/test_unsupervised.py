import numpy as np
from m1_defect_anomaly_2026.beats import RecordData
from m1_defect_anomaly_2026.unsupervised import build_feature_matrix, build_split_matrix, feature_columns, fit_isolation_forest, anomaly_score


# a fake record: a regular rhythm of one beat per second, where the beats listed
# in wide_beats are wider than the others, like a ventricular beat would be
def make_record(name="fake", seed=0, wide_beats=(), n_beats=60, fs=360):
    rng = np.random.default_rng(seed)
    signal = rng.normal(scale=0.05, size=fs * (n_beats + 10))
    peaks = []
    for i in range(n_beats):
        peak = 2 * fs + i * fs
        t = np.arange(-30, 30)
        width = 200.0 if i in wide_beats else 40.0
        signal[peak - 30:peak + 30] += 3 * np.exp(-(t ** 2) / width)
        peaks.append(peak)

    labels = np.array(["N"] * n_beats)
    labels[list(wide_beats)] = "V"
    y = np.array([0 if label == "N" else 1 for label in labels])
    return RecordData(name, signal, np.array(peaks), labels, y)


def test_feature_matrix_has_one_row_per_beat():
    rec = make_record(n_beats=40)
    df = build_feature_matrix(rec)
    assert len(df) == rec.n_beats


def test_feature_matrix_has_no_missing_values():
    df = build_feature_matrix(make_record())
    assert not df.isna().any().any()
    assert np.isfinite(df.to_numpy()).all()


def test_split_matrix_stacks_records_and_keeps_labels():
    records = [make_record("a", 0, {5}), make_record("b", 1, {9})]
    df = build_split_matrix(records)
    assert len(df) == sum(r.n_beats for r in records)
    assert set(df["record"]) == {"a", "b"}
    assert df["y"].sum() == 2


def test_feature_columns_drops_the_meta_columns():
    df = build_split_matrix([make_record()])
    columns = feature_columns(df)
    assert "record" not in columns and "label" not in columns and "y" not in columns
    assert len(columns) == df.shape[1] - 3




def test_isolation_forest_gives_outliers_a_higher_score():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(300, 4))
    outliers = rng.normal(loc=8.0, size=(10, 4))

    model = fit_isolation_forest(normal)
    normal_scores = anomaly_score(model, normal)
    outlier_scores = anomaly_score(model, outliers)

    assert outlier_scores.mean() > normal_scores.mean()


def test_anomaly_score_returns_one_value_per_row():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    scores = anomaly_score(fit_isolation_forest(X), X)
    assert scores.shape == (50,)
    assert np.isfinite(scores).all()


def test_isolation_forest_is_reproducible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    first = anomaly_score(fit_isolation_forest(X, seed=42), X)
    second = anomaly_score(fit_isolation_forest(X, seed=42), X)
    assert np.allclose(first, second)