import numpy as np
import pytest

# the deep learning part is optional: without torch the whole file is skipped
pytest.importorskip("torch")

from m1_defect_anomaly_2026.beats import RecordData, PRE
from m1_defect_anomaly_2026.autoencoders import (Conv1dAutoencoder, LstmAutoencoder,
                                                 beat_windows, beat_window_starts,
                                                 long_beat_windows, long_window_starts,
                                                 split_records, stack_windows,
                                                 train_autoencoder, reconstruct,
                                                 squared_error, reconstruction_error,
                                                 percentile_threshold, threshold_sweep,
                                                 error_curve, set_seed,
                                                 mean_window_baseline)
import torch


# the same fake record as the other test files: one beat per second, and the beats listed
# in wide_beats are wider, like a ventricular beat
def make_record(name="fake", seed=0, wide_beats=(), n_beats=40, fs=360):
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


# ------------------------------------------------------------------ windows


def test_beat_windows_have_one_row_per_beat():
    rec = make_record()
    windows = beat_windows(rec)
    assert windows.shape[0] == rec.n_beats
    assert windows.dtype == np.float32


def test_long_windows_are_centred_on_the_r_peak():
    rec = make_record()
    windows = long_beat_windows(rec, seconds=2.0, decimate=6)
    assert windows.shape == (rec.n_beats, 120)
    # the time step that falls on the middle of the window is the R peak itself
    middle = windows[:, 360 // 6]
    assert np.allclose(middle, rec.signal[rec.r_peaks], atol=1e-5)


def test_long_windows_survive_a_beat_at_the_very_start():
    rec = make_record()
    rec.r_peaks = np.array([5, 1000])
    windows = long_beat_windows(rec)
    assert windows.shape == (2, 120)
    assert np.isfinite(windows).all()


def test_window_starts_match_the_windows():
    rec = make_record()
    assert (beat_window_starts(rec) == rec.r_peaks - PRE).all()
    assert (long_window_starts(rec, seconds=2.0) == rec.r_peaks - 360).all()


def test_split_records_is_disjoint_and_reproducible():
    records = [make_record(str(i)) for i in range(22)]
    train, val = split_records(records, n_val=5)
    assert len(train) == 17 and len(val) == 5
    assert not {r.name for r in train} & {r.name for r in val}
    _, val_again = split_records(records, n_val=5)
    assert [r.name for r in val] == [r.name for r in val_again]


def test_stack_windows_keeps_only_normal_beats_when_asked():
    records = [make_record("a", 0, {2}), make_record("b", 1, {4, 5})]
    X, meta = stack_windows(records, normal_only=True)
    assert len(X) == len(meta) == (40 - 1) + (40 - 2)
    assert (meta["y"] == 0).all()
    assert set(meta["record"]) == {"a", "b"}


def test_stack_windows_keeps_everything_by_default():
    X, meta = stack_windows([make_record("a", 0, {2})])
    assert len(X) == 40
    assert meta["y"].sum() == 1


# ------------------------------------------------------------------ models


@pytest.mark.parametrize("length", [234, 120, 200, 99])
def test_conv_autoencoder_gives_back_the_same_length(length):
    model = Conv1dAutoencoder(input_length=length, latent_dim=8)
    x = torch.randn(4, length)
    assert model(x).shape == x.shape


def test_conv_autoencoder_really_has_a_bottleneck():
    model = Conv1dAutoencoder(input_length=234, latent_dim=16)
    z = model.encode(torch.randn(4, 234))
    # 16 numbers for 234 samples: the model cannot just copy its input
    assert z.shape == (4, 16)
    assert z.shape[1] < 234


@pytest.mark.parametrize("length", [120, 60])
def test_lstm_autoencoder_gives_back_the_same_length(length):
    model = LstmAutoencoder(hidden_size=8, latent_dim=4)
    x = torch.randn(4, length)
    assert model(x).shape == x.shape


def test_lstm_autoencoder_has_a_bottleneck():
    model = LstmAutoencoder(hidden_size=16, latent_dim=4)
    assert model.encode(torch.randn(3, 120)).shape == (3, 4)


def test_lstm_autoencoder_works_in_one_direction_too():
    model = LstmAutoencoder(hidden_size=8, latent_dim=4, bidirectional=False)
    x = torch.randn(4, 60)
    assert model.encode(x).shape == (4, 4)
    assert model(x).shape == x.shape


def test_lstm_decoder_does_not_give_a_flat_line():
    # the decoder receives the same latent at every time step, so without its position
    # signal the only thing it can answer is one constant repeated - which is exactly how
    # the first version of this model failed
    set_seed(0)
    model = LstmAutoencoder(hidden_size=16, latent_dim=4)
    out = model(torch.randn(2, 60)).detach().numpy()
    assert out.std(axis=1).min() > 1e-6


# ------------------------------------------------------------------ training and scoring


def test_training_lowers_the_loss():
    set_seed(0)
    X = beat_windows(make_record(n_beats=60))
    model = Conv1dAutoencoder(input_length=X.shape[1], latent_dim=8)
    history = train_autoencoder(model, X, epochs=8, batch_size=16, verbose=False)
    assert history["train"][-1] < history["train"][0]


# ---- regression test ----
# the LSTM autoencoder used to train down to a reconstruction error of ~0.95 on a signal of
# variance 1, which is what answering a flat line gives. It looked like a loss, it was the
# model learning nothing, and every score built on it was noise. Here it has to clearly
# beat the flat answer on an easy signal.
def _sine_windows(n=512, length=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2 * np.pi, length)
    phase = rng.uniform(0.0, 2 * np.pi, size=n)[:, None]
    amplitude = rng.uniform(0.5, 1.5, size=n)[:, None]
    return (amplitude * np.sin(t[None, :] + phase)).astype(np.float32)


def test_lstm_autoencoder_beats_the_flat_answer():
    set_seed(0)
    X = _sine_windows()
    model = LstmAutoencoder(hidden_size=32, latent_dim=4)
    train_autoencoder(model, X, epochs=150, batch_size=64, lr=3e-3, verbose=False)

    baseline = mean_window_baseline(X)
    learned = float(reconstruction_error(model, X).mean())
    assert learned < 0.5 * baseline


def test_mean_window_baseline_is_the_error_of_the_average_window():
    X = np.stack([np.zeros(10), np.full(10, 2.0)]).astype(np.float32)
    # the average window is 1 everywhere, so every sample is off by 1
    assert np.isclose(mean_window_baseline(X), 1.0)


def test_training_accepts_a_gradient_clip():
    set_seed(0)
    X = beat_windows(make_record(n_beats=60))
    model = Conv1dAutoencoder(input_length=X.shape[1], latent_dim=8)
    history = train_autoencoder(model, X, epochs=8, batch_size=16, verbose=False,
                                grad_clip=0.5)
    assert np.isfinite(history["train"]).all()
    assert history["train"][-1] < history["train"][0]


def test_reconstruction_error_has_one_score_per_window():
    X = beat_windows(make_record(n_beats=20))
    model = Conv1dAutoencoder(input_length=X.shape[1], latent_dim=8)
    assert reconstruction_error(model, X).shape == (20,)
    assert squared_error(model, X).shape == X.shape
    assert reconstruct(model, X).shape == X.shape


def test_a_model_trained_on_normal_beats_scores_wide_beats_higher():
    set_seed(0)
    train_record = make_record("train", seed=1, n_beats=80)
    test_record = make_record("test", seed=2, wide_beats={3, 8, 15}, n_beats=20)

    X_train = beat_windows(train_record)
    X_test = beat_windows(test_record)

    model = Conv1dAutoencoder(input_length=X_train.shape[1], latent_dim=8)
    train_autoencoder(model, X_train, epochs=40, batch_size=16, verbose=False)

    errors = reconstruction_error(model, X_test)
    anomalous = errors[test_record.y == 1]
    normal = errors[test_record.y == 0]
    assert anomalous.mean() > normal.mean()


def test_the_same_seed_gives_the_same_model():
    X = beat_windows(make_record(n_beats=30))
    scores = []
    for _ in range(2):
        set_seed(7)
        model = Conv1dAutoencoder(input_length=X.shape[1], latent_dim=8)
        train_autoencoder(model, X, epochs=3, batch_size=16, seed=7, verbose=False)
        scores.append(reconstruction_error(model, X))
    assert np.allclose(scores[0], scores[1])


# ------------------------------------------------------------------ threshold


def test_percentile_threshold_leaves_the_expected_share_above():
    errors = np.arange(1000, dtype=float)
    threshold = percentile_threshold(errors, 95.0)
    assert 0.04 < (errors >= threshold).mean() < 0.06


def test_threshold_sweep_returns_one_row_per_percentile():
    rng = np.random.default_rng(0)
    val_errors = rng.random(500)
    y_test = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([rng.random(90) * 0.5, 0.5 + rng.random(10) * 0.5])
    table = threshold_sweep(val_errors, y_test, scores, percentiles=(90.0, 95.0, 99.0))
    assert len(table) == 3
    assert set(["percentile", "threshold", "precision", "recall", "f1"]) <= set(table.columns)
    # a higher percentile is a stricter threshold, so it flags fewer beats
    assert table["threshold"].is_monotonic_increasing


# ------------------------------------------------------------------ error curve


def test_error_curve_covers_every_window_and_averages_the_overlap():
    curve = error_curve(400, np.array([0, 100]),
                        np.stack([np.full(234, 1.0), np.full(234, 3.0)]), step=1)
    assert np.isclose(curve[50], 1.0)     # only the first window
    assert np.isclose(curve[150], 2.0)    # both windows, so the average
    assert np.isclose(curve[300], 3.0)    # only the second window


def test_error_curve_spreads_a_decimated_window_over_the_right_samples():
    values = np.arange(120, dtype=float)[None, :]
    curve = error_curve(2000, np.array([100]), values, step=6)
    assert np.isnan(curve[99]) and np.isnan(curve[820])
    assert np.allclose(curve[100:106], 0.0)
    assert np.allclose(curve[106:112], 1.0)
    assert np.allclose(curve[814:820], 119.0)


def test_error_curve_ignores_what_falls_outside_the_recording():
    curve = error_curve(500, np.array([-50]), np.ones((1, 234)), step=1)
    assert (~np.isnan(curve)).sum() == 234 - 50


def test_error_curve_on_a_record_lines_up_with_the_signal():
    rec = make_record(n_beats=20)
    X = beat_windows(rec)
    model = Conv1dAutoencoder(input_length=X.shape[1], latent_dim=8)
    errors = squared_error(model, X)
    curve = error_curve(len(rec.signal), beat_window_starts(rec), errors)
    assert curve.shape == rec.signal.shape
    assert np.isfinite(curve[rec.r_peaks]).all()