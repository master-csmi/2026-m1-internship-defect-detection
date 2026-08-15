import math

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .beats import PRE, POST, extract_beat_matrix
from .evaluation import evaluate_scores
from .preprocessing import FS


# the long window used by the LSTM: 2 seconds around the R peak, so the beat before
# and the beat after are inside the window too
LONG_SECONDS = 2.0
# the bandpass only keeps what is under 10 Hz, so keeping 1 sample out of 6 (60 Hz)
# loses nothing and makes the sequence 6 times shorter for the LSTM
DECIMATE = 6

META_COLUMNS = ["record", "label", "y"]


def set_seed(seed=42):
    np.random.seed(seed) # picking random samples 
    torch.manual_seed(seed)



def beat_windows(rec):
    return extract_beat_matrix(rec).astype(np.float32)

def beat_window_starts(rec, pre=PRE):
    return np.asarray(rec.r_peaks,dtype=int) - pre



# a longer window, centred on the same R peak: it holds the neighbouring beats,
# so a beat that arrives too early can be seen from the window alone
def long_beat_windows(rec, seconds=LONG_SECONDS, decimate=DECIMATE, fs=FS):
    half = int(round(seconds * fs / 2))
    # padding with the edge value lets the first and last beats keep a full window
    padded = np.pad(rec.signal, (half, half), mode="edge")
    peaks = np.asarray(rec.r_peaks, dtype=int)
    offsets = np.arange(0, 2 * half, decimate)
    # index p of the padded signal is index p - half of the original one, so the window
    # [peak - half, peak + half) becomes padded[peak : peak + 2 * half]
    return padded[peaks[:, None] + offsets[None, :]].astype(np.float32)


def long_window_starts(rec, seconds=LONG_SECONDS, fs=FS):
    half = int(round(seconds * fs / 2))
    return np.asarray(rec.r_peaks, dtype=int) - half

# hold a few whole records out of the training set: the threshold is then chosen on
# patients the autoencoder never saw, the same idea as the DS1 / DS2 split
def split_records(records, n_val=5, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    val_positions = set(order[:n_val].tolist())
    train = [rec for i, rec in enumerate(records) if i not in val_positions]
    val = [rec for i, rec in enumerate(records) if i in val_positions]
    return train, val



# stack the windows of several records and keep track of where each one comes from
# normal_only=True keeps the N beats: that is all an autoencoder is trained on
def stack_windows(records, window_fn=beat_windows, normal_only=False):
    windows, meta = [], []
    for rec in records:
        w = window_fn(rec)
        keep = rec.y == 0 if normal_only else np.ones(len(rec.y), dtype=bool)
        windows.append(w[keep])
        meta.append(pd.DataFrame({"record": rec.name,
                                  "label": rec.labels[keep],
                                  "y": rec.y[keep]}))
    X = np.concatenate(windows).astype(np.float32)
    return X, pd.concat(meta, ignore_index=True)







## we start the two models 

# each stride-2 convolution halves the length, rounding up
def _conv_out_length(length, n_layers):
    for _ in range(n_layers):
        length = -(-length // 2)
    return length


# the convolutions read the shape of the beat through small sliding filters, and the
# Linear layer in the middle squeezes the whole window into latent_dim numbers
# that squeeze is the point: with only 16 numbers to describe 234 samples the model has
# to learn what a normal beat looks like instead of copying its input
class Conv1dAutoencoder(nn.Module):
    def __init__(self, input_length, latent_dim=16, channels=(16, 32, 64), kernel_size=7):
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim = int(latent_dim)
        self.channels = tuple(channels)
        pad = kernel_size // 2
        c1, c2, c3 = self.channels

        self.encoder = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size, stride=2, padding=pad), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size, stride=2, padding=pad), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size, stride=2, padding=pad), nn.ReLU())

        self.bottleneck_length = _conv_out_length(self.input_length, len(self.channels))
        flat = c3 * self.bottleneck_length
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)

        # upsample + convolution instead of a transposed convolution: same job, and the
        # length is easier to follow
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c3, c2, kernel_size, padding=pad), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c2, c1, kernel_size, padding=pad), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c1, c1, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv1d(c1, 1, kernel_size, padding=pad))

    def encode(self, x):
        h = self.encoder(x.unsqueeze(1))
        return self.to_latent(h.flatten(1))

    def decode(self, z):
        h = self.from_latent(z).view(z.shape[0], self.channels[-1], self.bottleneck_length)
        out = self.decoder(h)
        # three upsamplings give 8 * bottleneck_length, which is a little more than the
        # input when the length is not a power of two, so we resize back to the exact size
        out = F.interpolate(out, size=self.input_length, mode="linear", align_corners=False)
        return out.squeeze(1)

    def forward(self, x):
        return self.decode(self.encode(x))




# the LSTM walks along the window one time step after the other and keeps a memory of
# what it has seen, so it can use the distance between two beats, not only their shape
#
# two things here are there to make the model trainable at all - the first version of this
# class stayed stuck at a reconstruction error of ~0.95, which on a signal of variance 1 is
# what you get by answering a flat line:
#   - the encoder reads the window in BOTH directions. With a single direction the summary
#     of the window is the memory state after 120 steps, and what happened at the start of
#     the window has to survive all 120 of them.
#   - the decoder is given sines and cosines of the position instead of a single 0 -> 1
#     ramp. A ramp tells it where it is, but it still has to invent every oscillation of
#     the ECG from scratch; with sines it only has to combine them.
class LstmAutoencoder(nn.Module):
    def __init__(self, hidden_size=64, latent_dim=16, num_layers=1,
                 bidirectional=True, clock_harmonics=8):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.clock_harmonics = int(clock_harmonics)

        self.encoder = nn.LSTM(1, self.hidden_size, num_layers=self.num_layers,
                               batch_first=True, bidirectional=self.bidirectional)
        directions = 2 if self.bidirectional else 1
        self.to_latent = nn.Linear(self.hidden_size * directions, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, self.hidden_size)
        self.clock_size = 2 * self.clock_harmonics
        self.decoder = nn.LSTM(self.hidden_size + self.clock_size, self.hidden_size,
                               num_layers=self.num_layers, batch_first=True)
        self.head = nn.Linear(self.hidden_size, 1)

    def encode(self, x):
        _, (hidden, _) = self.encoder(x.unsqueeze(-1))
        # hidden is (layers * directions, batch, hidden_size), the last layer is at the end
        if self.bidirectional:
            summary = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            summary = hidden[-1]
        return self.to_latent(summary)

    # sin(2 pi k t) and cos(2 pi k t) for k = 1 .. clock_harmonics, t going from 0 to 1
    # across the window: a fixed set of waves the decoder can add up
    def _clock(self, length, device, dtype):
        t = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype).view(length, 1)
        k = torch.arange(1, self.clock_harmonics + 1, device=device, dtype=dtype).view(1, -1)
        angle = (2.0 * math.pi) * k * t
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1).unsqueeze(0)

    def decode(self, z, length):
        seed = self.from_latent(z).unsqueeze(1).expand(-1, length, -1)
        clock = self._clock(length, z.device, z.dtype).expand(z.shape[0], -1, -1)
        out, _ = self.decoder(torch.cat([seed, clock], dim=-1))
        return self.head(out).squeeze(-1)

    def forward(self, x):
        return self.decode(self.encode(x), x.shape[1])




## training

def _to_tensor(X):
    return torch.as_tensor(np.asarray(X, dtype=np.float32))



# the target is the input itself: the model is only asked to give back what it was given
# the validation loss is measured on normal beats of held-out records, and the weights of
# the best epoch are the ones we keep
def train_autoencoder(model, X_train, X_val=None, epochs=30, batch_size=256, lr=1e-3,
                      patience=5, seed=42, verbose=True, grad_clip=1.0):
    set_seed(seed)
    loader = DataLoader(TensorDataset(_to_tensor(X_train)), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_loss, best_state, waited = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for (batch,) in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            # an LSTM can produce one huge gradient and destroy the weights in a single
            # step - this caps the size of the step, and costs nothing for the CNN
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        train_loss = total / max(seen, 1)
        history["train"].append(train_loss)

        watched = train_loss
        if X_val is not None:
            val_loss = float(reconstruction_error(model, X_val, batch_size=512).mean())
            history["val"].append(val_loss)
            watched = val_loss

        if verbose:
            message = f"epoch {epoch + 1:3d}/{epochs}  train {train_loss:.5f}"
            if X_val is not None:
                message += f"  val {history['val'][-1]:.5f}"
            print(message)

        if watched < best_loss - 1e-6:
            best_loss = watched
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                if verbose:
                    print(f"stopped early at epoch {epoch + 1}, best loss {best_loss:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# the error of the laziest possible model: always answer the average training window
# any autoencoder must beat this, otherwise its scores mean nothing
def mean_window_baseline(X_train, X_eval=None):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_eval = X_train if X_eval is None else np.asarray(X_eval, dtype=np.float32)
    return float(((X_eval - X_train.mean(axis=0)) ** 2).mean())


@torch.no_grad()
def reconstruct(model, X, batch_size=512):
    model.eval()
    X = _to_tensor(X)
    if len(X) == 0:
        return np.empty((0, X.shape[1] if X.ndim > 1 else 0), dtype=np.float32)
    pieces = [model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    return torch.cat(pieces).cpu().numpy()


# how wrong the model was at every single sample of the window, used for the curve
# drawn on top of the raw signal
def squared_error(model, X, batch_size=512):
    X = np.asarray(X, dtype=np.float32)
    return (X - reconstruct(model, X, batch_size)) ** 2


# one number per beat: the anomaly score
# the model has only ever seen normal windows, so a window it cannot rebuild is a window
# that does not look like anything it learned
def reconstruction_error(model, X, batch_size=512):
    return squared_error(model, X, batch_size).mean(axis=1)





## choose the threshold 


# the decision boundary: we accept that a small share of the normal validation beats is
# flagged, and everything above that error is called an anomaly
# no anomaly label is needed here, only normal beats
def percentile_threshold(errors, percentile=99.0):
    return float(np.percentile(np.asarray(errors, dtype=float), percentile))


# try several percentiles and see what each one costs on the test set
def threshold_sweep(val_errors, y_test, test_scores,
                    percentiles=(90.0, 95.0, 97.5, 99.0, 99.5)):
    rows = []
    for q in percentiles:
        threshold = percentile_threshold(val_errors, q)
        result = evaluate_scores(y_test, test_scores, threshold, method=f"percentile {q}")
        rows.append({"percentile": q, "threshold": threshold,
                     "precision": result["precision"], "recall": result["recall"],
                     "f1": result["f1"], "fp": result["fp"], "fn": result["fn"]})
    return pd.DataFrame(rows)







## plotting helper

# put the per-sample error back where it came from in the recording, so it can be drawn
# under the raw signal
# windows overlap, so a sample covered twice gets the average of the two errors
# step is the decimation factor: one time step of a decimated window covers step samples
def error_curve(signal_length, starts, per_sample_error, step=1):
    per_sample_error = np.asarray(per_sample_error, dtype=float)
    starts = np.asarray(starts, dtype=int)
    width = per_sample_error.shape[1]

    index = starts[:, None] + (np.arange(width) * step)[None, :]
    values = per_sample_error
    if step > 1:
        index = np.repeat(index, step, axis=1) + np.tile(np.arange(step), width)[None, :]
        values = np.repeat(values, step, axis=1)

    index = index.ravel()
    values = values.ravel()
    inside = (index >= 0) & (index < signal_length)

    total = np.bincount(index[inside], weights=values[inside], minlength=signal_length)
    count = np.bincount(index[inside], minlength=signal_length)
    return np.divide(total, count, out=np.full(signal_length, np.nan, dtype=float),
                     where=count > 0)







