import numpy as np
import pandas as pd
import pywt # library that handles wavelets transforms
from .preprocessing import FS



WAVELET = "db4"
LEVEL = 4


# cut one beat into frequency bands
# cA4 is the slow shape of the beat, cD1 is the fastest detail
# "periodization" keeps the energy conserved, so the ratios below add up to 1
def dwt_features(beat, wavelet=WAVELET, level=LEVEL):
    coeffs = pywt.wavedec(beat, wavelet, mode="periodization", level=level)
    names = [f"cA{level}"] + [f"cD{level - i}" for i in range(level)]

    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = energies.sum() + 1e-12

    features = {}
    for name, c, energy in zip(names, coeffs, energies):
        # a wide abnormal beat puts more energy in the slow bands, a sharp or noisy one in the fast bands
        features[f"{name}_energy"] = float(energy / total)
        features[f"{name}_std"] = float(c.std())
    return features



# apply dwt_features to every beat of a record
def beat_dwt_matrix(beat_matrix):
    return pd.DataFrame([dwt_features(b) for b in beat_matrix])




BANDS = [(0, 5), (5, 15), (15, 30), (30, 50)]  # in Hz

# the CWT keeps the time axis: we know not only which frequencies are in the beat
# but when they happen inside it, the FFT in spectral.py loses this
def cwt_scalogram(beat, fs=FS, wavelet="morl"):
    scales = np.arange(1, 64)
    coeffs, freqs = pywt.cwt(beat, scales, wavelet, sampling_period=1.0 / fs)
    return freqs, coeffs ** 2  # the square is the energy at each frequency and each moment



# two numbers per band: how much energy it holds, and when that energy peaks
def cwt_features(beat, fs=FS):
    freqs, power = cwt_scalogram(beat, fs)
    total = power.sum() + 1e-12

    features = {}
    for low, high in BANDS:
        band = power[(freqs >= low) & (freqs < high)]
        energy_in_time = band.sum(axis=0)  # collapse the frequencies, keep the time
        features[f"cwt_{low}_{high}_energy"] = float(band.sum() / total)
        # 0 = start of the beat window, 1 = end of it
        features[f"cwt_{low}_{high}_time"] = float(np.argmax(energy_in_time) / (len(beat) - 1))
    return features


def beat_cwt_matrix(beat_matrix, fs=FS):
    return pd.DataFrame([cwt_features(b, fs) for b in beat_matrix])





