import numpy as np
import pandas as pd
from .preprocessing import FS

# function that will return us frequencies for one beat
def power_spectrum(beat,fs=FS,use_window=True):
    x= beat -beat.mean()
    # we are gonna use the hann window to cut the sharp edges in a signal
    # bcz when we use the FFt it reads them as energy that is not in the signal
    if use_window:
        x=x*np.hanning(len(x))  

    spectrum=np.fft.rfft(x)
    power = np.abs(spectrum)**2 # Takes the magnitude of each frequency component and squares it
    freqs=np.fft.rfftfreq(len(x),d=1.0/fs)
    return freqs, power




# gives six numbers describing one beat's spectrum
def spectral_features(beat,fs=FS):
    freqs,power=power_spectrum(beat,fs)
    total=power.sum()+1e-12
    p=power/total # we use it to normalise and behaves like a probab distribution

    centroid=float((freqs*p).sum())
    bandwidth= float(np.sqrt((((freqs-centroid)**2)*p).sum()))
    entropy=float(-(p*np.log(p+1e-12)).sum()/np.log(len(p)))
    dominant=float(freqs[np.argmax(power)])
    lf_ratio=float(p[freqs<10].sum())
    hf_ratio=float(p[freqs>20].sum())

    return {"spectral_centroid":centroid,
            "spectral_bandwdth":bandwidth,
            "spectral_entropy":entropy,
            "dominant_freq":dominant,
            "lf_ratio":lf_ratio,
            "hf_ratio":hf_ratio}


# apply spectral_features to every beat
def beat_feature_matrix(beat_matrix,fs=FS):
    return pd.DataFrame([spectral_features(b,fs) for b in beat_matrix])







