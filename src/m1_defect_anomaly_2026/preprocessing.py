import numpy as np
from scipy.signal import butter, filtfilt, medfilt

FS = 360 # sampling rate in Hz

#keeps only 0.5 to 40 Hz (where the heartbeat lives)
def bandpass(signal, fs=FS, low=0.5,high=10.0,order=3):
    half_fs = fs/2
    b, a =butter(order,[low/half_fs, high/half_fs],btype="band")
    return filtfilt(b,a,signal)



# another method is by drifting with median filters, subtract it
def remove_baseline_median(signal,fs=FS):
    w1=int(0.2*fs)
    w2=int(0.6*fs)
    #ensure window size are odd numbers
    w1+=1 if w1%2==0 else 0
    w2+=1 if w2%2==0 else 0

    #we remove tall spikes
    step1=medfilt(signal, kernel_size=w1)
    #remove wider waves
    baseline=medfilt(step1,kernel_size=w2)
    return signal -baseline

# rescale our data
def normalize(signal):
    std=signal.std()
    if std<1e-12:# this if to avoid dividing by zero
        return signal - signal.mean()
    return(signal - signal.mean())/std

def preprocess(signal, fs=FS, method="bandpass"):
    if method == "bandpass":
        clean = bandpass(signal,fs)
    elif method =="median":
        clean=remove_baseline_median(signal,fs)
    elif method =="none":
        clean=signal
    else:
        raise ValueError(f"Unknown method {method}")

    return normalize(clean)



