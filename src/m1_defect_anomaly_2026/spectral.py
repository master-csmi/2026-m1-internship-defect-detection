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







