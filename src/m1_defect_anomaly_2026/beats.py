import numpy as np
from .preprocessing import FS, preprocess
from .splits import AAMI_MAP, BEAT_SYMBOLS

# window around R-peak

PRE = int(0.25*FS)
POST = int(0.40*FS)

class RecordData:

    def __init__(self, name, signal, r_peaks, labels, y):
        self.name=name
        self.signal=signal
        self.r_peaks=r_peaks
        self.labels=labels
        self.y=y


    @property
    def n_beats(self):
        return len(self.r_peaks)


    @property
    def anomaly_rate(self):
        return float(self.y.mean()) if self.n_beats else 0.0


