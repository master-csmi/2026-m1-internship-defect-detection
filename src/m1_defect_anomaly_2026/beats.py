import numpy as np
from .preprocessing import FS, preprocess
from .splits import AAMI_MAP, BEAT_SYMBOLS
import wfdb

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




# return only the MLII lead
def pick_lead(record):
    if "MLII" in record.sig_name:
        return record.sig_name.index("MLII")

    return 0


# read one record from the disk and return a record data object
def load_record_data(name,data_dir="data/raw",preprocess_method="bandpass"):
    path=f"{data_dir}/{name}"
    record=wfdb.rdrecord(path) # to get the signal
    ann=wfdb.rdann(path,"atr") # to get the labels

    lead = pick_lead(record)
    raw=record.p_signal[:,lead]
    signal =preprocess(raw,fs=record.fs,method=preprocess_method)
    

    r_peaks,labels=[],[]
    for sample,symbol in zip(ann.sample,ann.symbol):
        if symbol not in BEAT_SYMBOLS: continue
        if sample-PRE<0 or sample+POST>=record.sig_len: #drop edge beats
            continue

        r_peaks.append(sample)
        labels.append(AAMI_MAP[symbol])


    r_peaks = np.asarray(r_peaks,dtype=int)
    labels=np.asarray(labels)
    y=np.array([0 if label=="N" else 1 for label in labels])

    return RecordData(name,signal,r_peaks,labels,y)

"""ran this test and got the results:
python -c "
from m1_defect_anomaly_2026.beats import load_record_data
rec = load_record_data('208')
print('name:', rec.name)
print('signal length:', rec.signal.shape)
print('beats:', rec.n_beats)
print('anomaly rate:', round(rec.anomaly_rate, 3))
"
name: 208
signal length: (650000,)
beats: 2953
anomaly rate: 0.463
"""

# function that helps load a whole list of records
def load_split(record_names,data_dir="data/raw", preprocess_method="bandpass"):
    return [load_record_data(n,data_dir,preprocess_method) for n in record_names]


# create a function that walks to each heartbeat looks at the window and take the loadest shout
def aggregate_to_beats(sample_scores,r_peaks, pre=PRE,post=POST):
    n=len(sample_scores)
    out=np.empty(len(r_peaks),dtype=float)
    for i,peak in enumerate(r_peaks):
        low=max(0,peak-pre)
        high=min(n,peak+post)
        out[i]=sample_scores[low:high].max()
    return out




# function that slices the long recording into individual heartbeats and puts them into a table
def extract_beat_matrix(rec):
    return np.stack([rec.signal[p-PRE:p+POST] for p in rec.r_peaks])






