import numpy as np


# for each point we look at the window samples around it and compute the average and std
def rolling_stats(signal,window):
    n=len(signal)
    half=window//2
    mean=np.empty(n)
    std=np.empty(n)

    for i in range(n):
        low=max(0,i-half) #start of the window
        high=min(n,i+half+1) #end of the window
        chunk=signal[low:high]
        mean[i]=chunk.mean()
        std[i]=chunk.std()
    return mean,std



# score each sample by how far it sits from its average
def zscore_detector(signal,window=360,eps=1e-6):
    mean, std=rolling_stats(signal,window)
    return np.abs(signal-mean)/(std+eps) # abs because a spike up or a dip down are both unusual




# CUSUM is a detector that detects gradual and sustained shifts in the signal
# At each step we add how far the signal is from normal inus a small allowance k
# window if is "normal" it is measuredon a rolling window instead the whole signal
def cusum_detector(signal, threshold_k=0.5,window=None):
    if window is None:
        mean = signal.mean()
        std=signal.std() +1e-12
        z=(signal-mean)/std
    else:
        mean,std=rolling_stats(signal, window)
        z=(signal-mean)/(std+1e-6)


    n=len(z)
    up_scores=np.zeros(n)
    down_scores=np.zeros(n)
    up=0.0
    down=0.0
    for i in range(n):
        up=max(0.0,up+z[i]-threshold_k)
        down=max(0.0,down-z[i]-threshold_k)
        up_scores[i]=up
        down_scores[i]=down

    return np.maximum(up_scores,down_scores)


