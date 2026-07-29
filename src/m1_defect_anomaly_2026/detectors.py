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



