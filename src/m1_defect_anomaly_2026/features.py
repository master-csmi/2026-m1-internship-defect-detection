import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# count how many times the signal crosses zero
# a noisy or fragmented beat crosses much more often than a clean one
def zero_crossings(beat):
    signs = np.sign(beat)
    signs[signs==0]=1
    return int(np.sum(signs[1:]!=signs[:-1]))# count where there is consecutive elements have different signs