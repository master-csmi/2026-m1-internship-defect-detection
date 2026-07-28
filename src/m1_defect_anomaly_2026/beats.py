import numpy as np
from .preprocessing import FS, preprocess
from .splits import AAMI_MAP, BEAT_SYMBOLS

# window around R-peak

PRE = int(0.25*FS)
POST = int(0.40*FS)



