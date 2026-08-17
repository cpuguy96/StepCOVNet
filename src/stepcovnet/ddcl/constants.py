"""Hyperparameters for DDCL beat-grid placement (`omalley2025ddcl`).

Defaults follow the official trainer
https://github.com/miguelomalley/DDCL/blob/5b1375c642bb708b3c66baf5d880fbf865b85097/train_onset_model.py
and ``get_onset_model`` in ``models.py``.
"""

from stepcovnet.ddc import constants as ddc_constants

SAMPLE_RATE = ddc_constants.SAMPLE_RATE
FRAME_RATE = ddc_constants.FRAME_RATE
HOP_LENGTH = ddc_constants.HOP_LENGTH
N_MELS = ddc_constants.N_MELS
N_CHANNELS = ddc_constants.N_CHANNELS
N_SLOTS = 48
N_FRAMES_PER_BEAT = 32
MEMLEN = 15
STREAM_DIM = 2
CONV1_FILTERS = 16
CONV2_FILTERS = 32
CONV1_KERNEL = (7, 3)
CONV2_KERNEL = (3, 3)
POOL_SIZE = (1, 1, 3)
LSTM_UNITS = 200
LSTM_LAYERS = 2
DENSE_SIZES = (512, 256)
DROPOUT_RATE = 0.2
ADAM_LR = 1e-4
ADAM_CLIPNORM = 1.0
THRESHOLD_05 = 0.5
FEATURE_CACHE_SUFFIX = ddc_constants.FEATURE_CACHE_SUFFIX
LOG_EPS = ddc_constants.LOG_EPS
# ITGPT Table 2 on *expanded* Fraxtil (D-frax-exp), not Dataset A.
PUBLISHED_F1_AT_05_EXPANDED = 0.70
PUBLISHED_F1_MAX_EXPANDED = 0.76
