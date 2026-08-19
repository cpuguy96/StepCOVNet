"""Hyperparameters for ITGPT beat-grid placement (`omalley2026itgpt`).

Defaults follow ``OnsetConfig`` / ``train_onset`` in
https://github.com/miguelomalley/ITGPT/blob/main/onset.py
"""

from stepcovnet.ddcl import constants as ddcl_constants

N_MELS = ddcl_constants.N_MELS
N_CHANNELS = ddcl_constants.N_CHANNELS
N_SLOTS = ddcl_constants.N_SLOTS
N_FRAMES_PER_BEAT = ddcl_constants.N_FRAMES_PER_BEAT
CNN_HIDDEN = 32
CNN_FRAMES_OUT = 24
CHUNK_ALIGN = 64
D_MODEL = 256
N_HEADS = 8
N_ENC_LAYERS = 8
DROPOUT_RATE = 0.1
MAX_BEATS = 2000
MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 50.0
MIN_BPM = 40.0
MAX_BPM = 400.0
ADAM_LR = 1e-4
ADAM_WEIGHT_DECAY = 1e-2
ADAM_CLIPNORM = 1.0
THRESHOLD_05 = ddcl_constants.THRESHOLD_05
# ITGPT Table 2 on expanded Fraxtil (D-frax-exp).
PUBLISHED_F1_AT_05_EXPANDED = 0.78
PUBLISHED_F1_MAX_EXPANDED = 0.80
# 16th / 24th / 32nd grid weights from ``get_grid_importance_weights``.
GRID_WEIGHT_16TH = 2.0
GRID_WEIGHT_24TH = 1.0
GRID_WEIGHT_MICRO = 0.5
INDICES_16TH = (0, 12, 24, 36)
INDICES_24TH = (8, 16, 32, 40)
INDICES_32ND = (6, 18, 30, 42)
