"""Constants used throughout the stepcovnet project."""

# Target sample rate (Hz) for audio loading and processing (e.g. spectrogram, BPM estimation)
TARGET_SR = 44100

# Onset frame spacing in seconds (10 ms at 44.1 kHz with default hop_length)
HOP_COEFF = 0.01

# Number of Mel bands to generate
N_MELS = 128

# MERT hidden-state width (m-a-p/MERT-v1-330M layer outputs)
MERT_HIDDEN_SIZE = 1024

# Arrow prediction: 4 panels, base-4 encoding -> 4^4 = 256 types
N_ARROW_TYPES = 256

# Class index used for padding / ignore in arrow labels (trainer, dataset, metrics)
ARROW_PADDING_CLASS = 0

# Maximum number of arrows (steps) per sequence; used for padding and XLA fixed shapes
MAX_STEPS = 2048
