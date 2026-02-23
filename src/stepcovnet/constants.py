"""Constants used throughout the stepcovnet project."""

# Number of Mel bands to generate
N_MELS = 128

# Arrow prediction: 4 panels, base-4 encoding -> 4^4 = 256 types (0 = padding)
N_ARROW_TYPES = 256

# Maximum number of arrows (steps) per sequence; used for padding and XLA fixed shapes
MAX_STEPS = 2048
