"""Event-based onset detection: raw audio in, continuous times and confidence out."""

from stepcovnet.onset_events import config
from stepcovnet.onset_events import inference
from stepcovnet.onset_events import models
from stepcovnet.onset_events import trainers

build_onset_event_model = models.build_onset_event_model
predict_onsets = inference.predict_onsets
train_onset_event = trainers.train_onset_event


__all__ = [
    "config",
    "build_onset_event_model",
    "train_onset_event",
    "predict_onsets",
]
