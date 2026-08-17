"""DDCL beat-grid step placement (`omalley2025ddcl`).

This package reimplements the published **onset / placement** stack from Dance
Dance ConvLSTM (O'Malley 2025). It is a cited baseline, not a new contribution.

Upstream code (commit ``5b1375c642bb708b3c66baf5d880fbf865b85097``):

- https://github.com/miguelomalley/DDCL
- ``smfiler.py`` (beat dicts / 48-slot labels)
- ``util.py`` (``label_to_vect_dict``, ``make_onset_feature_context_range``)
- ``models.py`` (``get_onset_model`` ConvLSTM)

Shared 10 ms 80-band log-mel is DDC PRE (`donahue2017ddc`) via
``stepcovnet.ddc.features``. Metric ``M-slot48`` is **not** ``M-ddc-20ms``.
"""

UPSTREAM_REPO = "https://github.com/miguelomalley/DDCL"
UPSTREAM_COMMIT = "5b1375c642bb708b3c66baf5d880fbf865b85097"
