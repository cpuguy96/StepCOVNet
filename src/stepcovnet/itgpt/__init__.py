"""ITGPT hierarchical-transformer step placement (`omalley2026itgpt`).

This package reimplements the published **onset / placement** stack from
ITGPT (O'Malley 2026). It is a cited baseline, not a new contribution.

Upstream: https://github.com/miguelomalley/ITGPT (``onset.py``).

Beat-grid PRE and ``M-slot48`` labels are shared with DDCL
(``stepcovnet.ddcl``). Diagnostic-net regularization is omitted: upstream
``--lambda_diag`` defaults to 0.
"""

UPSTREAM_REPO = "https://github.com/miguelomalley/ITGPT"
