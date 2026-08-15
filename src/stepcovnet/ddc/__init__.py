"""DDC-faithful step placement (`donahue2017ddc`).

PRE is 10 ms 80-band log-mel at 23/46/93 ms windows (`schluter2014onset`,
`hamel2012multiscale`). POST/metric ``M-ddc-20ms`` is Hamming peak-pick with a
±20 ms match. The C-LSTM is a cited baseline, not a new contribution.
"""
