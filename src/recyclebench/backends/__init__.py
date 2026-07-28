"""Real (non-mock) backends: target models, the intent gate, judges, and the reformulator.

All heavy dependencies are imported lazily inside these modules, so recyclebench installs
and its mock/validation paths run with only numpy/scipy/pandas/statsmodels present.
"""
