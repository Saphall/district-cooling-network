"""ML experiments: leakage-safe splits and run tracking.

Models/features are intentionally not prescribed here. Experiments read the
consolidated dataset (``datasets/<id>/consolidated/dataset.parquet``) and write
only under ``results/ml/``; they never touch simulator code or raw scenarios.
"""
