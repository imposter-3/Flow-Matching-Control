"""Flow matching tier: model, sources, training, checkpoints, adapters.

This package may import torch and may not import pydrake. The layering keeps
the 24-worker evaluation harness light and the training stack simulator-free.
"""
