"""
Model utility functions for MATR.
Minimal implementation for cricket highlights functionality.
"""

import torch
import torch.nn as nn


def count_parameters(model, verbose=True):
    """Count trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total trainable parameters: {total_params:,}")
    return total_params


def reset_parameters(model):
    """Reset all parameters in a model."""
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
