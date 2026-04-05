"""
Tensor utility functions for MATR.
Minimal implementation for cricket highlights functionality.
"""

import torch


def mask_logits(inputs, mask, mask_value=-1e30):
    """Mask logits with a large negative value."""
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


def make_mask(lengths, max_len=None):
    """Create a mask from lengths."""
    if max_len is None:
        max_len = lengths.max()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask
