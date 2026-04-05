import torch


def generalized_temporal_iou(spans1, spans2):
    """Generalized IoU for temporal spans.
    Args:
        spans1: (N, 2) tensor of [x1, x2]
        spans2: (M, 2) tensor of [x1, x2]
    Returns:
        giou: (N, M) tensor
    """
    # Clamp to ensure x1 <= x2
    spans1 = torch.stack([spans1[:, 0], torch.max(spans1[:, 0], spans1[:, 1])], dim=1)
    spans2 = torch.stack([spans2[:, 0], torch.max(spans2[:, 0], spans2[:, 1])], dim=1)

    inter_x1 = torch.max(spans1[:, None, 0], spans2[None, :, 0])
    inter_x2 = torch.min(spans1[:, None, 1], spans2[None, :, 1])
    inter = (inter_x2 - inter_x1).clamp(min=0)

    union_x1 = torch.min(spans1[:, None, 0], spans2[None, :, 0])
    union_x2 = torch.max(spans1[:, None, 1], spans2[None, :, 1])

    area1 = spans1[:, 1] - spans1[:, 0]
    area2 = spans2[:, 1] - spans2[:, 0]
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union.clamp(min=1e-6)
    enclosing = union_x2 - union_x1
    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-6)
    return giou


def span_xx_to_cxw(xx_spans):
    """Convert span from [x1, x2] to [cx, w] format.
    Args:
        xx_spans: (..., 2) tensor, x1 < x2
    Returns:
        cxw_spans: (..., 2) tensor
    """
    return torch.stack(
        [(xx_spans[..., 0] + xx_spans[..., 1]) / 2,
         xx_spans[..., 1] - xx_spans[..., 0]], dim=-1
    )


def span_cxw_to_xx(cxw_spans):
    """Convert span from [cx, w] to [x1, x2] format.
    Args:
        cxw_spans: (..., 2) tensor
    Returns:
        xx_spans: (..., 2) tensor
    """
    return torch.stack(
        [cxw_spans[..., 0] - cxw_spans[..., 1] / 2,
         cxw_spans[..., 0] + cxw_spans[..., 1] / 2], dim=-1
    )
