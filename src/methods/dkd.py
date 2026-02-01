import torch
import torch.nn.functional as F


def _get_masks(target: torch.Tensor, num_classes: int):
    gt_mask = F.one_hot(target, num_classes=num_classes).bool()
    other_mask = ~gt_mask
    return gt_mask, other_mask


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = student_logits.shape[1]
    gt_mask, other_mask = _get_masks(target, num_classes)

    s = F.softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)

    s_gt = (s * gt_mask).sum(dim=1, keepdim=True)
    s_other = (s * other_mask).sum(dim=1, keepdim=True)
    t_gt = (t * gt_mask).sum(dim=1, keepdim=True)
    t_other = (t * other_mask).sum(dim=1, keepdim=True)

    s_tckd = torch.cat([s_gt, s_other], dim=1)
    t_tckd = torch.cat([t_gt, t_other], dim=1)
    tckd = F.kl_div(torch.log(s_tckd + eps), t_tckd, reduction="batchmean")

    s_nckd = (s * other_mask) / (s_other + eps)
    t_nckd = (t * other_mask) / (t_other + eps)
    nckd = F.kl_div(torch.log(s_nckd + eps), t_nckd, reduction="batchmean")

    return (alpha * tckd + beta * nckd) * (temperature ** 2)
