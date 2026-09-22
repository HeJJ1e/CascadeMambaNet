import torch
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present', ignore=None):
    if probas.numel() == 0:
        return probas * 0.0
    if ignore is not None:
        valid = labels != ignore
        probas = probas[valid]
        labels = labels[valid]
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        (errors_sorted, perm) = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()

def lovasz_softmax(logits, labels, classes='present', ignore=None):
    probas = F.softmax(logits, dim=-1)
    return lovasz_softmax_flat(probas, labels, classes=classes, ignore=ignore)
