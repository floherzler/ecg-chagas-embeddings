import torch


def tpr_at_5(preds, targets):
    k = int(0.05 * len(preds))
    topk_idx = torch.topk(preds, k).indices
    tp = (targets[topk_idx] > 0.5).sum().float()
    fn = (targets.sum() - tp).float()
    return (tp / (tp + fn + 1e-8)).item()
