import numpy as np
import pandas as pd
import torch


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(labels)
    acc = correct.float().mean().item()
    return acc


def cosine_similarity(query_mat, target_mat):
    """
    query_mat: (n, dim)
    target_mat: (m, dim)
    """
    assert type(query_mat) == type(target_mat)

    inner_dot_mat = torch.matmul(query_mat, target_mat.t())  # (n, dim) @ (dim, m) -> (n, m)

    norm_mat = torch.matmul(
        torch.norm(query_mat, p=2, dim=1, keepdim=True),
        torch.norm(target_mat, p=2, dim=1, keepdim=True).t()
    )

    return inner_dot_mat / norm_mat


def euclidean_distance(query_mat, target_mat):
    """
    query_mat: (n, dim)
    target_mat: (m, dim)
    """
    assert type(query_mat) == type(target_mat)

    inner_dot_mat = torch.matmul(query_mat, target_mat.t())  # (n, dim) @ (dim, m) -> (n, m)
    qnorm = query_mat.pow(2).sum(1, keepdim=True)
    tnorm = target_mat.pow(2).sum(1, keepdim=True).t()
    dist = qnorm - 2 * inner_dot_mat + tnorm

    return dist


def map_at_k(y_true, y_pred, k=100):
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.ndim == 1 and y_pred.ndim == 2

    is_correct_mat = np.expand_dims(y_true, axis=1) == y_pred
    cumsum_mat = np.apply_along_axis(np.cumsum, axis=1, arr=is_correct_mat)
    arange_mat = np.expand_dims(np.arange(1, k + 1), axis=0)

    return np.mean((cumsum_mat / arange_mat) * is_correct_mat)


def gap(pred, conf, true, return_x=False):
    """
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    """
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap
