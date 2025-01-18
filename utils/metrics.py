import torch

def accuracy_f(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2_f(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance_f(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)