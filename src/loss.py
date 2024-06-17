from typing import Optional
import lightning as L

import numpy as np
import torch
import torch.nn.functional as F
from anytree import Node
from torch import nn

from .tree import (add_weight_to_nodes, get_ancestor_matrix,
                   get_subset_masks_and_weights, get_subsets_and_weights,
                   get_U)
from .metrics import get_dist_matrix


def get_hierarchical_loss(hierarchy, **kwargs):
    method = kwargs["method"]
    if method == "top_bottom":
        add_weight_to_nodes(hierarchy, method="greedy", total_weight=.5)
        add_weight_to_nodes(hierarchy, method="lazy", total_weight=.5, reduce="add")
    else:
        add_weight_to_nodes(hierarchy, **kwargs)

    return HierarchicalLoss(hierarchy)

def get_label_smoothing_loss(hierarchy, **kwargs):
    add_weight_to_nodes(hierarchy, **kwargs)
    distance_matrix = 1 - get_dist_matrix(hierarchy)
    distribution_matrix = nn.functional.normalize(distance_matrix, dim=1, p=1)

    return LabelSmoothingLoss(kwargs["smoothing_value"], distribution_matrix)

def get_soft_labels_loss(hierarchy, **kwargs):
    add_weight_to_nodes(hierarchy, method="ones")

    distribution_matrix = get_dist_matrix(hierarchy)

    return SoftLabelsLoss(kwargs["beta"], distribution_matrix)



def get_bertinetto_loss(hierarchy, **kwargs):
    add_weight_to_nodes(hierarchy, method="bertinetto", **kwargs)
    return BertinettoHierarchicalLoss(hierarchy)


class HierarchicalLoss(nn.Module):
    """Compute hierarchical loss from a given hierarchy.

    Args:
        hierarchy: weighted hierarchy as a `treenode` with `weight` attributes
    """

    def __init__(self, hierarchy: Node):
        super().__init__()
        self.hierarchy = hierarchy
        self.depth = hierarchy.height

        # Masks of subsets for each class. Size is `n_classes==n_leaves` * `n_leaves` *
        # `depth`. Weights corresponding to these subsets. Size is `n_leaves` * `depth`.
        masks, weights = get_subset_masks_and_weights(self.hierarchy)

        self.register_buffer(f"masks", torch.tensor(masks, dtype=torch.bool))
        self.register_buffer(f"weights", torch.tensor(weights, dtype=torch.float))

    def forward(self, input, target):
        # Subset masks for each sample. Size is `n_samples` * `n_leaves` *
        # `depth`
        mask = self.masks[target]

        # Weights associated with each subset for each class. Size is
        # `n_samples` * `depth`
        weights = self.weights[target]

        # Duplicating `input` along a third new dimension so that `dup_input` is
        # `n_samples` * `n_leaves` * `depth`
        dup_input = input.unsqueeze(-1).repeat(1, 1, self.depth)

        # Setting to -inf value that are not part of a subset for logsumexp to
        # ignore them
        dup_input[~mask] = float("-inf")

        # Log-sum-exp of each subset. Size `n_samples` * `depth`
        lse_subsets = torch.logsumexp(dup_input, dim=1) - torch.logsumexp(input, dim=1, keepdim=True)

        # Some logsumexps might be -inf because there is fewer subsets than
        # maximum depth
        lse_subsets[lse_subsets == float("-inf")] = 0

        # Add weight in front of each log
        lse_subsets = lse_subsets * weights

        # Sum all weighted log, average over number of samples
        n_samples = len(target)
        return - torch.sum(lse_subsets) / n_samples


class BertinettoHierarchicalLoss(HierarchicalLoss):
    """Compute hierarchical loss from Bertinetto et al.

    Args:
        hierarchy: weighted hierarchy as a `treenode` with `weight` attributes
    """

    def __init__(self, hierarchy: Node):
        super().__init__(hierarchy)
        self.hierarchy = hierarchy
        self.depth = hierarchy.height

        # Masks of subsets for each class. Size is `n_classes==n_leaves` * `n_leaves` *
        # `depth`. Weights corresponding to these subsets. Size is `n_leaves` * `depth`.
        masks, weights = get_subset_masks_and_weights(self.hierarchy)

        # Adapt weights to match the definition of hierarchical loss from
        # Bertinetto. This works because nodes are listed in topological order
        # by `get_nodes`.
        weights[:, 1:] = weights[:, 1:] - weights[:, :-1]

        self.register_buffer("masks", torch.tensor(masks, dtype=torch.bool))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float))


class HierarchicalRegularization(nn.Module):
    """Compute hierarchical regularization of logistic regression weights."""

    def __init__(self, hierarchy: Node, lambdas=None):
        super().__init__()
        self.hierarchy = hierarchy

        if lambdas is not None:
            self.lambdas = lambdas
        else:
            self.lambdas = np.ones(len(hierarchy.descendants) + 1)

        L = get_ancestor_matrix(self.hierarchy)
        U = get_U(hierarchy, self.lambdas)
        self.register_buffer("C", torch.tensor(L @ U @ L.T))

    def forward(self, weight):
        return torch.sum(self.C * (weight @ weight.T))


class HierarchicalRegularizationGamma(nn.Module):
    """Compute hierarchical regularization of logistic regression weights."""

    def __init__(self, dist_matrix):
        super().__init__()

        gamma = torch.inverse(torch.eye(dist_matrix.shape[0]) - torch.pow(dist_matrix,2)/2)
        self.register_buffer("C", gamma)

    def forward(self, weight):
        return torch.sum(self.C * (weight @ weight.T))


class CrossEntropyThenHierarchicalLoss(L.LightningModule):

    def __init__(self, hierarchy: Node, method: str, total_weight: float, epoch_for_switch: int):
        super().__init__()
        self.hierarchical_loss = get_hierarchical_loss(hierarchy, **{"method": method, "total_weight": total_weight})
        self.cross_entropy = nn.CrossEntropyLoss()
        self.epoch_for_switch = epoch_for_switch

    def forward(self, input, target):
        if self.current_epoch < self.epoch_for_switch:
            return self.cross_entropy(input, target)
        else:
            return self.hierarchical_loss(input ,target)


class LabelSmoothingLoss(nn.Module):
    """Class implementing a label smoothing loss function.

    Args:
        smoothing (float): The smoothing parameter, a real number between 0 and
            1. A higher value results in more smoothing. Default is 0.1.
        dists (torch.Tensor or str): The probability distribution matrix for
            smoothing or a string indicating a uniform distribution. If `dists`
            is the string "uniform", a uniform distribution is used. Each row of
            the matrix is the probability distribution of one class that is
            linearly combined with a one-hot distribution. Default is "uniform".
        num_classes (int): The number of classes. This parameter is required if
            `dists` is a string indicating a uniform distribution. Default is
            None.

    Raises:
        Exception: If `dists` is a string "uniform" but `num_classes` is not
            specified.

    Methods:
        forward(outputs, target): Computes the label smoothing loss. `outputs`
            are the logits and `target` the labels.

    """
    def __init__(self, smoothing: float = 0.1, dists="uniform", num_classes=None):
        super(LabelSmoothingLoss, self).__init__()

        if isinstance(dists, str) and num_classes is None:
            raise Exception("If `dists` is a string, `num_classes` must be specified")

        if dists == "uniform":
            dists = torch.full((num_classes, num_classes), 1 / num_classes)
        else:
            num_classes = dists.shape[0]

        self.smoothing = smoothing
        self.register_buffer("dists", (1 - self.smoothing) * torch.eye(num_classes) + self.smoothing * dists)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        target = self.dists[target]

        return self.loss(outputs, target)


class SoftLabelsLoss(nn.Module):
    """Class implementing a Soft Label Loss function as definied in Bertinetto et al.

    Args:
        beta (float): the beta parameter for the loss
        dists (torch.Tensor or str): The distance matrix between the classes in the hierarchy.

    Methods:
        forward(outputs, target): Computes the soft label loss. `outputs`
            are the logits and `target` the labels.

    """
    def __init__(self, beta: float = 10, dists=None):
        super(SoftLabelsLoss, self).__init__()

        self.beta = beta
        self.exp_dists = torch.exp( - beta * dists)
        self.register_buffer("dists", self.exp_dists/torch.sum(self.exp_dists, dim=0))
        self.loss = nn.CrossEntropyLoss()



    def forward(self, outputs, target):
        target = self.dists[target]

        return self.loss(outputs, target)