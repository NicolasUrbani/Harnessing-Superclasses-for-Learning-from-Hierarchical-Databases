from typing import Literal, Optional

import numpy as np
import torch
from anytree import Node, util
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.metric import Metric
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import dim_zero_cat

from .tree import add_weight_to_nodes, get_ancestor_matrix, get_nodes


class WassersteinTreeAccuracy(MulticlassAccuracy):
    """Compute accuracy with corrected logits.

    Compute a classic accuracy from given logits but apply a correction on the
    softmax values before based on the hierarchy.

    Args:
        hierarchy: weighted hierarchy as a `treenode` with `weight` attributes
        average: per-class (none) or aggregated (macro) accuracy

    """

    higher_is_better: bool = True

    def __init__(
        self,
        hierarchy: Node,
        average: Optional[Literal["none", "macro"]] = "macro",
    ):
        super().__init__(average=average, num_classes=len(hierarchy.leaves))
        self.correction_matrix = 1 - get_dist_matrix(hierarchy)

    def update(self, logits, target):
        softmaxes = torch.softmax(logits, dim=1)
        super().update(softmaxes @ self.correction_matrix.to(logits.device), target)


class WassersteinTreeDistance(Metric):
    """Compute Wasserstein tree distance

    Compute a distance between a distribution on leaves (softmax) and an
    observed class. The distance takes into account that there is a hierarchical
    structure on the leaves.

    Args:
        hierarchy: weighted hierarchy as a `treenode` with `weight` attributes
        average: per-class (none), aggregated (macro) accuracy or for each
        sample (samplewise)
    """

    higher_is_better: bool = False

    def __init__(
        self,
        hierarchy: Node,
        average: Optional[Literal["none", "macro", "samplewise"]] = "macro",
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.ancestor_matrix = get_ancestor_matrix(self.hierarchy, include_root=False)
        self.weights = [n.weight for n in get_nodes(hierarchy, include_root=False)]
        self.average = average

        num_classes = len(self.hierarchy.leaves)

        if self.average == "samplewise":
            self.add_state("dists", default=[], dist_reduce_fx=None)
        elif self.average in ["none", "macro"]:
            size = num_classes if average == "none" else 1
            self.add_state(
                "num_samples",
                default=torch.zeros(size, dtype=torch.long),
                dist_reduce_fx="sum",
            )
            self.add_state("dists", default=torch.zeros(size), dist_reduce_fx="sum")
        else:
            raise ValueError("Unknown aggregation", self.average)

    def update(self, preds, target):
        device = preds.device

        ancestor_matrix = torch.tensor(self.ancestor_matrix, device=device, dtype=torch.float)
        weights = torch.tensor(self.weights, device=device)

        a = torch.einsum("ln,bl,n->bn", ancestor_matrix, preds, weights)
        b = ancestor_matrix[target] * weights
        values = torch.sum(torch.abs(a - b), dim=1)

        if self.average == "samplewise":
            self.dists.append(values)
        elif self.average == "macro":
            values = torch.sum(values)
            self.dists += values
            self.num_samples += len(target)
        else:
            self.dists.scatter_add_(0, target, values)
            self.num_samples.scatter_add_(0, target, torch.ones_like(target, device=device))

    def compute(self):
        if self.average == "samplewise":
            return dim_zero_cat(self.dists)
        else:
            return _safe_divide(self.dists, self.num_samples)


def get_dist_matrix(tree: Node, attr="weight") -> Tensor:
    """Return n_leaves * n_leaves tensor of distances between leaves given hierarchy `tree`.

    Distance between two leaves is the maximum length between lowest common
    ancestor and leaf.

    If the weights are balanced (sum to 1 for each path from root to leaf) the
    distance is half the length between leaves.

    If the weights are all 1, the distance is the height of the lowest common ancestor.

    """

    n_leaves = len(tree.leaves)
    dist = torch.zeros((n_leaves, n_leaves), dtype=torch.float32)

    for i, leaf_i in enumerate(tree.leaves):
        for j, leaf_j in enumerate(tree.leaves):
            if j > i:
                continue
            elif i == j:
                dist[i, i] = 0
            else:
                ancestor = util.commonancestors(leaf_i, leaf_j)[-1]

                value_i = 0
                current = leaf_i
                while current != ancestor:
                    value_i += getattr(current, attr)
                    current = current.parent

                value_j = 0
                current = leaf_j
                while current != ancestor:
                    value_j += getattr(current, attr)
                    current = current.parent

                dist[i, j] = dist[j, i] = max(value_i, value_j)

    return dist


class HierarchicalAccuracy(Metric):
    """Compute hierarchical accuracy for multiclass tasks.

    Hierarchical accuracy is one minus the distance between predicted leaf and
    observed leaf. The distance between two leaves is the length of any of the
    two leaves to their deepest common ancestor. The weights of the hierarchy
    must sum to 1 to each path from any leaf to the root.

    Args:
        hierarchy: weighted hierarchy as a `treenode` with `weight` attributes
        average: per-class (none) or aggregated (macro) accuracy

    """

    is_differentiable: bool = False
    higher_is_better: bool = False

    def __init__(
        self,
        hierarchy: Node,
        average: Optional[Literal["none", "macro"]] = "macro",
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.average = average
        self.score_matrix = 1 - get_dist_matrix(hierarchy)

        num_classes = len(self.hierarchy.leaves)
        size = num_classes if average == "none" else 1
        self.add_state(
            "num_samples",
            default=torch.zeros(size, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state("scores", default=torch.zeros(size), dist_reduce_fx="sum")

    def update(self, preds, target):
        device = preds.device

        values = self.score_matrix.to(device)[preds, target]

        if self.average == "macro":
            values = torch.sum(values)
            self.scores += values
            self.num_samples += len(target)
        else:
            self.scores.scatter_add_(0, target, values)
            self.num_samples.scatter_add_(0, target, torch.ones_like(target, device=device))

    def compute(self):
        return _safe_divide(self.scores, self.num_samples)


class AccuracyThreshold(Metric):
    """Compute accuracy at different depths according to a hierarchy.

    Compute accuracies on pruned hierarchies. Hierarchies are pruned according
    to given thresholds. For one threshold, leaves are merged if their distance
    are less than the threshold. No pruning if the threshold is zero, tree is
    pruned down to the root only if threshold is 1 (and total_weight is 1).

    Args:
        hierarchy: weighted hierarchy as a `treenode` with `weight` attributes
        thresholds: list of thresholds to prune the tree
        average: per-class (none) or aggregated (macro) accuracy

    """

    def __init__(
        self,
        hierarchy: Node,
        thresholds: list,
        average: Optional[Literal["none", "micro"]] = "micro",
    ):
        super().__init__()
        self.tree = hierarchy
        self.thresholds = thresholds

        self.renumber_classes, self.merge_softmaxes = get_merge_tensors(
            self.tree, self.thresholds
        )

        num_classes = len(self.tree.leaves)
        self.accuracies = MetricCollection(
            {
                f"acc_{i}": MulticlassAccuracy(average=average, num_classes=num_classes)
                for i in range(len(thresholds))
            }
        )

    def update(self, preds, target, softmax_output):
        device = preds.device

        # Merge softmaxes according to all thresholds
        merged_softmaxes = torch.einsum(
            "tkc,bk->tbc", self.merge_softmaxes.to(device), softmax_output
        )

        # Compute new classes for all thresholds
        new_classes = self.renumber_classes.to(device)[:, target]

        for i, (softmaxes, klass) in enumerate(zip(merged_softmaxes, new_classes)):
            self.accuracies[f"acc_{i}"].update(softmaxes, klass)

    def compute(self):
        return {
            f"acc_{threshold}": self.accuracies[f"acc_{i}"].compute()
            for i, threshold in enumerate(self.thresholds)
        }


def get_thresholds(tree):
    weights = np.full((len(tree.leaves), tree.height + 1), float("-inf"))

    for i, leaf in enumerate(tree.leaves):
        ancestors = list(leaf.ancestors) + [leaf]
        leaf_weights = np.array([n.weight for n in ancestors])
        weights[i, : len(leaf_weights)] = leaf_weights

    acc_weights = np.cumsum(weights, axis=1)
    acc_weights = acc_weights[acc_weights >= 0]
    acc_weights = np.unique(acc_weights)

    # Get mid-point thresholds
    cut_thresholds = (acc_weights[1:] + acc_weights[:-1]) / 2

    # Remove to close
    cut_thresholds = cut_thresholds[np.diff(acc_weights) > 1e-5]

    # List all of these
    return cut_thresholds


def super_class_threshold(tree, threshold):
    """Return a tuple of superclasses and aggregation matrix according to `threshold` in `tree`."""

    def set_klass_attr(node, cumulative_weight):
        for child in node.children:
            if child.weight + cumulative_weight > threshold and not child.is_leaf:
                # Same klass to each descendant leaf
                klass = child.leaves[0].klass
                for leaf in child.leaves:
                    leaf.klass = klass
            else:
                set_klass_attr(child, child.weight + cumulative_weight)

    # Set `klass` attribute for each leaf to its fine-grain class
    for i, leaf in enumerate(tree.leaves):
        leaf.klass = i

    # Set `klass` attribute for each leaf according to `threshold`
    set_klass_attr(tree, 0)

    # Extract new super-classes for each leaf
    super_classes = np.array([leaf.klass for leaf in tree.leaves])

    # Renumber these super classes to have a contiguous set of new classes
    # [0, 1, 2, 2, 2, 5, 6] -> [0, 1, 2, 2, 2, 3, 4]
    _, super_classes = np.unique(super_classes, return_inverse=True)

    aggregation_matrix = np.zeros((len(super_classes), len(super_classes)))
    aggregation_matrix[np.arange(len(super_classes)), super_classes] = 1.0

    return super_classes, aggregation_matrix


def get_merge_tensors(tree, thresholds):
    """Return a couple of merging tensors.

    First tensor is `n_thresholds` * `n_classes` * `n_classes`. Each `n_classes`
    * `n_classes` slice is a matrix that is merging the softmaxes according to
    the hierarchy and the chosen threshold.

    Second tensor is `n_thresholds` * `n_classes`. Each row contains the
    renumbering of each class.

    """

    class_and_matrix_list = [
        super_class_threshold(tree, threshold) for threshold in thresholds
    ]

    super_classes_list = [
        torch.tensor(a, dtype=torch.float32) for a, b in class_and_matrix_list
    ]
    aggregation_matrix_list = [
        torch.tensor(b, dtype=torch.float32) for a, b in class_and_matrix_list
    ]

    return torch.stack(super_classes_list), torch.stack(aggregation_matrix_list)


