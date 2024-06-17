import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from anytree import Node, PreOrderIter
from src.loss import BertinettoHierarchicalLoss, HierarchicalLoss
from src.tree import add_weight_to_nodes


def params(d):
    return pytest.mark.parametrize(
        argnames=(argnames := sorted({k for v in d.values() for k in v.keys()})),
        argvalues=[[v.get(k) for k in argnames] for v in d.values()],
        ids=d.keys(),
    )


tree1 = Node("root", children=[
    Node("sub0", children=[
        Node("sub0B"),
        Node("sub0A"),
    ]),
    Node("sub1", children=[
        Node("sub1A"),
        Node("sub1B", children=[]),
        Node("sub1C", children=[
            Node("sub1Ca"),
        ]),
    ]),
])


def compute_result(logits, target):
    if len(target) == 1 and target[0] == 0:
        denom = sum(math.exp(e) for e in logits[0])
        subset1 = -3 * math.log(math.exp(1) / denom)
        subset2 = - 3 * math.log((math.exp(1.0) + math.exp(2.0)) / denom)
        return subset1 + subset2

    raise NotImplementedError


@params(dict(
    blah = dict(
        logits=[1., 2., 1.5, 2.5, 0],
        target=[0]
    )
))
def test_hierarchical_loss(logits, target):
    add_weight_to_nodes(tree1, total_weight=6)
    loss = HierarchicalLoss(tree1)

    logits = torch.tensor([logits,], dtype=torch.double)
    target = torch.tensor(target)

    value = loss(logits, target).item()
    value_target = compute_result(logits, target)

    assert value == value_target


def compute_bertinetto_result(tree, logits, target):
    softmax = F.softmax(logits, dim=1).squeeze()

    for leaf, value in zip(tree.leaves, softmax):
        leaf.value = value.item()

    leaf = list(tree.leaves)[target]
    alpha=1

    current_node = leaf
    loss = 0
    while not current_node.is_root:
        sum_current_node = sum(e.value for e in current_node.leaves)
        sum_parent_node = sum(e.value for e in current_node.parent.leaves)

        loss += - math.exp(- alpha * current_node.depth) * math.log(sum_current_node / sum_parent_node)

        current_node = current_node.parent

    return loss


@pytest.mark.parametrize("target", range(5))
@params(dict(
    setup0 = dict(
        tree=tree1,
        logits=[1., 2., 1.5, 2.5, 0],
    )
))
def test_bertinetto_hierarchical_loss(tree, logits, target):
    add_weight_to_nodes(tree, total_weight=1, method="bertinetto", alpha=1)
    loss = BertinettoHierarchicalLoss(tree)

    logits = torch.tensor([logits,], dtype=torch.double)
    target = torch.tensor([target])

    value = loss(logits, target).item()
    value_target = compute_bertinetto_result(tree, logits, target)

    assert np.allclose(value, value_target)
