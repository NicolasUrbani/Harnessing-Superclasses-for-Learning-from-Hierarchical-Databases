import numpy as np
import pytest
import torch
from anytree import Node
from src.metrics import (HierarchicalAccuracy, WassersteinTreeAccuracy,
                         WassersteinTreeDistance, get_dist_matrix)
from src.tree import add_weight_to_nodes


tree1 = Node("root", children=[Node("sub0"), Node("sub1")])
add_weight_to_nodes(tree1)

tree2 = Node("root", children=[Node("sub0"), Node("sub1", children=[Node("sub0a"), Node("sub0b")])])
add_weight_to_nodes(tree2)


def params(d):
    "Adapted from https://github.com/pytest-dev/pytest/issues/7568#issuecomment-665966315"
    return pytest.mark.parametrize(
        argnames=(argnames := sorted({k for v in d for k in v.keys()})),
        argvalues=[[v.get(k) for k in argnames] for v in d],
    )


@params([
     dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        input = [[0.9, 0.1]],
        target = [0],
        result = [(1 - 0.9) + 0.1]
    ),
    dict(
        tree = tree1,
        kwargs = dict(average="none"),
        input = [[0.9, 0.1]],
        target = [0],
        result = [(1 - 0.9) + 0.1, 0]
    ),
    dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        input = [[0.9, 0.1],
                 [0.2, 0.8]],
        target = [0, 1],
        result = [((1 - 0.9) + 0.1 + 0.2 + (1 - 0.8))/2]
    ),
    dict(
        tree = tree1,
        kwargs = dict(average="none"),
        input = [[0.9, 0.1],
                 [0.2, 0.8]],
        target = [0, 1],
        result = [(1 - 0.9) + 0.1, 0.2 + (1 - 0.8)]
    ),
    dict(
        tree = tree2,
        kwargs = dict(average="macro"),
        input = [[0.9, 0.05, 0.05]],
        target = [0],
        result = [1*(1 - 0.9)+0.5*0.05+0.5*0.05+.5*(0.05+0.05)]
    ),
    dict(
        tree = tree2,
        kwargs = dict(average="none"),
        input = [[0.9, 0.05, 0.05]],
        target = [0],
        result = [1*(1 - 0.9)+0.5*0.05+0.5*0.05+.5*(0.05+0.05), 0, 0]
    )
])
def test_wasserstein_tree_distance_1(tree, kwargs, input, target, result):
    m = WassersteinTreeDistance(tree, **kwargs)

    input = torch.tensor(input, dtype=torch.float)
    target = torch.tensor(target)
    m.update(input, target)

    res = m.compute()

    assert torch.allclose(res, torch.tensor(result))


@params([
    dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        inputs = (
            [[0.9, 0.1]],
            [[0.2, 0.8]]
        ),
        targets = (
            [0],
            [1]
        ),
        result = [((1 - 0.9) + 0.1 + 0.2 + (1 - 0.8))/2]
    ),
    dict(
        tree = tree1,
        kwargs = dict(average="none"),
        inputs = (
            [[0.9, 0.1]],
            [[0.2, 0.8]]
        ),
        targets = (
            [0],
            [1]
        ),
        result = [(1 - 0.9) + 0.1, 0.2 + (1 - 0.8)]
    ),
    dict(
        tree = tree1,
        kwargs = dict(average="samplewise"),
        inputs = (
            [[0.9, 0.1]],
            [[0.3, 0.7]],
            [[0.2, 0.8]]
        ),
        targets = (
            [0],
            [0],
            [1]
        ),
        result = [(1 - 0.9) + 0.1, (1 - 0.3) + 0.7, 0.2 + (1 - 0.8)]
    ),
])
def test_wasserstein_tree_distance_2(tree, kwargs, inputs, targets, result):
    m = WassersteinTreeDistance(tree, **kwargs)

    for input, target in zip(inputs, targets):
        input = torch.tensor(input, dtype=torch.float)
        target = torch.tensor(target)
        m.update(input, target)

    res = m.compute()

    assert torch.allclose(res, torch.tensor(result))


@params([
    dict(
        tree=tree1,
        result=[[0., 1], [1, 0]]
    ),
    dict(
        tree=tree2,
        result=[[0, 1, 1], [1, 0, 0.5], [1, 0.5, 0]]
    ),
])
def test_get_dist_matrix(tree, result):
    A = get_dist_matrix(tree)
    result = torch.tensor(result)
    assert torch.allclose(A, result)


@params([
     dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        input = [0],
        target = [0],
        result = [1.]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        input = [0],
        target = [1],
        result = [0.]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        input = [0, 0],
        target = [0, 1],
        result = [.5]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="macro"),
        input = [0, 1, 0],
        target = [0, 1, 1],
        result = [0.666666666666]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="none"),
        input = [0],
        target = [0],
        result = [1., 0]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="none"),
        input = [0],
        target = [1],
        result = [0., 0.]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="none"),
        input = [0, 0],
        target = [0, 1],
        result = [1, 0.]
     ),
     dict(
        tree = tree1,
        kwargs = dict(average="none"),
        input = [0, 1, 0],
        target = [0, 1, 1],
        result = [1, .5]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="macro"),
        input = [0],
        target = [0],
        result = [1.]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="macro"),
        input = [1],
        target = [2],
        result = [0.5]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="macro"),
        input = [0, 1],
        target = [0, 2],
        result = [.75]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="macro"),
        input = [0, 1, 0],
        target = [0, 2, 1],
        result = [0.5]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="none"),
        input = [0],
        target = [0],
        result = [1., 0, 0]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="none"),
        input = [0],
        target = [1],
        result = [0., 0., 0]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="none"),
        input = [0, 1],
        target = [0, 2],
        result = [1, 0, 0.5]
     ),
     dict(
        tree = tree2,
        kwargs = dict(average="none"),
        input = [0, 1, 0],
        target = [0, 2, 2],
        result = [1., 0, 0.25]
     ),
])
def test_hierarchical_accuracy(tree, kwargs, input, target, result):
    m = HierarchicalAccuracy(tree, **kwargs)

    input = torch.tensor(input)
    target = torch.tensor(target)

    m.update(input, target)
    res = m.compute()

    assert torch.allclose(res, torch.tensor(result))



# def test_wasserstein_tree_accuracy(tree, kwargs, input, target, result):
#     m = WassersteinTreeAccuracy(tree, **kwargs)

#     input = torch.tensor(input)
#     target = torch.tensor(target)

#     m.update(input, target)
#     res = m.compute()

#     assert torch.allclose(res, torch.tensor(result))
