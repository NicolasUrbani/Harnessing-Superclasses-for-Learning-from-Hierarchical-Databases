import anytree
import hydra
import numpy as np
import pytest
from anytree import Node, PreOrderIter
from hydra import compose, initialize
from omegaconf import OmegaConf
from src.tree import (add_weight_to_nodes, get_ancestor_matrix,
                      get_subset_masks_and_weights, get_subsets_and_weights)


tree = Node("root", children=[
    Node("sub0", children=[
        Node("sub0B"),
        Node("sub0A"),
    ]),
    Node("sub1", children=[
        Node("sub1A"),
        Node("sub1B"),
        Node("sub1C", children=[
            Node("sub1Ca"),
        ]),
    ]),
])

ancestor_matrix = np.array([
    [1, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1],
])

ancestor_matrix_no_root = np.array([
    [1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1],
])

ancestor_matrix_no_leaves = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
])

ancestor_matrix_no_root_no_leaves = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
])

test_data = [
    (tree, ancestor_matrix, {}),
    (tree, ancestor_matrix_no_root, dict(include_root=False)),
    (tree, ancestor_matrix_no_leaves, dict(include_leaves=False)),
    (tree, ancestor_matrix_no_root_no_leaves, dict(include_root=False, include_leaves=False))
]


@pytest.mark.parametrize("tree,ancestor_matrix,kwargs", test_data)
def test_get_ancestor_matrix(tree, ancestor_matrix, kwargs):
    am = get_ancestor_matrix(tree, **kwargs)
    assert np.allclose(am, ancestor_matrix)


test_data = [
    (tree, dict(total_weight=6, method="balanced"), (
        ("sub0", 3),
        ("sub0B", 3),
        ("sub0A", 3),
        ("sub1", 2),
        ("sub1A", 4),
        ("sub1B", 4),
        ("sub1C", 2),
        ("sub1Ca", 2),
    )),
    (tree, dict(total_weight=6, method="greedy"), (
        ("sub0", 4),
        ("sub0B", 2),
        ("sub0A", 2),
        ("sub1", 2),
        ("sub1A", 4),
        ("sub1B", 4),
        ("sub1C", 2),
        ("sub1Ca", 2),
    )),
    (tree, dict(total_weight=6, method="lazy"), (
        ("sub0", 2),
        ("sub0B", 4),
        ("sub0A", 4),
        ("sub1", 2),
        ("sub1A", 4),
        ("sub1B", 4),
        ("sub1C", 2),
        ("sub1Ca", 2),
    )),
    (tree, dict(total_weight=42, method="power", decay=1/2), (
        ("sub0", 28),
        ("sub0B", 14),
        ("sub0A", 14),
        ("sub1", 24),
        ("sub1A", 18),
        ("sub1B", 18),
        ("sub1C", 12),
        ("sub1Ca", 6),
    )),
]


@pytest.mark.parametrize("tree,kwargs,result", test_data)
def test_add_weight_to_nodes(tree, kwargs, result):
    add_weight_to_nodes(tree, **kwargs)
    for name, weight in result:
        assert anytree.find_by_attr(tree, name, name="name").weight == weight


testdata = [
    ("tinyimagenet", "balanced"),
    ("tinyimagenet", "lazy"),
    ("tinyimagenet", "greedy"),
    ("tinyimagenet", "lazy"),
    ("tinyimagenet", "leaf_proportion")
]

@pytest.mark.parametrize("dataset,method", testdata)
def test_add_weight_to_nodes_2(dataset, method):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="main", overrides=[f"dataset={dataset}"])
        hierarchy = hydra.utils.instantiate(cfg.dataset.datasets.hierarchy)

        add_weight_to_nodes(hierarchy, method=method)
        for leaf in hierarchy.leaves:
            assert np.allclose(sum(n.weight for n in list(leaf.ancestors) + [leaf] if not n.is_root), 1)


def test_get_subsets_and_weights():
    class_indexes, class_weights = get_subsets_and_weights(tree, total_weight=6, method="balanced")

    class_indexes_target = [np.array([[1, 1],
                                      [1, 0],
                                      [0, 0],
                                      [0, 0],
                                      [0, 0]]),
                            np.array([[1, 0],
                                      [1, 1],
                                      [0, 0],
                                      [0, 0],
                                      [0, 0]]),
                            np.array([[0, 0],
                                      [0, 0],
                                      [1, 1],
                                      [1, 0],
                                      [1, 0]]),
                            np.array([[0, 0],
                                      [0, 0],
                                      [1, 0],
                                      [1, 1],
                                      [1, 0]]),
                            np.array([[0, 0, 0],
                                      [0, 0, 0],
                                      [1, 0, 0],
                                      [1, 0, 0],
                                      [1, 1, 1]])]

    assert len(class_indexes) == len(class_indexes_target)

    for a, b in zip(class_indexes_target, class_indexes):
        assert np.allclose(a, b)

    class_weights_target = [np.array([3., 3.]),
                            np.array([3., 3.]),
                            np.array([2., 4.]),
                            np.array([2., 4.]),
                            np.array([2., 2., 2.])]

    assert len(class_weights) == len(class_weights_target)

    for a, b in zip(class_weights_target, class_weights):
        assert np.allclose(a, b)


def test_get_subset_masks_and_weights():
    leaf_1 = np.array([[True, True, False],
                       [False, True, False],
                       [False, False, False],
                       [False, False, False],
                       [False, False, False]])

    leaf_2 = np.array([[False, True, False],
                       [True, True, False],
                       [False, False, False],
                       [False, False, False],
                       [False, False, False]])


    leaf_3 = np.array([[False, False, False],
                       [False, False, False],
                       [True, True, False],
                       [False, True, False],
                       [False, True, False]])


    leaf_4 = np.array([[False, False, False],
                       [False, False, False],
                       [False, True, False],
                       [True, True, False],
                       [False, True, False]])

    leaf_5 = np.array([[False, False, False],
                       [False, False, False],
                       [False, False, True],
                       [False, False, True],
                       [True, True, True]])

    add_weight_to_nodes(tree, total_weight=6)
    sm_array, sw_array = get_subset_masks_and_weights(tree)

    assert((sm_array[0] == leaf_1).all())
    assert((sm_array[1] == leaf_2).all())
    assert((sm_array[2] == leaf_3).all())
    assert((sm_array[3] == leaf_4).all())
    assert((sm_array[4] == leaf_5).all())

    assert(np.allclose(
        sw_array,
        np.array([
            [3., 3., 0],
            [3., 3., 0],
            [4., 2., 0],
            [4., 2., 0],
            [2., 2., 2.],
        ])
    ))

