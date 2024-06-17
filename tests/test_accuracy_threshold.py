import numpy as np
import pytest
import torch
from anytree import Node
from src.metrics import AccuracyThreshold, get_thresholds
from src.tree import add_weight_to_nodes


def params(d):
    return pytest.mark.parametrize(
        argnames=(argnames := sorted({k for v in d.values() for k in v.keys()})),
        argvalues=[[v.get(k) for k in argnames] for v in d.values()],
        ids=d.keys(),
    )


tree1 = Node("v_11",
            children=[
                Node("v_1"),
                Node("v_8", children=[
                    Node("v_2"),
                    Node("v_3"),
                    Node("v_4")
                ]),
                Node("v_10", children=[
                    Node("v_5"),
                    Node("v_9", children=[
                        Node("v_6"),
                        Node("v_7")
                    ])
                ])
            ])


def sum_to_1(t):
    return t / torch.sum(t, dim=1, keepdim=True)


@params(dict(
    setup1=dict(
        tree=tree1,
        result=np.array([0.16666667, 0.41666667, 0.58333333, 0.83333333])
    )
))
def test_get_thresholds(tree, result):
    add_weight_to_nodes(tree)
    thresholds = get_thresholds(tree)
    assert np.allclose(thresholds, result)



@params(dict(
    setup1=dict(
        tree=tree1,
        preds=sum_to_1(torch.tensor([[1, 5, 5, 5, 0, 6, 7]], dtype=torch.float)),
        target=torch.tensor([5]),
        thresholds=np.array([0.16, 0.41, 0.58, 0.83]),
        result={
            "acc_0.16": torch.tensor([0.]),
            "acc_0.41": torch.tensor([0.]),
            "acc_0.58": torch.tensor([1.]),
            "acc_0.83": torch.tensor([0.]),
        }
    ),
    setup2=dict(
        tree=tree1,
        preds=sum_to_1(torch.tensor([[1, 5, 5, 5, 0, 7, 6]], dtype=torch.float)),
        target=torch.tensor([5]),
        thresholds=np.array([0.16, 0.41, 0.58, 0.83]),
        result={
            "acc_0.16": torch.tensor([0.]),
            "acc_0.41": torch.tensor([0.]),
            "acc_0.58": torch.tensor([1.]),
            "acc_0.83": torch.tensor([1.]),
        }
    ),
    setup3=dict(
        tree=tree1,
        preds=sum_to_1(torch.tensor([[1, 5, 5, 5, 0, 7, 6]], dtype=torch.float)),
        target=torch.tensor([0]),
        thresholds=np.array([0.16, 0.41, 0.58, 0.83]),
        result={
            "acc_0.16": torch.tensor([0.]),
            "acc_0.41": torch.tensor([0.]),
            "acc_0.58": torch.tensor([0.]),
            "acc_0.83": torch.tensor([0.]),
        }
    ),
    setup4=dict(
        tree=tree1,
        preds=sum_to_1(torch.tensor([[1, 5, 5, 5, 0, 7, 6]], dtype=torch.float)),
        target=torch.tensor([1]),
        thresholds=np.array([0.16, 0.41, 0.58, 0.83]),
        result={
            "acc_0.16": torch.tensor([1.]),
            "acc_0.41": torch.tensor([1.]),
            "acc_0.58": torch.tensor([0.]),
            "acc_0.83": torch.tensor([0.]),
        }
    ),
    setup5=dict(
        tree=tree1,
        preds=sum_to_1(torch.tensor([[1, 5, 5, 5, 0, 6, 7],
                                     [1, 5, 5, 5, 0, 7, 6],
                                     [1, 5, 5, 5, 0, 7, 6],
                                     [1, 5, 5, 5, 0, 7, 6]], dtype=torch.float)),
        target=torch.tensor([5, 5, 0, 1]),
        thresholds=np.array([0.16, 0.41, 0.58, 0.83]),
        result={
            "acc_0.16": torch.tensor([.25]),
            "acc_0.41": torch.tensor([.25]),
            "acc_0.58": torch.tensor([0.5]),
            "acc_0.83": torch.tensor([0.25]),
        }
    )
))
def test_accuracy_threshold(tree, preds, target, thresholds, result):
    add_weight_to_nodes(tree)
    at = AccuracyThreshold(tree, thresholds=thresholds)
    at(preds, target)
    res = at.compute()

    assert res == result
