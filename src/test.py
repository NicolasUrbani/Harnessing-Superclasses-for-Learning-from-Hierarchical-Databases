from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    func = globals()[cfg.entrypoint]
    func(cfg)


def tree_render(cfg: DictConfig):
    """Png export of tree and weights"""

    from anytree.exporter import DotExporter
    from anytree import Node
    from src.tree import add_weight_to_nodes

    tree = Node("root", children=[Node("sub0"), Node("sub1", children=[Node("sub0a"), Node("sub0b")])])

    tree = Node("v_11",
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


    tree = hydra.utils.instantiate(cfg.dataset.datasets.hierarchy)
    # add_weight_to_nodes(tree, method="n_leaves")
    add_weight_to_nodes(tree, method="power", decay=2)

    def edgeattrfunc(node, child):
        return 'label="%f"' % (child.weight)

    def nodenamefunc(node):
        # return '%s' % (node.name)
        return '%s\n%s' % (node.name, node.pos_offset)

    DotExporter(tree, nodenamefunc=nodenamefunc, edgeattrfunc=edgeattrfunc).to_picture("tree.png")


def image_size(cfg: DictConfig):
    """Inspect size of images"""

    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    for input, target in datamodule.train_dataloader():
        print(input.shape)


def image_augment(cfg: DictConfig):
    cfg.dataset.loaders.train.batch_size = 1
    cfg.dataset.loaders.train.shuffle = False

    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    for input, target in datamodule.train_dataloader():
        print(input)
        break


def tree_weights(cfg: DictConfig):
    """Visualizations of weights in a tree"""

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from anytree import Node
    from anytree.exporter import DotExporter

    from src.tree import add_weight_to_nodes

    tree = hydra.utils.instantiate(cfg.dataset.datasets.hierarchy)
    add_weight_to_nodes(tree, method="n_leaves")

    def data(tree):
        for leaf in tree.leaves:
            weight = sum(n.weight for n in list(leaf.ancestors) + [leaf])
            height = leaf.height
            depth = leaf.depth
            yield weight, depth


    df = pd.DataFrame(data(tree), columns=["weight", "depth"])


    sns.histplot(data=df, x="weight")
    plt.show()


def tree_weights_path(cfg: DictConfig):
    """Visualizations of weights in a tree"""

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from anytree import Node
    from anytree.exporter import DotExporter

    from src.tree import add_weight_to_nodes, get_nodes

    tree = hydra.utils.instantiate(cfg.dataset.datasets.hierarchy)
    add_weight_to_nodes(tree, method="power", decay=2)

    def data(tree):
        # for q in [1.2, 1, .8, 5]:
        for q in [1]:
            add_weight_to_nodes(tree, method="power", decay=q)
            for i, n in enumerate(get_nodes(tree)):
                dist = sum(nn.weight for nn in list(n.ancestors) + [n])
                yield dist, n.weight, str(q), i

    df = pd.DataFrame(data(tree), columns=["dist", "weight", "decay", "idx_node"])


    sns.scatterplot(data=df, x="dist", y="weight", hue="idx_node")
    plt.show()


def test(cfg: DictConfig):
    print("In test()")

if __name__ == "__main__":
    main()
