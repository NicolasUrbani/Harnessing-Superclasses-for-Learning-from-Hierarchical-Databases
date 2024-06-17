import json
import os

import hydra
import nltk
import bigtree
from nltk.corpus import wordnet as wn
from omegaconf import DictConfig


def gen_paths_from_pos_offset(filename):
    """Parse one-per-line collection of synsets ("n01632777", ...) and generate / separated paths.

    Generates lines like entity.n.01/physical_entity.n.01/.../true_lobster.n.01/american_lobster.n.02
    """

    def synset_name(synset):
        """Friendly name of synset"""
        return synset.lemmas()[0].name()

    with open(filename, "r") as f:
        for line in f:
            if not line:
                continue

            # Get Synset object from line
            pos, offset = line[0], int(line[1:9])
            synset = wn.synset_from_pos_and_offset(pos, offset)

            # Get ancestors
            hyper_list = [synset.name()]
            while synset.hypernyms():
                synset = synset.hypernyms()[0]
                hyper_list.append(synset.name())

            yield "/".join(hyper_list[::-1])


def create_nested_dict(map_fname) -> dict:
    """Return tree as dictionary from list of leaves in file `map_fname`."""

    # Get tree from a list of synsets
    paths = gen_paths_from_pos_offset(map_fname)
    bigtree_tree = bigtree.list_to_tree(paths)

    while len(bigtree_tree.children) == 1:
        bigtree_tree = bigtree_tree.children[0]

    # Move attr name to synset, add more friendly name
    for node in bigtree.preorder_iter(bigtree_tree):
        synset = wn.synset(node.name)
        node.synset = node.name
        node.name = synset.lemmas()[0].name()
        node.pos_offset = f"{synset.pos()}{synset.offset():08}"

    return bigtree.tree_to_nested_dict(bigtree_tree, all_attrs=True)


def shorten_tree(tree):
    """Shorten paths that are linear.

    Update weights along each path.
    """

    new_children = []
    for child in tree.children:
        weight = child.weight
        new_child = child
        while len(new_child.children) == 1:
            new_child = new_child.children[0]
            weight += new_child.weight
        new_children.append(new_child)
        new_child.weight = weight

    tree.children = new_children

    for child in tree.children:
        shorten_tree(child)


@hydra.main(config_path="../configs", version_base=None, config_name="tree.yaml")
def main(cfg: DictConfig):
    try:
        nltk.download("wordnet", download_dir=cfg.paths.hierarchy_dir, raise_on_error=True)
    except ValueError:
        pass

    map_fname = os.path.join(
        cfg.paths.hierarchy_dir, cfg.dataset.datasets.synset_filename
    )

    tree_dict = create_nested_dict(map_fname)

    tree_fname = os.path.join(
        cfg.paths.hierarchy_dir, cfg.dataset.datasets.hierarchy_filename
    )
    with open(tree_fname, "w") as f:
        json.dump(tree_dict, f, indent=2)


if __name__ == "__main__":
    main()
