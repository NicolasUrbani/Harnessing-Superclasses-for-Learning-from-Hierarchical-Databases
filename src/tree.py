import math
import numpy as np
from anytree import Node, PostOrderIter
from anytree.importer import JsonImporter


def get_nodes(tree, include_leaves=True, include_root=True):
    """Return list of nodes of `tree` starting with leaves.

    Nodes order is leaf first and then backtracking along ancestors in
    topological order.

    """

    # Get nodes leaves first
    nodes = list(tree.leaves) + list(PostOrderIter(tree, filter_=lambda n: not n.is_leaf))

    # Select according to `include_root` and `include_leaves`
    nodes = [n for n in nodes if include_root or not n.is_root]
    nodes = [n for n in nodes if include_leaves or not n.is_leaf]

    return nodes


def get_subset_masks_and_weights(tree):
    """Return boolean array encoding subsets of each leaf and weights of these subsets.

    Returned array is of size `n_leaves` * `n_leaves` * `depth`. For each leaf
    (first dimension), a subset of leaves (second dimension) is encoded as a
    mask. The third dimension is the maximum number of subsets per leaf.

    Weight array is of size `n_leaves` * `depth`.

    """

    am = get_ancestor_matrix(tree, include_root=False)
    nodes = get_nodes(tree, include_root=False)
    weights = np.array([n.weight for n in nodes])

    # Get maximum of subsets per leaf
    depth = tree.height
    n_leaves = len(tree.leaves)

    sm_array = np.full((n_leaves, n_leaves, depth), False)
    sw_array = np.zeros((n_leaves, depth))

    for i, leaf in enumerate(tree.leaves):
        # Select all ancestors of `leaf`, `leaf` included, root excluded
        mask = am[i] == 1

        # Select corresponding leaves for each ancestors and associated weights
        subsets = am[:, mask]
        weights_i = weights[mask]

        n_subsets = len(weights_i)
        sm_array[i, :, :n_subsets] = (subsets == 1)
        sw_array[i, :n_subsets] = weights_i

    return sm_array, sw_array


def get_ancestor_matrix(tree, include_root=True, include_leaves=True):
    """Return the ancestor matrix of `tree`.

    Return a `n_leaves` * `n_nodes` indicator matrix that is 1 if the leaf is a
    descendant of the node and 0 otherwise. Both root and leaves are included in
    the list of nodes by default. Nodes order are that of `get_nodes`.

    """

    # List nodes leaves first
    nodes = get_nodes(tree)

    leaf2index = {l: i for i, l in enumerate(tree.leaves)}
    n_leaves = len(leaf2index)
    vs = []

    for node in nodes:
        if node.is_root and not include_root:
            continue

        if node.is_leaf and not include_leaves:
            continue

        idxs = [leaf2index[l] for l in node.leaves]

        v = np.zeros(n_leaves, dtype=int)
        v[idxs] = 1
        vs.append(v)

    return np.stack(vs, axis=1)


def add_weight_to_nodes(tree, total_weight=1, method="balanced", reduce="reset", **kwargs):
    """Add a weight on each non-root node.

    The weights sum to `total_weight` for each path from root to leaf. The
    different methods can be accumulated with `reduce` set to "add".

    Different methods are implemented:

    - `balanced`: `total_weight` / height of `tree` is recalculated for each
      node. This gives an approximately uniform distribution of weights.

    - `lazy`: nodes are given the same weight (`total_weight` / height of `tree`)
      except leaves that receives the rest to sum to `total_weight`.

    - `greedy`: give node a weight such that all descendants will receive the
      same weight (`total_weight` / height of `tree`).

    - `power`: weights follow as much as possible a power decay law of parameter
      `decay`.

    - `leaf_proportion`: each node receives the proportion of leaves of parent
      minus proportion of leaves of current node.

    """

    if reduce == "reset":
        for node in PostOrderIter(tree):
            node.weight = 0
    elif reduce == "add":
        for node in PostOrderIter(tree):
            if not hasattr(node, "weight"):
                node.weight = 0
    else:
        raise Exception("Unknown reduce:", reduce)


    # Set get_weight based on method
    if method == "balanced":
        def get_weight(node, total_height, current_weight, total_weight):
            return current_weight / (node.height + 1)
    elif method == "lazy":
        def get_weight(node, total_height, current_weight, total_weight):
            if node.is_leaf:
                return total_weight / total_height * (total_height - node.depth + 1)
            else:
                return total_weight / total_height
    elif method == "greedy":
        def get_weight(node, total_height, current_weight, total_weight):
            rest = round(current_weight / (total_weight / total_height))
            return total_weight / total_height * (rest - node.height)
    elif method == "power":
        q = kwargs["decay"]
        def get_weight(node, total_height, current_weight, total_weight):
            if q == 1:
                return current_weight / (node.height + 1)
            else:
                return abs(1 - q) * current_weight / abs(1 - q ** (node.height + 1))
    elif method == "n_leaves":
        def get_weight(node, total_height, current_weight, total_weight):
            if node.is_leaf:
                return 1
            elif len(node.children) == 1:
                return 0
            else:
                return 1 / len(node.leaves)
    elif method == "bertinetto":
        alpha = kwargs["alpha"]
        def get_weight(node, total_height, current_weight, total_weight):
            return math.exp(- alpha * node.depth)
    elif method == "leaf":
        height = kwargs["height"]
        def get_weight(node, total_height, current_weight, total_weight):
            if node.height > height:
                return 0
            else:
                return current_weight / (node.height + 1)
    elif method == "leaf_proportion":
        def get_weight(node, total_height, current_weight, total_weight):
            if node.is_leaf:
                return current_weight
            else:
                return current_weight - len(node.leaves) / len(node.root.leaves)
    elif method == "ones":
        def get_weight(node, total_height, current_weight, total_weight):
            return 1
    else:
        raise Exception("Unknown method:", method)

    def add_weight_to_nodes_aux(node, total_height, current_weight, total_weight, get_weight_fn):
        for child in node.children:
            child.weight += get_weight_fn(child, total_height, current_weight, total_weight)
            add_weight_to_nodes_aux(child, total_height, current_weight - child.weight, total_weight, get_weight_fn)

    add_weight_to_nodes_aux(tree, tree.height, total_weight, total_weight, get_weight)


def get_subsets_and_weights(tree, total_weight=1, method="balanced"):
    """Return subsets and weights for each class.

    Return two lists of length the number of leaves in `tree` (which is the
    number of classes). First list contains tensors of size `n_leaves` *
    (`n_ancestors` for current leaf) which is the ancestor matrix restricted to
    node that are ancestors of considered class (or leaf). Second list contains
    1-D array of weights for each ancestor (leaf included, root excluded) for
    current leaf (or class).

    """

    add_weight_to_nodes(tree, total_weight=total_weight, method=method)
    nodes = get_nodes(tree, include_leaves=True, include_root=False)
    am = get_ancestor_matrix(tree, include_leaves=True, include_root=False)
    node2idx = {n: i for i, n in enumerate(nodes)}
    node_weights = np.array([n.weight for n in nodes])

    class_indexes = []
    class_weights = []
    for leaf_idx, leaf in enumerate(tree.leaves):
        ancestors = list(leaf.ancestors[1:]) + [leaf]
        ancestors_idx = [node2idx[a] for a in ancestors]
        n_ancestors = len(ancestors)

        class_indexes.append(am[:, ancestors_idx])
        class_weights.append(node_weights[ancestors_idx])

    return class_indexes, class_weights


def get_U(tree, lambdas):
    """Return U matrix."""

    # List nodes leaves first
    nodes = get_nodes(tree)

    U = np.zeros((len(nodes), len(nodes)))

    # Iterate on edges parent->child
    for child in nodes:
        if child.is_root:
            continue

        idx_child = nodes.index(child)
        n_leaves_child = len(child.leaves)

        parent = child.parent
        idx_parent = nodes.index(parent)
        n_leaves_parent = len(parent.leaves)

        U[idx_child, idx_child] += lambdas[idx_parent]

        U[idx_parent, idx_parent] += (
            lambdas[idx_parent] * n_leaves_child ** 2 / n_leaves_parent ** 2
        )

        value = -lambdas[idx_parent] * n_leaves_child / n_leaves_parent
        U[idx_parent, idx_child] += value
        U[idx_child, idx_parent] += value

    return U


def load_tree_from_file(filename):
    if filename.endswith(".txt"):
        return get_tree_from_file(filename)
    else:
        importer = JsonImporter()
        return importer.read(open(filename))
    

def get_tree_from_file(file_tree):

    file = open(file_tree, "r")
    all_lines = file.readlines()
    all_lines = [line[:-1] for line in all_lines]
    root = Node("root")

    R = root
    L_before = 0

    for line in all_lines[1:]:
        # count the number of "-" at the beginning
        L_actual = count_symbol(line)

        # if it's bigger, mean we are going into a next level
        if L_actual > L_before:
            # update the value of L
            L_before = L_actual
            # create a node with the parent
            node = Node(f"{line[L_actual:]}", parent=R)
            # update the parent
            R = node
        # if it's in the same level
        elif L_actual == L_before:
            # i need the parent, I am not coing deeper in the hierarchy
            R = node.parent
            node = Node(f"{line[L_actual:]}", parent=R)
            # update the node
            R = node
        else:
            # if we have to return back, firstly assign the parent as root
            R = node.parent
            # then we add one parent for each couple o "-"
            for i in range(((L_before - L_actual) // 2)):
                R = R.parent
            node = Node(f"{line[L_actual:]}", parent=R)
            # update the node
            R = node
            L_before = L_actual

    # print(RenderTree(root))
    # print("-" * 100)

    return root

def count_symbol(line):
    L = 0
    for i in line:
        if i == "-":
            L += 1
        else:
            break
    return L

if __name__ == '__main__':
    f = Node("f")
    b = Node("b", parent=f)
    a = Node("a", parent=b)
    d = Node("d", parent=b)
    c = Node("c", parent=d)
    e = Node("e", parent=d)
    g = Node("g", parent=f)
    i = Node("i", parent=g)
    h = Node("h", parent=i)

    U = get_U(tree, lambdas)
    L = get_ancestor_matrix(tree)
    C = torch.tensor(L @ U @ L.T)

    loss = torch.sum(C * (B.T @ B))
