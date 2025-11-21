import numpy as np
import pandas as pd
import lineageot.inference as lot_inf
from pathlib import Path
from typing import Any, Dict, List
import networkx as nx
from scipy.sparse import issparse
import torch
from Bio import Phylo
import io
import anndata


def prepare_data(fpath: Path,
        *,
        ttp: float = 100.0,
):
    import scanpy as sc
    adata = sc.read(fpath)
    tree = adata.uns["tree"]
    rna = adata.X.A.copy() if issparse(adata.X) else adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    true_tree = build_true_trees(
        rna,
        barcodes,
        meta=adata.obs,
        tree=tree,
        ttp=ttp
    )

    return true_tree


def build_true_trees(
        rna: np.ndarray,
        barcodes: np.ndarray,
        meta: pd.DataFrame,
        *,
        tree: str,
        ttp: float = 100.0):
    print(f"rna.shape[0]: {rna.shape[0]}")
    print(f"Type of rna.shape[0]: {type(rna.shape[0])}")
    # cell_arr_adata = [rna[nid] for nid in range(rna.shape[0])]
    seed = np.random.randint(2**31)
    cell_arr_adata = [
    lot_inf.sim.Cell(rna[nid], barcodes[nid], seed) for nid in range(rna.shape[0])
]

    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = newick2digraph(tree)
    G = annotate(G, cell_arr_adata, metadata, ttp=ttp)
    for s, t in G.edges:
        sn, tn = G.nodes[s], G.nodes[t]
        assert is_valid_edge(sn, tn), (s, t)
    tree = G
    return tree


def is_leaf(G: nx.DiGraph, n: Any) -> bool:
    return not list(nx.descendants(G, n))



def is_valid_edge(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    r"""Assumes the following state tree:

       /-4
      7
     / \-3
    5
     \ /-1
      6
       \-2
    """
    state_tree = nx.from_edgelist([(5, 6), (5, 7), (6, 1), (6, 2), (7, 3), (7, 4)])
    try:
        # parent, cluster, depth
        p1, c1, d1 = n1["parent"], n1["cluster"], n1["depth"]
        p2, c2, d2 = n2["parent"], n2["cluster"], n2["depth"]
    except KeyError:
        # no metadata, assume true
        return True

    # root, anything is permitted
    if (p1, c1, d1) == (5, 6, 0):
        return True

    # sanity checks
    assert p1 in [5, 6, 7], p1
    assert p2 in [5, 6, 7], p2
    assert c1 in [1, 2, 3, 4, 6, 7], c1
    assert c2 in [1, 2, 3, 4, 6, 7], c2

    if p1 == p2:
        if c1 == c2:
            # check if depth of a parent is <=
            return d1 <= d2
        # sanity check that clusters are valid siblings
        return (c1, c2) in state_tree.edges

    # parent-cluster relationship
    assert c1 == p2, (c1, p2)
    # valid transition
    assert (c1, c2) in state_tree.edges, (c1, c2)
    return True


def newick2digraph(tree: str) -> nx.DiGraph:
    def trav(clade, prev: Any, depth: int) -> None:
        nonlocal cnt
        if depth == 0:
            name = "root"
        else:
            name = clade.name
            if name is None:
                name = cnt
                cnt -= 1
            else:
                name = int(name[1:]) - 1

        G.add_node(name, node_depth=depth)
        if prev is not None:
            G.add_edge(prev, name)

        for c in clade.clades:
            trav(c, name, depth + 1)

    G = nx.DiGraph()
    cnt = -1
    tree = Phylo.read(io.StringIO(tree), "newick")
    trav(tree.clade, None, 0)

    start = max([n for n in G.nodes if n != "root"]) + 1
    for n in list(nx.dfs_preorder_nodes(G)):
        if n == "root":
            pass
        if is_leaf(G, n):
            continue

        assert start not in G.nodes
        G = nx.relabel_nodes(G, {n: start}, copy=False)
        start += 1

    return G


def annotate(
        G: nx.DiGraph,
        cell_arr_data: List[lot_inf.sim.Cell],
        meta: List[Dict[str, Any]],
        ttp: int = 100,
) -> nx.DiGraph:
    G = G.copy()
    n_leaves = len([n for n in G.nodes if not len(list(G.successors(n)))])
    assert (n_leaves & (n_leaves - 1)) == 0, f"{n_leaves} is not power of 2"
    max_depth = int(np.log2(n_leaves))

    n_expected_nodes = 2 ** (max_depth + 1) - 1
    assert len(G) == n_expected_nodes, "graph is not a full binary tree"

    if len(cell_arr_data) != n_expected_nodes:  # missing root, add after observed nodes
        seed = np.random.randint(2**31)
        dummy_cell = lot_inf.sim.Cell([], [], seed)
        cell_arr_data += [dummy_cell]
        meta += [{}]

    for nid in G.nodes:
        depth = G.nodes[nid]["node_depth"]
        metadata = {
            **meta[nid],  # contains `depth`, which is different from `node_depth`
            "cell": cell_arr_data[nid],
            "nid": nid,
            "time": depth * ttp,
            "time_to_parent": ttp,
        }
        G.nodes[nid].update(metadata)

    for eid in G.edges:
        G.edges[eid].update({"time": ttp})
    return nx.relabel_nodes(G, {n_leaves: "root"}, copy=False)


def truncate_tree(G: nx.DiGraph, depth: int):
    """
    Processing the tree 'G' returned by the 'annotate' function:
    Truncate the tree at 'depth' and extract all nodes at or above that depth (retaining the original node ID).
    Return all information of each node, including gene expression, barcode, time, etc.

    : paramG: annotated nx. DiGraph cell evolutionary tree
    : param depth: Truncated depth, extract this depth and all nodes before it
    Return: (Induction point information, overall dataset information)
    """
    all_nodes = {n: G.nodes[n] for n in G.nodes if G.nodes[n]["node_depth"] <= depth}

    all_nodes_filtered = {n: data for n, data in all_nodes.items() if
                          not isinstance(data["cell"].x, list) or len(data["cell"].x) > 0}

    return all_nodes_filtered


def create_adata_from_tree(all_data):
    """
    """
    def construct_adata(data):
        """ Convert dictionary data to AnnData object (retaining original cell ID, including cluster information) """
        cell_ids = list(data.keys())
        X = np.array([data[n]["cell"].x for n in cell_ids])
        obs = pd.DataFrame({
            "time": [data[n]["time"] for n in cell_ids],
            "node_depth": [data[n]["node_depth"] for n in cell_ids],
            "cluster": [data[n].get("cluster", -1) for n in cell_ids]
        }, index=cell_ids)

        obsm = {
            "barcodes": np.array([data[n]["cell"].barcode for n in cell_ids])
        }
        adata = anndata.AnnData(X=X, obs=obs, obsm=obsm)
        return adata

    adata_all = construct_adata(all_data)
    return adata_all


def scaled_Hamming_distance(barcode1, barcode2):
    """
    Computes the distance between two barcodes, adjusted for
    (1) the number of sites where both cells were measured and
    (2) distance between two scars is twice the distance from
        scarred to unscarred
    """
    shared_indices = (barcode1 >= 0) & (barcode2 >= 0)
    b1 = barcode1[shared_indices]

    # There may not be any sites where both were measured
    if len(b1) == 0:
        return np.nan
    b2 = barcode2[shared_indices]

    differences = b1 != b2
    double_scars = differences & (b1 != 0) & (b2 != 0)

    if isinstance(differences, torch.Tensor):
        differences = differences.cpu().numpy()
    if isinstance(double_scars, torch.Tensor):
        double_scars = double_scars.cpu().numpy()

    return (np.sum(differences) + np.sum(double_scars)) / len(b1)


def barcode_distances_between_arrays(barcode_array1, barcode_array2):
    """
    Calculate all pairwise distances between two barcode arrays and return the distance matrix
    """
    num_cells1 = barcode_array1.shape[0]
    num_cells2 = barcode_array2.shape[0]
    distance_matrix = np.zeros((num_cells1, num_cells2))
    for i in range(num_cells1):
        barcode1 = barcode_array1[i, :]
        for j in range(num_cells2):
            barcode2 = barcode_array2[j, :]
            distance_matrix[i, j] = scaled_Hamming_distance(barcode1, barcode2)
    return distance_matrix



