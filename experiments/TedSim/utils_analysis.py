import io
from copy import deepcopy
from typing import Any, Dict, List, Literal, Mapping, Optional

import lineageot.inference as lot_inf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from Bio import Phylo

node_palette = [
    "#4E79A7",
    "#F28E2B",
    "#BAB0AC",
    "#9C755F",
    "#76B7B2",
    "#B07AA1",
    "#5CA8D6",
    "#EDC948",
     "#59A14F",
]


def _is_leaf(G: nx.DiGraph, n: Any) -> bool:
    return not list(nx.descendants(G, n))


def _newick2digraph(tree: str) -> nx.DiGraph:
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
        if _is_leaf(G, n):
            continue

        assert start not in G.nodes
        G = nx.relabel_nodes(G, {n: start}, copy=False)
        start += 1

    return G

def state_tree_draw(state_tree="((t1:2, t2:2):1, (t3:2, t4:2):1):2;", path=None):
    """Draw the state tree"""
    fig, axs = plt.subplots(figsize=(4, 4))
    state_tree_ = _newick2digraph(state_tree)
    pos = nx.drawing.nx_agraph.graphviz_layout(state_tree_, prog="dot")

    node_list = [2,3,0,1,5,6,4]
    node_color = [node_palette[node_list[node]] for node in state_tree_.nodes]

    nx.draw(
        state_tree_,
        pos,
        node_color=node_color,
        node_size=400,
        arrowsize=20,
        arrows=True,
        ax=axs,
    )
    axs.set_title("cell state tree", fontsize=22)
    plt.tight_layout()
    if path is not None:
        save_path = path / "cell_state_tree.pdf"
        plt.savefig(
            save_path,
            bbox_inches="tight",
            transparent=True,
            dpi=300,
        )
    plt.show()



