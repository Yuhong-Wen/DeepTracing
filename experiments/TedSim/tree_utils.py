import io
from copy import deepcopy
from typing import Any, Dict, List, Literal, Mapping, Optional
from sklearn.cluster import KMeans
import lineageot.inference as lot_inf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from Bio import Phylo
from TedSim_data_prepare import newick2digraph, annotate, is_valid_edge


node_colors = [
    "#bd6a47",
    "#bd6a47",
    "#779538",
    "#dbd4c0",
    "#b18393",
    "#8a6a4b",
    "#3e70ab",
    "#9d3d58",
    "#525566",
    "#3f5346",
    "#dbd4c0",
    "#4E79A7",  # muted blue
    "#F28E2B",  # warm orange
    "#59A14F",  # soft green
    "#76B7B2",  # teal/cyan
    "#EDC948",  # muted yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # soft pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]
custom_palette = [
    "#4E79A7",  # muted blue
    "#F28E2B",  # warm orange
    "#59A14F",  # soft green
    "#EDC948",  # muted yellow
    "#76B7B2",  # teal/cyan
    "#B07AA1",  # purple
    "#FF9DA7",  # soft pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]

custom_colors_tree = [
    "#4E79A7",  # muted blue
    "#F28E2B",  # warm orange
    "#59A14F",  # soft green
    "#76B7B2",  # teal/cyan
]

vintage_palette = [
    "#bd6a47", "#779538", "#dbd4c0", "#b18393", "#8a6a4b",
    "#3e70ab", "#9d3d58", "#525566", "#3f5346", "#5b9279",
    "#a26769", "#c4a287", "#7c8577", "#a3c9a8", "#d0b49f",
    "#845a6d", "#a19782", "#6689a1", "#c2948a", "#796465"
]



def _process_data(
    rna_arrays: Mapping[Literal["early", "late"], np.ndim], *, n_pcs: int = 30, pca=True
) -> np.ndarray:
    adata = AnnData(rna_arrays["early"], dtype=np.float32).concatenate(
        AnnData(rna_arrays["late"], dtype=np.float32),
        batch_key="time",
        batch_categories=["0", "1"],
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    if pca:
        sc.tl.pca(adata, use_highly_variable=False)
        return adata.obsm["X_pca"][:, :n_pcs].astype(float, copy=True)
    return adata.X.astype(float, copy=True)


def _annotate(
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
        dummy_cell = lot_inf.sim.Cell([], [])
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


def _cut_at_depth(G: nx.DiGraph, *, max_depth: Optional[int] = None) -> nx.DiGraph:
    if max_depth is None:
        return deepcopy(G)
    selected_nodes = [n for n in G.nodes if G.nodes[n]["node_depth"] <= max_depth]
    G = deepcopy(G.subgraph(selected_nodes).copy())

    # relabel because of LOT
    leaves = sorted(n for n in G.nodes if not list(G.successors(n)))
    for new_name, n in enumerate(leaves):
        G = nx.relabel_nodes(G, {n: new_name}, copy=False)
    return G


def _build_true_trees(
    rna: np.ndarray,
    barcodes: np.ndarray,
    meta: pd.DataFrame,
    *,
    tree: str,
    depth: int,
    n_pcs: int = 30,
    pca: bool = True,
    ttp: float = 100.0,
) -> Dict[Literal["early", "late"], nx.DiGraph]:
    cell_arr_adata = [
        lot_inf.sim.Cell(rna[nid], barcodes[nid]) for nid in range(rna.shape[0])
    ]
    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = tree

    trees = {"early": _cut_at_depth(G, max_depth=depth), "late": _cut_at_depth(G)}
    rna_arrays = {
        kind: np.asarray(
            [
                trees[kind].nodes[n]["cell"].x
                for n in trees[kind].nodes
                if _is_leaf(trees[kind], n)
            ]
        )
        for kind in ["early", "late"]
    }
    data = _process_data(rna_arrays, n_pcs=n_pcs, pca=pca)

    n_early_leaves = len([n for n in trees["early"] if _is_leaf(trees["early"], n)])
    data_early, data_late = data[:n_early_leaves], data[n_early_leaves:]

    for kind, data in zip(["early", "late"], [data_early, data_late]):
        i, G = 0, trees[kind]
        for n in G.nodes:
            if _is_leaf(G, n):
                G.nodes[n]["cell"].x = data[i]
                i += 1
            else:
                G.nodes[n]["cell"].x = np.full((n_pcs,), np.nan)

    return trees



def _build_true_tree(
        rna: np.ndarray,
        barcodes: np.ndarray,
        meta: pd.DataFrame,
        *,
        tree: str,
        depth: int,
        max_depth: Optional[int] = None,
        n_pcs: int = 30,
        pca: bool = True,
        lognorm: bool = True,
        ttp: float = 100.0):
    print(f"rna.shape[0]: {rna.shape[0]}")
    print(f"Type of rna.shape[0]: {type(rna.shape[0])}")
    # cell_arr_adata = [rna[nid] for nid in range(rna.shape[0])]
    seed = np.random.randint(2 ** 31)
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


def _build_true_trees_draw(
    rna: np.ndarray,
    barcodes: np.ndarray,
    meta: pd.DataFrame,
    *,
    tree: str,
    depth: int,
    n_pcs: int = 30,
    pca: bool = True,
    ttp: float = 100.0,
) -> Dict[Literal["early", "late"], nx.DiGraph]:
    cell_arr_adata = [
        lot_inf.sim.Cell(rna[nid], barcodes[nid]) for nid in range(rna.shape[0])
    ]
    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = _newick2digraph(tree)
    G = _annotate(G, cell_arr_adata, metadata, ttp=ttp)

    trees = {"early": _cut_at_depth(G, max_depth=depth), "late": _cut_at_depth(G)}
    rna_arrays = {
        kind: np.asarray(
            [
                trees[kind].nodes[n]["cell"].x
                for n in trees[kind].nodes
                if _is_leaf(trees[kind], n)
            ]
        )
        for kind in ["early", "late"]
    }
    data = _process_data(rna_arrays, n_pcs=n_pcs, pca=pca)

    n_early_leaves = len([n for n in trees["early"] if _is_leaf(trees["early"], n)])
    data_early, data_late = data[:n_early_leaves], data[n_early_leaves:]

    for kind, data in zip(["early", "late"], [data_early, data_late]):
        i, G = 0, trees[kind]
        for n in G.nodes:
            if _is_leaf(G, n):
                G.nodes[n]["cell"].x = data[i]
                i += 1
            else:
                G.nodes[n]["cell"].x = np.full((n_pcs,), np.nan)

    return trees


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

    node_list = [3, 4, 1, 2, 5, 7, 6]
    node_color = [node_colors[node_list[node]] for node in state_tree_.nodes]

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
        plt.savefig(
            path + "/cell_state_tree.png",
            bbox_inches="tight",
            transparent=True,
            dpi=300,
        )
    plt.show()


def subtree_clusters_tree(tree, depth, obs_feature):
    split_depth = depth
    tree = tree.copy()
    for node_id in tree.nodes:
        current_node = node_id
        while True:
            if current_node == 'root':
                tree.nodes[node_id][obs_feature] = "unassigned"
                break
            try:
                current_depth = tree.nodes[int(current_node)]["node_depth"]
            except (KeyError, ValueError):
                tree.nodes[node_id][obs_feature] = "unassigned"
                break
            if current_depth == split_depth:
                tree.nodes[node_id][obs_feature] = current_node
                break
            predecessors = list(tree.predecessors(int(current_node)))
            if not predecessors:
                tree.nodes[node_id][obs_feature] = "unassigned"
                break
            current_node = predecessors[0]
    return tree



def tree_draw_clustres(adata,tree, depth=8, path=None, color_by='cluster',color_map=None):
    """Draw tree with Scanpy color scheme"""
    tree = tree.copy()
    rna = adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    true_trees = _build_true_trees(
        rna, barcodes, meta=adata.obs, tree=tree, depth=depth, pca=False
    )

    g_ = true_trees["early"]

    # 获取所有节点的cluster值
    clusters = [g_.nodes[node].get(color_by, "unknown") for node in g_.nodes]
    unique_clusters = sorted(set(clusters))



    if len(unique_clusters) <= 12:
        palette = custom_colors_tree[:len(unique_clusters)]
    elif len(unique_clusters) <= 20:
        palette = sns.color_palette("tab20", n_colors=len(unique_clusters))
    else:
        palette = sns.color_palette("husl", n_colors=len(unique_clusters))
    color_map = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

    cols = [color_map.get(g_.nodes[node].get(color_by, "unknown"), "#CCCCCC")
            for node in g_.nodes]

    pos = nx.drawing.nx_agraph.graphviz_layout(g_, prog="dot")

    fig, axs = plt.subplots(figsize=(8, 5))
    nx.draw(g_, pos, node_color=cols, node_size=50, arrowsize=10, arrows=True, ax=axs)
    axs.set_title(f"The early cell division tree (up to depth {depth})", fontsize=22)

    legend_handles = []
    for cluster, color in color_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=str(cluster)))

    axs.legend(handles=legend_handles, loc='center left',
               bbox_to_anchor=(1.05, 0.5), frameon=False)

    plt.tight_layout()
    if path is not None:
        # plt.savefig(path + "/tree_colored_by_cluster.png", bbox_inches="tight", transparent=True, dpi=300)
        plt.savefig(path + "/tree_colored_by_cluster.pdf", bbox_inches="tight", transparent=True)
    plt.show()



def tree_draw_subtrees(adata, tree,  depth=8, path=None, color_by='subtree'):
    """Draw tree with Scanpy color scheme"""
    tree = tree.copy()
    rna = adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    true_trees = _build_true_trees(
        rna, barcodes, meta=adata.obs, tree=tree, depth=depth, pca=False
    )

    g_ = true_trees["early"]

    g_ = subtree_clusters_tree(g_, 4, color_by)


    clusters = [str(g_.nodes[node].get(color_by, "unassigned")) for node in g_.nodes]
    assigned_clusters = sorted(set(c for c in clusters if c != "unassigned"))

    if len(assigned_clusters) <= 12:
        palette = custom_palette[:len(assigned_clusters)]
    elif len(assigned_clusters) <= 20:
        # palette = vintage_palette[:len(assigned_clusters)]
        palette = sns.color_palette("tab20", n_colors=len(assigned_clusters))
    else:
        palette = sns.color_palette("husl", n_colors=len(assigned_clusters))

    color_map = {cluster: palette[i] for i, cluster in enumerate(assigned_clusters)}
    color_map["unassigned"] = "#000000"

    cols = [color_map.get(str(g_.nodes[node].get(color_by, "unassigned")), "#000000")
            for node in g_.nodes]

    pos = nx.drawing.nx_agraph.graphviz_layout(g_, prog="dot")

    fig, axs = plt.subplots(figsize=(8, 5))
    nx.draw(g_, pos, node_color=cols, node_size=50, arrowsize=10, arrows=True, ax=axs)
    axs.set_title(f"The early cell division tree (up to depth {depth})", fontsize=22)

    legend_handles = []
    for cluster, color in color_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=str(cluster)))

    axs.legend(handles=legend_handles,
               loc='upper left',
               bbox_to_anchor=(1.01, 1),
               frameon=False,
               fontsize=8,
               ncol=2,
               borderaxespad=0.2,
               handletextpad=0.3,
               columnspacing=0.8,
               handlelength=1.2)

    plt.tight_layout()
    if path is not None:
        # plt.savefig(path + "/tree_colored_by_subtree.png", bbox_inches="tight", transparent=True, dpi=300)
        plt.savefig(path + "/tree_colored_by_subtree.pdf", bbox_inches="tight", transparent=True)
    plt.show()

    return color_map



def plot_cost(lp, depth_early=8, depth_late=12):
    """Plot cost matrix"""
    for problem in lp.problems:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sns.heatmap(
            lp.problems[problem].x.data,
            ax=axs[0],
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axs[0].set_title(
            f"Barcode distances\n(cells at depth {depth_early})", fontsize=16
        )
        axs[0].set_xlabel("cells", fontsize=14)
        axs[0].set_ylabel("cells", fontsize=14)

        sns.heatmap(
            lp.problems[problem].y.data,
            ax=axs[1],
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axs[1].set_title(
            f"Barcode distances\n(cells at depth {depth_late})", fontsize=16
        )
        axs[1].set_xlabel("cells", fontsize=14)
        axs[1].set_ylabel("cells", fontsize=14)

        plt.tight_layout()
        plt.show()



def subtree_clusters(adata, tree, depth, obs_feature):
    adata = adata.copy()
    split_depth = depth
    tree = tree.copy()
    subtree_clusters = []
    for node_id in adata.obs.index:
        current_node = node_id
        while True:
            if current_node == 'root':
                subtree_clusters.append("unassigned")
                break
            try:
                current_depth = tree.nodes[int(current_node)]["node_depth"]
            except (KeyError, ValueError):
                subtree_clusters.append("unassigned")
                break
            if current_depth == split_depth:
                subtree_clusters.append(current_node)
                break
            predecessors = list(tree.predecessors(int(current_node)))
            if not predecessors:
                subtree_clusters.append("unassigned")
                break
            current_node = predecessors[0]
    adata.obs[obs_feature] = pd.Categorical(subtree_clusters).astype(str)
    return adata
