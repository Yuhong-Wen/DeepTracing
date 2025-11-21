import numpy as np

def calculate_tree_distance(tree, cell_id1, cell_id2):
    lca = tree.lowest_common_ancestor([cell_id1, cell_id2])

    dist1 = tree.find(cell_id1).distance(lca)

    dist2 = tree.find(cell_id2).distance(lca)

    return dist1 + dist2

def calculate_all_tree_distances(tree, cell_ids):
    n_cells = len(cell_ids)
    dist_matrix = np.zeros((n_cells, n_cells))

    for i in range(n_cells):
        for j in range(i+1, n_cells):
            dist = calculate_tree_distance(tree, cell_ids[i], cell_ids[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix
