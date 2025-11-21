library(TedSim)
library(dplyr)
library(reshape2)
library(ctc)
library(ape)
library(dichromat)
library(scales)

p_a_list <- c(0.2, 0.4, 0.6, 0.8)

ncells <- 4096  # 2^12
phyla <- read.tree(text='((t1:2, t2:2):1, (t3:2, t4:2):1):2;')
N_nodes <- 2 * ncells - 2
ngenes <- 500
max_walk <- 6
n_cif <- 30
n_diff <- 20
cif_step <- 1
p_d <- 0
mu <- 0.1
N_char <- 32
unif_on <- FALSE


generate <- function(p_a) {
    modifier <- paste("", p_a, cif_step, sep = "_")
    set.seed(0)

    output_dir <- paste0("D:/data/DeepTracing/experiments/TedSim/data/", p_a)
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    returnlist <- SIFGenerate(phyla, n_diff, step = cif_step)

    cifs <- SimulateCIFs(ncells, phyla,
                         p_a = p_a,
                         n_CIF = n_cif,
                         n_diff = n_diff,
                         step = cif_step,
                         p_d = p_d,
                         mu = mu,
                         Sigma = 0.5,
                         N_char = N_char,
                         max_walk = max_walk,
                         SIF_res = returnlist,
                         unif_on = unif_on)

    cif_leaves <- lapply(c(1:3), function(parami) {
        cifs[[1]][[parami]][c(1:ncells), ]
    })
    cif_res <- list(cif_leaves, cifs[[2]])


    states <- cifs[[2]]
    states <- states[1:N_nodes, ]
    muts <- cifs[[7]]
    rownames(muts) <- paste("cell", states[, 4], sep = "_")


    true_counts_res <- CIF2Truecounts(ngenes = ngenes,
                                      ncif = n_cif,
                                      ge_prob = 0.3,
                                      ncells = N_nodes,
                                      cif_res = cifs)

    data(gene_len_pool)
    gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
    observed_counts <- True2ObservedCounts(true_counts = true_counts_res[[1]],
                                           meta_cell = true_counts_res[[3]],
                                           protocol = "UMI",
                                           alpha_mean = 0.2,
                                           alpha_sd = 0.05,
                                           gene_len = gene_len,
                                           depth_mean = 1e5,
                                           depth_sd = 3e3)


    gene_expression_dir <- paste0(output_dir, "/counts_tedsim", modifier, ".csv")
    cell_meta_dir <- paste0(output_dir, "/cell_meta_tedsim", modifier, ".csv")
    character_matrix_dir <- paste0(output_dir, "/character_matrix", modifier, ".txt")
    tree_gt_dir <- paste0(output_dir, "/tree_gt_bin_tedsim", modifier, ".newick")
    edges_dir <- paste0(output_dir, "/edges", modifier, ".csv")
    states_full_dir <- paste0(output_dir, "/states_full", modifier, ".csv")


    cat(">>> The file will be saved to the following location:\n")
    cat(gene_expression_dir, "\n", cell_meta_dir, "\n", character_matrix_dir, "\n", tree_gt_dir, "\n")

    write.tree(cifs[[4]], tree_gt_dir)
    write.csv(observed_counts[[1]], gene_expression_dir, row.names = FALSE)
    write.csv(states, cell_meta_dir)
    write.table(muts, character_matrix_dir)
    write.csv(cifs[[4]]$edge, edges_dir)
    write.csv(states, states_full_dir)
}

for (p_a in p_a_list) {
    generate(p_a)
}


