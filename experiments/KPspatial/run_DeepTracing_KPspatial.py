import math, os
from time import time
import torch
from DeepTracing import DEEPTRACING
import numpy as np
import scanpy as sc
from preprocess import normalize, inducing_points_select_from_dist
import pickle
import anndata as ad
from pathlib import Path
from sklearn.cluster import KMeans

torch.manual_seed(0)

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Lineage dependency-aware variational autoencoder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='./data')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=5, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--GP_S_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=5, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool,
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta_gaussian', default=10, type=float, help='initial coefficient of the Gaussian KL loss')
    parser.add_argument('--min_beta_gaussian', default=4, type=float, help='minimal coefficient of the Gaussian KL loss')
    parser.add_argument('--max_beta_gaussian', default=25, type=float, help='maximal coefficient of the Gaussian KL loss')
    parser.add_argument('--init_beta_gp', default=10, type=float, help='initial coefficient of the GP KL loss')
    parser.add_argument('--min_beta_gp', default=4, type=float, help='minimal coefficient of the GP KL loss')
    parser.add_argument('--max_beta_gp', default=25, type=float, help='maximal coefficient of the GP KL loss')
    parser.add_argument('--init_beta_gp_S', default=10, type=float, help='initial coefficient of the GP KL loss')
    parser.add_argument('--min_beta_gp_S', default=4, type=float, help='minimal coefficient of the GP KL loss')
    parser.add_argument('--max_beta_gp_S', default=25, type=float, help='maximal coefficient of the GP KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool,
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale_lineage', default=20., type=float)
    parser.add_argument('--kernel_scale_space', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--gp_latent_file', default='gp_latent.txt')
    parser.add_argument('--gp_S_latent_file', default='gp_S_latent.txt')
    parser.add_argument('--gaussian_latent_file', default='gaussian_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--device', default='cuda:2')

    args = parser.parse_args()

    import os

    print("Actual file:", os.path.realpath(__file__))
    print("Working directory:", os.getcwd())

    adata = sc.read_h5ad(Path(args.data_file)  / "Puck1_Layer1_lineage_spatial.h5ad")

    if hasattr(adata.X, "toarray"):
        dense_array = adata.X.toarray()
    else:
        dense_array = np.asarray(adata.X)
    adata.X = dense_array

    adata.X = adata.X.astype(np.float64)
    print(adata)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000, subset=True)
    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    unique_lineage_distance = np.load(Path(args.data_file) / "lineage_distance.npy", mmap_mode="r")
    lineage_indices = adata.obs["lineage_index"].to_numpy().astype(int)
    unique_lineage_distance = np.asarray(unique_lineage_distance, dtype=np.float64)

    MIN_CELLS_PER_BRANCH = 10
    branch_counts = adata.obs["root_branch"].value_counts()

    valid_branches = branch_counts[
        branch_counts >= MIN_CELLS_PER_BRANCH
        ].index

    candidate_mask = (
        adata.obs["root_branch"]
        .isin(valid_branches)
        .to_numpy()
    )

    inducing_points_lineage = inducing_points_select_from_dist(
        select_feature=adata.obs["root_branch"].to_numpy()[candidate_mask],
        all_indices=lineage_indices[candidate_mask],
        full_dist_matrix=unique_lineage_distance,
        MIN_POINTS_PER_CLUSTER=2,
        CELLS_PER_ADDITIONAL_POINT=200,
    )

    initial_inducing_points_lineage = np.asarray(
        inducing_points_lineage,
        dtype=int
    )


    assert np.all(
        np.isin(initial_inducing_points_lineage, lineage_indices[candidate_mask])
    )

    assert len(initial_inducing_points_lineage) == len(
        np.unique(initial_inducing_points_lineage)
    )

    print("Training cells:", adata.n_obs)
    print("Valid root branches:", len(valid_branches))
    print("Candidate cells:", candidate_mask.sum())
    print("Inducing points:", len(initial_inducing_points_lineage))

    print(initial_inducing_points_lineage)


    spatial_coordinates = (adata.obsm["spatial"].astype(np.float64))
    loc = np.array(adata.obsm["spatial"]).astype('float64')

    loc_min = loc.min(axis=0)
    loc_max = loc.max(axis=0)
    loc = (loc - loc_min) / (loc_max - loc_min) * args.loc_range

    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = (np.mgrid[0:(1 + eps):(1. / args.inducing_point_steps), 0:(1 + eps):(
                1. / args.inducing_point_steps)].reshape(2, -1).T
                                   * args.loc_range)
        # * loc_range)
        print(initial_inducing_points.shape)
    else:
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
        np.savetxt("location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt("location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    if args.batch_size == "auto":
        if adata.X.shape[0] <= 1024:
            args.batch_size = 128
        elif adata.X.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)
    print(args)

    model = DEEPTRACING(input_dim=adata.n_vars, GP_dim=args.GP_dim, GP_S_dim=args.GP_S_dim, Normal_dim=args.Normal_dim,
                        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
                        distance_lineage=unique_lineage_distance,
                        initial_inducing_points_lineage=initial_inducing_points_lineage,
                        initial_inducing_points_space=initial_inducing_points,
                        fixed_inducing_points=args.fix_inducing_points,
                        fixed_gp_params=args.fixed_gp_params, kernel_scale_lineage=args.kernel_scale_lineage,
                        kernel_scale_space=args.kernel_scale_space, N_train=adata.n_obs, KL_loss=args.KL_loss,
                        dynamicVAE=args.dynamicVAE,
                        init_beta_gaussian=args.init_beta_gaussian, min_beta_gaussian=args.min_beta_gaussian,
                        max_beta_gaussian=args.max_beta_gaussian,
                        init_beta_gp=args.init_beta_gp, min_beta_gp=args.min_beta_gp, max_beta_gp=args.max_beta_gp,
                        init_beta_gp_S=args.init_beta_gp_S, min_beta_gp_S=args.min_beta_gp_S,
                        max_beta_gp_S=args.max_beta_gp_S,
                        dtype=torch.float64, device=args.device)

    print(str(model))

    if not os.path.isfile(args.model_file):
        t0 = time()
        model.train_model(indices_lineage=lineage_indices, pos=loc, ncounts=adata.X,
                          raw_counts=adata.raw.X, size_factors=adata.obs["size_factors"].to_numpy(),
                          lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                          num_samples=args.num_samples,
                          train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True,
                          model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        model.load_model(args.model_file)

    # model.curve_loss()

    final_latent, gp_latent, gp_S_latent, gaussian_latent = model.batching_latent_samples(
        X_b=lineage_indices, X_s=loc, Y=adata.X, batch_size=adata.X.shape[0])
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.gp_latent_file, gp_latent, delimiter=",")
    np.savetxt(args.gp_S_latent_file, gp_S_latent, delimiter=",")
    np.savetxt(args.gaussian_latent_file, gaussian_latent, delimiter=",")