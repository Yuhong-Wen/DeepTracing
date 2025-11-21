import math, os
from time import time
import torch
import scanpy as sc
from preprocess import normalize, inducing_points_select_tumor
from DeepTracing import DEEPTRACING
from pathlib import Path
import numpy as np

torch.manual_seed(0)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='DeepTracing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='/data/DeepTracing/experiments/Tumor/data')
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
    parser.add_argument('--Normal_dim', default=5, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool,
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta_gaussian', default=5, type=float, help='initial coefficient of the gaussian KL loss')
    parser.add_argument('--min_beta_gaussian', default=1, type=float, help='minimal coefficient of the gaussian KL loss')
    parser.add_argument('--max_beta_gaussian', default=10, type=float, help='maximal coefficient of the gaussian KL loss')
    parser.add_argument('--init_beta_gp', default=10, type=float, help='initial coefficient of the gp KL loss')
    parser.add_argument('--min_beta_gp', default=4, type=float, help='minimal coefficient of the gp KL loss')
    parser.add_argument('--max_beta_gp', default=25, type=float, help='maximal coefficient of the gp KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--lambda_tc', default=10, type=float, help='coefficient of the TC loss')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--device', default='cuda:1')

    args = parser.parse_args()
    data_path=args.data_path
    name = '3724_NT_All'

    output_dir = Path(f"./results/tumor_{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    args.model_file = str(output_dir / f"tumor_{name}_model.pt")
    args.final_latent_file = str(output_dir / f"tumor_{name}_final_latent.txt")
    args.gp_latent_file = str(output_dir / f"tumor_{name}_gp_latent.txt")
    args.gaussian_latent_file = str(output_dir / f"tumor_{name}_gaussian_latent.txt")

    pruned_adata = sc.read_h5ad(f'{data_path}/preprocessed/{name}_filtered_pruned_adata.h5ad')
    print(pruned_adata)

    keep = np.array((pruned_adata.X > 0).sum(axis=0) >= 10).flatten()
    print('genes expressed in 10 or more cells:', np.sum(keep))

    pruned_adata = pruned_adata[:, keep].copy()
    if hasattr(pruned_adata.X, "toarray"):
        dense_array = pruned_adata.X.toarray()
    else:
        dense_array = np.asarray(pruned_adata.X)
    pruned_adata.X = dense_array

    sc.pp.highly_variable_genes(pruned_adata, flavor="seurat_v3", n_top_genes=200, subset=True)
    pruned_adata = normalize(pruned_adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    print(pruned_adata)

    dist_matrix = np.load(f'{data_path}/preprocessed/{name}_filtered_dist_matrix.npy')
    with open(f'{data_path}/preprocessed/{name}_filtered_cells.txt') as f:
        cell_names = np.array(f.read().splitlines())
    all_indices = np.arange(len(cell_names))
    all_indices = np.array(all_indices)


    inducing_points = inducing_points_select_tumor(adata=pruned_adata, distance_matrix=dist_matrix,
                                                   select_feature=pruned_adata.obs['subtree'],
                                                   all_indices=all_indices, CELLS_PER_ADDITIONAL_POINT=200)
    initial_inducing_points = np.array(inducing_points)
    print(initial_inducing_points)
    print(f"total number of inducing points: {len(initial_inducing_points)}")

    if args.batch_size == "auto":
        if pruned_adata.X.shape[0] <= 1024:
            args.batch_size = 128
        elif pruned_adata.X.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)
    print(args)

    model = DEEPTRACING(input_dim=pruned_adata.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        distance=dist_matrix, initial_inducing_points=initial_inducing_points, fixed_inducing_points=args.fix_inducing_points,
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=pruned_adata.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE,
        init_beta_gaussian=args.init_beta_gaussian, min_beta_gaussian=args.min_beta_gaussian, max_beta_gaussian=args.max_beta_gaussian, 
        init_beta_gp=args.init_beta_gp, min_beta_gp=args.min_beta_gp, max_beta_gp=args.max_beta_gp,
        dtype=torch.float64, device=args.device, lambda_tc = args.lambda_tc)

    print(str(model))

    if not os.path.isfile(args.model_file):
        t0 = time()
        model.train_model(indices=all_indices, ncounts=pruned_adata.X, raw_counts=pruned_adata.raw.X, size_factors=pruned_adata.obs.size_factors,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        model.load_model(args.model_file)

    # model.curve_loss()

    final_latent, gp_latent, gaussian_latent = model.batching_latent_samples(X=all_indices, Y=pruned_adata.X, batch_size=pruned_adata.X.shape[0])
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.gp_latent_file, gp_latent, delimiter=",")
    np.savetxt(args.gaussian_latent_file, gaussian_latent, delimiter=",")




