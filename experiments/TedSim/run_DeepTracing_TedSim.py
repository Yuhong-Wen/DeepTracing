import math, os
from time import time
from DeepTracing import DEEPTRACING
import scanpy as sc
from preprocess import normalize, inducing_points_select
from tree_utils import subtree_clusters
from TedSim_data_prepare import *
import pickle

running_seed = 0
torch.manual_seed(running_seed)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='DeepTracing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default="/data/DeepTracing/experiments/TedSim")
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
    parser.add_argument('--init_beta_gaussian', default=5, type=float, help='initial coefficient of the Gaussian KL loss')
    parser.add_argument('--min_beta_gaussian', default=1, type=float, help='minimal coefficient of the Gaussian KL loss')
    parser.add_argument('--max_beta_gaussian', default=10, type=float, help='maximal coefficient of the Gaussian KL loss')
    parser.add_argument('--init_beta_gp', default=10, type=float, help='initial coefficient of the GP KL loss')
    parser.add_argument('--min_beta_gp', default=4, type=float, help='minimal coefficient of the GP KL loss')
    parser.add_argument('--max_beta_gp', default=25, type=float, help='maximal coefficient of the GP KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--lambda_tc', default=10, type=float, help='coefficient of the TC loss')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--max_depth', default=12, type=int)
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--device', default='cuda')

    p_a_list = [0.2, 0.4, 0.6, 0.8]
    for p_a in p_a_list:
        output_dir = Path(f"./results/running_seed_{running_seed}/{p_a}")
        output_dir.mkdir(parents=True, exist_ok=True)

        args = parser.parse_args()

        args.model_file = str(output_dir / f"model_{p_a}.pt")
        args.final_latent_file = str(output_dir / f"final_latent_{p_a}.txt")
        args.gp_latent_file = str(output_dir / f"gp_latent_{p_a}.txt")
        args.gaussian_latent_file = str(output_dir / f"gaussian_latent_{p_a}.txt")

        BASE_DIR = Path(args.data_file)
        data_file = BASE_DIR / f"output_adatas/adata_{p_a}.h5ad"
        adata_raw = sc.read(data_file)
        print(adata_raw.X.shape)

        max_depth = args.max_depth
        tree = prepare_data(data_file)
        tree_file = output_dir / f"tree_{p_a}.pkl"
        with open(tree_file, "wb") as f:
            pickle.dump(tree, f)
        print(f"[p_a {p_a}] tree : {tree_file.resolve()}")
        with open(tree_file, "rb") as f:
            loaded_tree = pickle.load(f)
        tree = loaded_tree

        all_data = truncate_tree(tree, depth = max_depth)
        adata_all = create_adata_from_tree(all_data)

        mask = adata_all.obs['node_depth'].isin([max_depth])
        adata_need = adata_all[mask, :].copy()
        adata_need.X = np.array(adata_need.X) if isinstance(adata_need.X, pd.Series) else adata_need.X

        adata_need = subtree_clusters(adata_need, tree, 4, 'subtree')
        print("all dataset:", adata_need)

        # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000, subset=True)
        adata = normalize(adata_need,
                          size_factors=True,
                          normalize_input=True,
                          logtrans_input=True)


        barcodes_all = np.asarray(adata.obsm["barcodes"])  # (N, 32)
        unique_barcodes, inverse_indices = np.unique(barcodes_all, axis=0, return_inverse=True)
        indices_all = inverse_indices

        inducing_points = inducing_points_select(adata=adata, barcodes=barcodes_all,
                                                 select_feature=adata.obs['subtree'],
                                                 all_indices=inverse_indices, CELLS_PER_ADDITIONAL_POINT=100)
        initial_inducing_points = np.array(inducing_points)
        print(initial_inducing_points)
        print(f"total number of inducing points: {len(initial_inducing_points)}")

        distance_output_dir = Path("./results/distance")
        distance_matrix_file = distance_output_dir / f"unique_distance_matrix_{p_a}.npy"
        if not os.path.isfile(distance_matrix_file):
            unique_distance_matrix = barcode_distances_between_arrays(unique_barcodes, unique_barcodes)
            np.save(distance_matrix_file, unique_distance_matrix)
        unique_distance_matrix = np.load(distance_matrix_file)

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

        model = DEEPTRACING(input_dim=adata.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
            noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD, distance=unique_distance_matrix,
            initial_inducing_points=initial_inducing_points, fixed_inducing_points=args.fix_inducing_points,
            fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE,
            init_beta_gaussian=args.init_beta_gaussian, min_beta_gaussian=args.min_beta_gaussian, max_beta_gaussian=args.max_beta_gaussian,
            init_beta_gp=args.init_beta_gp, min_beta_gp=args.min_beta_gp, max_beta_gp=args.max_beta_gp,
            dtype=torch.float64, device=args.device, lambda_tc=args.lambda_tc)

        print(str(model))

        if not os.path.isfile(args.model_file):
            t0 = time()
            model.train_model(indices=indices_all, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs["size_factors"].to_numpy(),
                    lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                    train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
            print('Training time: %d seconds.' % int(time() - t0))
        else:
            model.load_model(args.model_file)

        # model.curve_loss()

        final_latent,gp_latent, gaussian_latent = model.batching_latent_samples(X=indices_all, Y=adata.X,
                                                                                batch_size=adata.X.shape[0])
        np.savetxt(args.final_latent_file, final_latent, delimiter=",")
        np.savetxt(args.gp_latent_file, gp_latent, delimiter=",")
        np.savetxt(args.gaussian_latent_file, gaussian_latent, delimiter=",")
