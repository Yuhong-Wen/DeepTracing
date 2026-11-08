import pickle
from torch import nn
from model import  AutoEncoder
from training import train, train_triplet
import torch
from pathlib import Path
import scanpy as sc
import numpy as np

torch.manual_seed(0)

name = "TedSim"
p_a_list = [0.2, 0.4, 0.6, 0.8]

for p_a in p_a_list:
    output_dir = Path(f"./results")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file =  Path(".../output_data") / f"preprocessed_adata_{p_a}.h5ad.h5ad"
    adata = sc.read(data_file)
    print(adata)


    def normalize(adata, size_factors=True, logtrans_input=True, normalize_input=True):
        adata.raw = adata.copy()
        if size_factors:
            sc.pp.normalize_per_cell(adata)
            adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        else:
            adata.obs['size_factors'] = 1.0
        if logtrans_input:
            sc.pp.log1p(adata)
        if normalize_input:
            sc.pp.scale(adata)

        return adata

    adata = normalize(adata,
                          size_factors=True,
                          normalize_input=True,
                          logtrans_input=True)


    distance_output_dir = Path(f".../results/train/p_a_{p_a}")
    distance_matrix_file = distance_output_dir / f"unique_distance_matrix_{p_a}.npy"
    unique_distance_matrix = np.load(distance_matrix_file)
    print(unique_distance_matrix.shape)


    barcodes_all = np.asarray(adata.obsm["barcodes"])  # (N, 32)
    unique_barcodes, inverse_indices = np.unique(barcodes_all, axis=0, return_inverse=True)

    dist_matrix = unique_distance_matrix[inverse_indices][:, inverse_indices]
    print(dist_matrix.shape)

    from tqdm import trange

    def get_apn_dist_triplet_lut(dist_matrix):
      n = len(dist_matrix)
      apn_lut = np.zeros((n, n, n), dtype=bool)
      for a in trange(len(dist_matrix)):
        # |a - n| >= 2 |a - p|
        apn_lut[a] = dist_matrix[a].reshape(1, -1) >= 2 * dist_matrix[a].reshape(-1, 1)
        apn_lut[a, a, :] = 0
        apn_lut[a, :, a] = 0
      return apn_lut


    apn_lut = get_apn_dist_triplet_lut(dist_matrix)

    # with open(f'./results_all/apn_lut/{name}_{p_a}_apn_pd_triplet_lut.npy', 'wb') as f:
    #   np.save(f, apn_lut)
    lut_path = f'./results/apn_lut/{name}_{p_a}_apn_pd_triplet_lut.npy'
    # apn_lut = np.load(lut_path)
    apn_lut = torch.from_numpy(apn_lut)

    #
    model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=1000, enc_layers=2, non_linearity=nn.LeakyReLU())
    loss_plot = train(model, model_path=f'{output_dir}/{name}_{p_a}_AELR-2-1000_lr1em4_e500_b128_my_hvg.pt',
                      gene_matrix=adata.X, num_epochs=500, device='cuda:3',
                      n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

    model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=100, enc_layers=2, non_linearity=nn.LeakyReLU())
    loss_plot = train(model, model_path=f'{output_dir}/{name}_{p_a}_AELR-2-100_lr1em4_e500_b128_my_hvg.pt',
                      gene_matrix=adata.X, num_epochs=500, device='cuda:3',
                      n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

    model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=1000, enc_layers=3, non_linearity=nn.LeakyReLU())
    loss_plot = train(model, model_path=f'{output_dir}/{name}_{p_a}_AELR-3-1000_lr1em4_e500_b128_my_hvg.pt',
                      gene_matrix=adata.X, num_epochs=500, device='cuda:3',
                      n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

    # train AE for 2000 epochs with triplet loss
    _ = train_triplet(model=None, model_path=f'{output_dir}/{name}_{p_a}_AELR-2-1000_lr1em4_e1000_b128_h1_pd_pre_my_hvg.pt',
                      training_seed=0, n_genes=adata.n_vars, h=1, gene_matrix=adata.X,apn_lut=apn_lut,lut_on_device=False,
                      apn_lut_path=lut_path, display=True, eval_final_all_if_display=False,
                      num_epochs=1000, device='cuda:3', batch_size=128,
                      lr=1e-4, init_path=f'{output_dir}/{name}_{p_a}_AELR-2-1000_lr1em4_e500_b128_my_hvg.pt',
                      save_epochs=[500,1000])


    def get_features(model, genes, device):
      model.to(device).eval()
      with torch.no_grad():
        _, x_z = model.forward(torch.from_numpy(genes).float().to(device), return_z=True)
      return x_z.cpu().numpy()


    features = get_features(torch.load(
        f'{output_dir}/{name}_{p_a}_AELR-2-1000_lr1em4_e1000_b128_h1_pd_pre_my_hvg_e1000.pt',
        map_location=torch.device('cuda:0')),
                            adata.X, 'cuda:0')
    np.save(f'{output_dir}/{name}_{p_a}_latent_genes_PORCELAN_1000.npy',
            features)




