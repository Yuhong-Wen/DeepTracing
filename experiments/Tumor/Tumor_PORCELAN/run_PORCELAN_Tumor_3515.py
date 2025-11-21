from torch import nn
from model import  AutoEncoder
from training import train, train_triplet
from pathlib import Path
import numpy as np
import scanpy as sc
from preprocess import normalize
import torch
from tqdm import trange

torch.manual_seed(0)

data_path = '/data/DeepTracing/experiments/Tumor/data'
name = '3515_Lkb1_T1_Fam'

results_path = Path(f"./results")
results_path.mkdir(parents=True, exist_ok=True)

pruned_adata = sc.read_h5ad(f'{data_path}/preprocessed/{name}_pruned_adata.h5ad')
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

adata = normalize(pruned_adata,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True)
print(adata)


dist_matrix = np.load(f'{data_path}/preprocessed/{name}_dist_matrix.npy')
with open(f'{data_path}/preprocessed/{name}_cells.txt') as f:
    cell_names = np.array(f.read().splitlines())
all_indices = np.arange(len(cell_names))
print(dist_matrix.shape)


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
lut_path = f'{results_path}/{name}_apn_pd_triplet_lut.npy'
# with open(lut_path, 'wb') as f:
#   np.save(f, apn_lut)
# apn_lut = np.load(lut_path)
apn_lut = torch.from_numpy(apn_lut)

model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=1000, enc_layers=2, non_linearity=nn.LeakyReLU())
loss_plot = train(model, model_path=f'{results_path}/{name}_AELR-2-1000_lr1em4_e500_b128.pt',
                  gene_matrix=adata.X, num_epochs=500, device='cuda:1',
                  n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=100, enc_layers=2, non_linearity=nn.LeakyReLU())
loss_plot = train(model, model_path=f'{results_path}/{name}_AELR-2-100_lr1em4_e500_b128.pt',
                  gene_matrix=adata.X, num_epochs=500, device='cuda:1',
                  n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=1000, enc_layers=3, non_linearity=nn.LeakyReLU())
loss_plot = train(model, model_path=f'{results_path}/{name}_AELR-3-1000_lr1em4_e500_b128_my.pt',
                  gene_matrix=adata.X, num_epochs=500, device='cuda:1',
                  n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

# train AE for 1000 epochs with triplet loss
_ = train_triplet(model=None, model_path=f'{results_path}/{name}_AELR-2-1000_lr1em4_e1000_b128_h1_pd_pre.pt',
                  training_seed=12345, n_genes=adata.n_vars, h=1, gene_matrix=adata.X,apn_lut=apn_lut,lut_on_device=False,
                  apn_lut_path=lut_path, display=True, num_epochs=1000, device='cuda:1', batch_size=128,
                  lr=1e-4, init_path=f'{results_path}/{name}_AELR-2-1000_lr1em4_e500_b128.pt',
                  save_epochs=[500, 1000])


def get_features(model, genes, device):
  model.to(device).eval()
  with torch.no_grad():
    _, x_z = model.forward(torch.from_numpy(genes).float().to(device), return_z=True)
  return x_z.cpu().numpy()

features = get_features(torch.load(f'{results_path}/{name}_AELR-2-1000_lr1em4_e1000_b128_h1_pd_pre_e1000.pt',
               map_location=torch.device('cuda:0')),
                        adata.X, 'cuda:0')

np.save(f'{results_path}/{name}_latent_genes_PORCELAN_1000.npy', features)