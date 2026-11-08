import pickle
import torch
from torch import nn
from model import  AutoEncoder
from training import train, train_triplet
import os
from pathlib import Path
import scanpy as sc
import numpy as np
from preprocess import normalize

torch.manual_seed(0)

name = "Puck1_Layer1"
data_path = ".../data/prepared/Puck1_Layer1"
output_dir = Path(f"./results_all")
output_dir.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(data_path)
data_file =  BASE_DIR / "Puck1_Layer1_lineage_spatial.h5ad"
adata = sc.read(data_file)

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

print(adata)

dist_matrix = np.load(Path(data_path) / "lineage_distance.npy", mmap_mode="r")
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

lut_dir = './results_all/apn_lut/'
os.makedirs(lut_dir, exist_ok=True)
# with open(f'./results_all/apn_lut/{name}_{p_a}_{step_size}_{seed}_apn_pd_triplet_lut.npy', 'wb') as f:
#   np.save(f, apn_lut)
lut_path = f'./results_all/apn_lut/{name}_apn_pd_triplet_lut.npy'
# apn_lut = np.load(lut_path)
apn_lut = torch.from_numpy(apn_lut)

#
model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=1000, enc_layers=2, non_linearity=nn.LeakyReLU())
loss_plot = train(model, model_path=f'{output_dir}/{name}_AELR-2-1000_lr1em4_e500_b128_my_hvg.pt',
                  gene_matrix=adata.X, num_epochs=500, device='cuda:2',
                  n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=100, enc_layers=2, non_linearity=nn.LeakyReLU())
loss_plot = train(model, model_path=f'{output_dir}/{name}_AELR-2-100_lr1em4_e500_b128_my_hvg.pt',
                  gene_matrix=adata.X, num_epochs=500, device='cuda:2',
                  n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

model = AutoEncoder(input_dim=adata.n_vars, hidden_dim=1000, enc_layers=3, non_linearity=nn.LeakyReLU())
loss_plot = train(model, model_path=f'{output_dir}/{name}_AELR-3-1000_lr1em4_e500_b128_my_hvg.pt',
                  gene_matrix=adata.X, num_epochs=500, device='cuda:2',
                  n_genes=adata.n_vars, batch_size=128, lr=1e-4, training_seed=0)

# train AE for 2000 epochs with triplet loss
_ = train_triplet(model=None, model_path=f'{output_dir}/{name}_AELR-2-1000_lr1em4_e1000_b128_h1_pd_pre_my_hvg.pt',
                  training_seed=0, n_genes=adata.n_vars, h=1, gene_matrix=adata.X,apn_lut=apn_lut,lut_on_device=False,
                  apn_lut_path=lut_path, display=True, eval_final_all_if_display=False,
                  num_epochs=1000, device='cuda:2', batch_size=128,
                  lr=1e-4, init_path=f'{output_dir}/{name}_AELR-2-1000_lr1em4_e500_b128_my_hvg.pt',
                  save_epochs=[500,1000, 2000])


def get_features(model, genes, device):
  model.to(device).eval()
  with torch.no_grad():
    _, x_z = model.forward(torch.from_numpy(genes).float().to(device), return_z=True)
  return x_z.cpu().numpy()


features = get_features(torch.load(
    f'{output_dir}/{name}_AELR-2-1000_lr1em4_e1000_b128_h1_pd_pre_my_hvg_e1000.pt',
    map_location=torch.device('cuda:0')),
                        adata.X, 'cuda:0')

np.save(f'{output_dir}/{name}_latent_genes_PORCELAN_1000.npy',
        features)




