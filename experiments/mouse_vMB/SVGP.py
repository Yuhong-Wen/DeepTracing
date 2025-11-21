import torch
import torch.nn as nn
import numpy as np
from kernel import Cauchy_Kernel, Kernel, Kernel_T


def _add_diagonal_jitter(matrix, jitter=1e-4):
    Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
    return matrix + jitter * Eye



class SVGP(nn.Module):
    def __init__(self,distance_barcode, distance_time, initial_inducing_points_barcode, initial_inducing_points_time, fixed_inducing_points, fixed_gp_params, kernel_scale_barcode, kernel_scale_time, jitter, N_train, dtype, device):
        super(SVGP, self).__init__()
        self.N_train = N_train
        self.jitter = jitter
        self.dtype = dtype
        self.device = device
        # inducing points
        if fixed_inducing_points:
            self.inducing_index_points_barcode = torch.tensor(initial_inducing_points_barcode, dtype=dtype).to(device)
            self.inducing_index_points_time = torch.tensor(initial_inducing_points_time, dtype=dtype).to(device)
        else:
            self.inducing_index_points_barcode = nn.Parameter(torch.tensor(initial_inducing_points_barcode, dtype=dtype).to(device),
                                                      requires_grad=True)
            self.inducing_index_points_time = nn.Parameter(torch.tensor(initial_inducing_points_time, dtype=dtype).to(device),
                                                      requires_grad=True)
        # length scale of the kernel
        self.kernel = Kernel(distance=distance_barcode, scale=kernel_scale_barcode,
                             fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        self.kernel_T = Kernel_T(distance=distance_time, scale=kernel_scale_time,
                             fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        # self.kernel = Cauchy_Kernel(scale=kernel_scale, fixed_scale=fixed_gp_params, distance=distance,dtype=dtype, device=device).to(device)

    def kernel_matrix(self, x, y):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        matrix = self.kernel(x, y)
        return matrix

    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        """
        Computes L_H for the data in the current batch.
        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :param mu_hat:
        :param A_hat:
        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        b = x.shape[0]
        m = self.inducing_index_points_barcode.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points_barcode,self.inducing_index_points_barcode) # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter)) # (m,m)


        K_nn = self.kernel_matrix(x, x).diagonal() # (b)

        K_nm = self.kernel_matrix(x, self.inducing_index_points_barcode)  # (b, m)
        K_mn = torch.transpose(K_nm, 0, 1)

#        S = A_hat

        # KL term
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))

        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        K_mm_log_det = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))

        KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                             torch.trace(torch.matmul(K_mm_inv, A_hat)) +
                             torch.sum(mu_hat * torch.matmul(K_mm_inv, mu_hat)))

        # diag(K_tilde), (b, )
        precision = 1 / noise

        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn))))

        # k_i \cdot k_i^T, (b, m, m)
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), torch.transpose(K_nm.unsqueeze(2), 1, 2))

        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))

        # Trace terms, (b,)
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(A_hat, lambda_mat))

        # L_3 sum part, (1,)
        L_3_sum_term = -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                                torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                                torch.sum(precision * (y - mean_vector) ** 2))

        return L_3_sum_term, KL_term

    def approximate_posterior_params(self, x_test,x_train=None, y=None, noise=None):
        """
        Computes parameters of q_S.
        :param index_points_test: X_*
        :param index_points_train: X_Train
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return: posterior mean at index points,
                 (diagonal of) posterior covariance matrix at index points
        """
        b = x_train.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points_barcode,self.inducing_index_points_barcode) # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter)) # (m,m)

        K_xx = self.kernel_matrix(x_test, x_test).diagonal()  # (x)
        K_xm = self.kernel_matrix(x_test, self.inducing_index_points_barcode)  # (x, m)
        K_mx = torch.transpose(K_xm, 0, 1)  # (m, x)

        K_nm = self.kernel_matrix(x_train,self.inducing_index_points_barcode)  # (N, m)
        K_mn = torch.transpose(K_nm, 0, 1)  # (m, N)

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:,None]) # (m, m)
        sigma_l_inv = torch.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter)) # (m, m)
        mean_vector = (self.N_train / b) * torch.matmul(K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y/noise)))

        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat

    def kernel_matrix_T(self, x, y):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        matrix = self.kernel_T(x, y)
        return matrix

    def variational_loss_T(self, x, y, noise, mu_hat, A_hat):
        """
        Computes L_H for the data in the current batch.
        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :param mu_hat:
        :param A_hat:
        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        b = x.shape[0]
        m = self.inducing_index_points_time.shape[0]

        K_mm = self.kernel_matrix_T(self.inducing_index_points_time, self.inducing_index_points_time)  # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)

        K_nn = self.kernel_matrix_T(x, x).diagonal()  # (b)

        K_nm = self.kernel_matrix_T(x, self.inducing_index_points_time)  # (b, m)
        K_mn = torch.transpose(K_nm, 0, 1)

        #        S = A_hat

        # KL term
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))

        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        K_mm_log_det = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))

        KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                         torch.trace(torch.matmul(K_mm_inv, A_hat)) +
                         torch.sum(mu_hat * torch.matmul(K_mm_inv, mu_hat)))

        # diag(K_tilde), (b, )
        precision = 1 / noise

        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn))))

        # k_i \cdot k_i^T, (b, m, m)
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), torch.transpose(K_nm.unsqueeze(2), 1, 2))

        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))

        # Trace terms, (b,)
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(A_hat, lambda_mat))

        # L_3 sum part, (1,)
        L_3_sum_term = -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                               torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                               torch.sum(precision * (y - mean_vector) ** 2))

        return L_3_sum_term, KL_term

    def approximate_posterior_params_T(self, x_test, x_train=None, y=None, noise=None):
        """
        Computes parameters of q_S.
        :param index_points_test: X_*
        :param index_points_train: X_Train
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return: posterior mean at index points,
                 (diagonal of) posterior covariance matrix at index points
        """
        b = x_train.shape[0]

        K_mm = self.kernel_matrix_T(self.inducing_index_points_time, self.inducing_index_points_time)  # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)

        K_xx = self.kernel_matrix_T(x_test, x_test).diagonal()  # (x)
        K_xm = self.kernel_matrix_T(x_test, self.inducing_index_points_time)  # (x, m)
        K_mx = torch.transpose(K_xm, 0, 1)  # (m, x)

        K_nm = self.kernel_matrix_T(x_train,self.inducing_index_points_time)  # (N, m)
        K_mn = torch.transpose(K_nm, 0, 1)  # (m, N)

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:, None])  # (m, m)
        sigma_l_inv = torch.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))  # (m, m)
        mean_vector = (self.N_train / b) * torch.matmul(K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y / noise)))

        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat
