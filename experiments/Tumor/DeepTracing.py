import math
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from SVGP import  SVGP
from I_PID import PIDControl
from VAE_utils import *
from collections import deque


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class DEEPTRACING(nn.Module):
    def __init__(self, input_dim, GP_dim, Normal_dim, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout,
                    distance,initial_inducing_points, fixed_inducing_points, fixed_gp_params, kernel_scale, N_train,
                    KL_loss, dynamicVAE, init_beta_gaussian, min_beta_gaussian, max_beta_gaussian,init_beta_gp, min_beta_gp, max_beta_gp, dtype, device, lambda_tc):
        super(DEEPTRACING, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(distance=distance,initial_inducing_points=initial_inducing_points, fixed_inducing_points=fixed_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.PID_gaussian = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta_gaussian, min_beta=min_beta_gaussian, max_beta=max_beta_gaussian)
        self.PID_gp = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta_gp, min_beta=min_beta_gp, max_beta=max_beta_gp)
        self.KL_loss = KL_loss          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.beta_gaussian = init_beta_gaussian           # beta controls the weight of reconstruction loss
        self.beta_gp = init_beta_gp
        self.lambda_tc = lambda_tc
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.noise = noise              # intensity of random noise
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, input_dim), MeanAct())
        self.dec_disp = nn.Parameter(torch.randn(self.input_dim), requires_grad=True)       # trainable dispersion parameter for NB loss

        self.NB_loss = NBLoss().to(self.device)
        self.MSELoss = MSELoss().to(self.device)
        self.to(device)

        self.log = {"validate_elbo_val":[]}

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def forward(self, x, y, raw_y, size_factors, num_samples=1):

        self.train()
        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        gp_KL_term = gp_ce_term - inside_elbo

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)
            mean_samples_ = self.dec_mean(hidden_samples)
            disp_samples_ = (torch.exp(torch.clamp(self.dec_disp, -15., 15.))).unsqueeze(0)

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            recon_loss += self.NB_loss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
            # recon_loss += self.MSELoss(x=y, mean=mean_samples_,scale_factor=size_factors)
        recon_loss = recon_loss / num_samples

        tc_loss = 0
        if num_samples > 0:
            for i in range(num_samples):
                std = torch.sqrt(p_v)
                eps = torch.randn_like(std)
                z_samples = p_m + eps * std
                log_qz, log_qz_prod = self.estimate_log_qz(z_samples, p_m, torch.log(p_v + 1e-6))
                tc_loss += (log_qz - log_qz_prod).mean()

            tc_loss = tc_loss / num_samples


        noise_reg = 0
        if self.noise > 0:
            for _ in range(num_samples):
                qnet_mu_, qnet_var_ = self.encoder(y + torch.randn_like(y)*self.noise)
                gp_mu_ = qnet_mu_[:, 0:self.GP_dim]
                gp_var_ = qnet_var_[:, 0:self.GP_dim]

#                gaussian_mu_ = qnet_mu_[:, self.GP_dim:]
#                gaussian_var_ = qnet_var_[:, self.GP_dim:]

                gp_p_m_, gp_p_v_ = [], []
                for l in range(self.GP_dim):
                    gp_p_m_l_, gp_p_v_l_, _, _ = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu_[:, l], gp_var_[:, l])
                    gp_p_m_.append(gp_p_m_l_)
                    gp_p_v_.append(gp_p_v_l_)

                gp_p_m_ = torch.stack(gp_p_m_, dim=1)
                gp_p_v_ = torch.stack(gp_p_v_, dim=1)
                noise_reg += torch.sum((gp_p_m - gp_p_m_)**2)
            noise_reg = noise_reg / num_samples

        if self.noise > 0:
            elbo = recon_loss + noise_reg * self.input_dim / self.GP_dim + \
                   self.beta_gp * gp_KL_term + self.beta_gaussian * gaussian_KL_term + \
                   self.lambda_tc * tc_loss
        else:
            elbo = recon_loss + self.beta_gp * gp_KL_term + \
                   self.beta_gaussian * gaussian_KL_term + self.lambda_tc * tc_loss

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg, tc_loss


    def batching_latent_samples(self, X, Y, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: Lineage information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        """

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)

        latent_samples = []
        gp_latents = []
        gaussian_latents = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(ybatch)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params_big_data(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())

            gp_latents.append(gp_p_m.data.cpu().detach())
            gaussian_latents.append(gaussian_mu.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)
        gp_latents = torch.cat(gp_latents, dim=0)
        gaussian_latents = torch.cat(gaussian_latents, dim=0)

        return latent_samples.numpy(),gp_latents.numpy(), gaussian_latents.numpy()


    def train_model(self, indices, ncounts, raw_counts, size_factors, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1,
            train_size=0.95, maxiter=5000, patience=200, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        Parameters:
        -----------
        indices:
            Message used to extract distance.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        size_factor: array_like, shape (n_spots)
            The size factor of each spot, which need for the NB loss.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, default = 0.001
            Weight decay for the opitimizer.
        train_size: float, default = 0.95
            proportion of training size, the other samples are validations.
        maxiter: int, default = 5000
            Maximum number of iterations.
        patience: int, default = 200
            Patience for early stopping.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

        self.train()

        dataset = TensorDataset(torch.tensor(indices, dtype=self.dtype), torch.tensor(ncounts, dtype=self.dtype),
                        torch.tensor(raw_counts, dtype=self.dtype), torch.tensor(size_factors, dtype=self.dtype))

        if train_size < 1:
            train_size_int = int(len(dataset) * train_size)
            validate_size_int = len(dataset) - train_size_int
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[train_size_int, validate_size_int])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if ncounts.shape[0]*train_size > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            tc_loss_val = 0
            noise_reg_val = 0
            num = 0
            for batch_idx, (indices_batch, y_batch, y_raw_batch, sf_batch) in enumerate(dataloader):
                indices_batch = indices_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_raw_batch = y_raw_batch.to(self.device)
                sf_batch = sf_batch.to(self.device)

                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg, tc_loss = \
                    self.forward(x=indices_batch, y=y_batch, raw_y=y_raw_batch, size_factors=sf_batch, num_samples=num_samples)
            # return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, \
                #             mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg
                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()
                tc_loss_val += tc_loss.item()
                if self.noise > 0:
                    noise_reg_val += noise_reg.item()

                num += indices_batch.shape[0]

                if self.dynamicVAE:
                    KL_val = (gp_KL_term.item() + gaussian_KL_term.item()) / indices_batch.shape[0]
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    self.beta_gaussain, _ = self.PID_gaussian.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                    self.beta_gp, _ = self.PID_gp.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                    if len(queue) >= 10:
                        queue.popleft()


            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num
            tc_loss_val = tc_loss_val / num
            noise_reg_val = noise_reg_val/num

            print(
                'Training epoch {}, ELBO:{:.8f}, NB loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f},TC loss:{:.8f}, noise regularization:{:8f}'.format(
                    epoch + 1, elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val, tc_loss_val,
                    noise_reg_val))
            print('Current beta_gaussain', self.beta_gaussain)
            print('Current beta_gp', self.beta_gp)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_indices_batch, validate_y_batch, validate_y_raw_batch, validate_sf_batch) in enumerate(validate_dataloader):
                    validate_indices_batch = validate_indices_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)
                    validate_y_raw_batch = validate_y_raw_batch.to(self.device)
                    validate_sf_batch = validate_sf_batch.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                        self.forward(validate_indices_batch, y=validate_y_batch, raw_y=validate_y_raw_batch, size_factors=validate_sf_batch, num_samples=num_samples)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_indices_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print("Training epoch {}, validating ELBO:{:.8f}".format(epoch+1, validate_elbo_val))
                self.log["validate_elbo_val"].append((float(epoch+1), float(validate_elbo_val)))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)


    def curve_loss(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 2))
        plt.plot(np.array(self.log["validate_elbo_val"])[:, 0], np.array(self.log["validate_elbo_val"])[:, 1])
        plt.xlabel('training epoch')
        plt.ylabel("validate_elbo_val")
        plt.title('Loss curve')
        plt.show()



    def estimate_log_qz(self, samples, mu, logvar):
        B, D = samples.size()

        samples_i = samples.unsqueeze(1)
        mu_j = mu.unsqueeze(0)
        logvar_j = logvar.unsqueeze(0)
        var_j = logvar_j.exp()

        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=samples.device))

        log_probs = -0.5 * (
                log_2pi + logvar_j + (samples_i - mu_j) ** 2 / var_j
        )

        log_qz_cond_x = log_probs.sum(2)
        log_qz = torch.logsumexp(log_qz_cond_x, dim=1) - torch.log(
            torch.tensor(B, dtype=torch.float32, device=samples.device))

        log_qz_prod = torch.logsumexp(log_probs, dim=1) - torch.log(
            torch.tensor(B, dtype=torch.float32, device=samples.device))
        log_qz_prod = log_qz_prod.sum(1)

        return log_qz, log_qz_prod
