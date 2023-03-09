import torch
import torch.nn as nn
import numpy as np

from torch.nn.functional import softplus
from torch.nn.functional import softplus, sigmoid
from torch.nn import functional as F
from model.helpers import normalize_log_probs, KL_standard_normal

torch.set_default_dtype(torch.float32)

class Encoder(nn.Module):

    def __init__(self, encoder_idx, hidden_dim, z_dim, nonlinearity=torch.nn.ReLU,
                 device="cpu"):
        """
         Encoder for the VAE (neural network that maps P-dimensional data to [mu_z, sigma_z])
        :param encoder_idx: indices for features to be fed into the encoder
        :param hidden_dim: # of neurons in the neural network
        :param z_dim: # of latent dimensions
        :param nonlinearity: 
        """
        super().__init__()
        
        
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Encoder: {device} specified, {self.device} used")
            
        elif device == 'mps':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Encoder: {device} specified, {self.device} used")
            
        else:
            self.device = torch.device("cpu")
            print(f"Encoder: {device} specified, {self.device} used")

        self.z_dim = z_dim
        self.data_dim = len(encoder_idx)
        self.encoder_idx = encoder_idx

        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(self.data_dim, hidden_dim),
            nonlinearity(),
            torch.nn.Linear(hidden_dim, 2*z_dim)
        )

    def forward(self, Y):
        out = self.mapping(Y)

        mu = out[:, 0:self.z_dim]
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])
        return mu, sigma


class ObservationDecoder(nn.Module):
    """Standard Neural Network decoder of a standard VAE."""
    
    def __init__(self, c_dim, x_dim, hidden_dim, z_dim, cov_dim, 
                 nonlinearity=nn.ReLU,
                 device="cpu"):
        super().__init__()
        
        self.cov_dim = cov_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        elif device == 'mps':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        
        self.noise_sd = nn.Parameter(-2.0 * torch.ones(1, c_dim))
        
        self.mapping_zx = nn.Sequential(
            nn.Linear(self.z_dim + self.cov_dim, self.hidden_dim),
            self.nonlinearity(),
            
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             self.nonlinearity(),
            
            nn.Linear(self.hidden_dim, x_dim)
        )
        
        self.mapping_zc = nn.Sequential(
            nn.Linear(self.z_dim + self.cov_dim, self.hidden_dim),
            self.nonlinearity(),
            
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             self.nonlinearity(),
            
            nn.Linear(self.hidden_dim, c_dim)
        )
        
    def forward(self, z, C):
        covar = torch.cat([z, C], 1)
        x_pred = self.mapping_zx(covar)
        c_pred = self.mapping_zc(covar)
        return torch.cat([x_pred, c_pred], axis=1)
        
    def loglik(self, y_obs, y_pred):
        # y_obs: N * (2 (t, s) + D) , y_pred: N * D, N: # of batch, D: observation dimension
        sigma = 1e-4 + softplus(self.noise_sd)
        
        log_p_x = -F.binary_cross_entropy_with_logits(y_pred[:, :self.x_dim, None], y_obs[:, 2 : 2 + self.x_dim, None], reduction='none')
        
        p_data = torch.distributions.normal.Normal(loc=y_pred[:, self.x_dim: self.x_dim + self.c_dim, None], scale=sigma[:, :, None])
        log_p_c = p_data.log_prob(y_obs[:, 2 + self.x_dim : 2 + self.x_dim + self.c_dim, None])
        

        return log_p_c.sum(), log_p_x.sum()

    def loss(self, y_obs, y_pred):

        loglik_c, loglik_x = self.loglik(y_obs, y_pred)


        return -loglik_c, -loglik_x
    


class SurvivalDecoder(nn.Module):
    """ODE-based Neural Network decoder of a survival VAE."""
    def __init__(self, x_dim, hidden_dim, z_dim, cov_dim,
                 nonlinearity=nn.ReLU,
                 device="cpu", n = 15):
        super().__init__()
        
        self.cov_dim = cov_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        elif device == 'mps':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        
        
        # for now, dudt = f(z, w); w=k w.p 1/K
        
        self.output_dim = 1
        

        self.dudt = nn.Sequential(
            nn.Linear(z_dim + cov_dim + 1, hidden_dim),
            nonlinearity(),

#             nn.Linear(hidden_dim, hidden_dim),
#             nonlinearity(),

            nn.Linear(hidden_dim, self.output_dim),
            nn.Softplus()
        )
        
        nn.init.kaiming_normal_(self.dudt[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dudt[2].weight, mode='fan_out', nonlinearity='relu')

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)

    def forward(self, x):
        # w_ : N * 1
        t = x[:, 0][:, None].to(self.device)
        cov = x[:, 1:].to(self.device)
        
        tau = torch.matmul(t/2, 1+self.u_n) # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        reppedx = torch.repeat_interleave(cov, torch.tensor([self.n]*t.shape[0], dtype=torch.long, device=self.device), dim=0)
        taux = torch.cat((tau_, reppedx),1) # Nn x (d+1)
        f_n = self.dudt(taux).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        pred = t/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1))

        return torch.tanh(pred)
    
    def loss(self, z, y, C, return_output = False):
        t = y[:, 0, None]
        s = y[:, 1, None]
        
        x = torch.cat([t, z, C], 1)
        
        N = t.shape[0]
        
        cens_ids = torch.nonzero(torch.eq(s,0))[:,0]
        ncens = cens_ids.size()[0]
        uncens_ids = torch.nonzero(torch.eq(s,1))[:,0]
        
        eps = 1e-8
        

        censterm = 0
        if torch.numel(cens_ids) != 0:
            cdf_cens = self.forward(x[cens_ids, :]).squeeze()
            s_cens = 1 - cdf_cens
            censterm = torch.log(s_cens + eps).sum()
            
        uncensterm = 0
        if torch.numel(uncens_ids) != 0:
            cdf_uncens = self.forward(x[uncens_ids, :]).squeeze()
            dudt_uncens = self.dudt(x[uncens_ids, :]).squeeze()
            uncensterm = (torch.log(1 - cdf_uncens**2 + eps) + torch.log(dudt_uncens + eps)).sum()
        
        if return_output == True:
            return censterm, uncensterm
        
#         print(f"p_t: {- (censterm + uncensterm)}")
        return - (censterm + uncensterm)

        
class cVAE(nn.Module):
    """
    VAE wrapper class (for combining a standard encoder and decoders)
    """

    def __init__(self, encoder, obs_decoder, surv_decoder, lr, beta = 1.0, alpha = 1.0):
        """
        :param lr: learning rate during the gradient step
        :param beta: additional constant for KL(q(z)|p(z)) scaling
        :param alpha: additional constant for Likelihood(P(c|z)) scaling
        """
        super().__init__()
        self.encoder = encoder.to(encoder.device)
        self.obs_decoder = obs_decoder.to(encoder.device)
        self.surv_decoder = surv_decoder.to(encoder.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr
        self.beta = beta
        self.alpha = alpha
        
        
    def forward(self, Y, C, encoder_idx):
        """
        :param Y: data matrix
        :param C: covariate matrix
        :param encoder_idx
        :return: (mu_z, sigma_z, loss)
        """
        # encode
        
        mu_z, sigma_z = self.encoder(Y[:, encoder_idx])
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps
        
        # decode

        y_pred = self.obs_decoder.forward(z, C)
        
        obs_con_loss, obs_bin_loss = self.obs_decoder.loss(Y, y_pred)
        
        survival_loss = self.surv_decoder.loss(z, Y, C)

        # latent space loss
        
        VAE_KL_loss = KL_standard_normal(mu_z, sigma_z)
        
        total_loss = self.alpha * obs_con_loss + obs_bin_loss + survival_loss + self.beta * VAE_KL_loss

        return mu_z, sigma_z, (total_loss, survival_loss, self.alpha * obs_con_loss, obs_bin_loss, VAE_KL_loss)
    
    def get_pdf(self, t, Y, C, encoder_idx, mean = False, seed = 13):
        z = self.get_latent_vars(Y, encoder_idx, mean, seed)
        x = torch.cat([t, z, C], 1)
        F = self.surv_decoder.forward(x).squeeze()
        du = self.surv_decoder.dudt(x).squeeze()
        return -(torch.log(du) + torch.log(1-F**2))
        
    def calculate_survival_lik(self, Y, C, encoder_idx):
        with torch.no_grad():
            mu_z, sigma_z = self.encoder(Y[:, encoder_idx])
            
            survival_loss = self.surv_decoder.loss(mu_z, Y, C)
            
            return -survival_loss
        
    def get_latent_vars(self, Y, encoder_idx, mean = False, seed = 13):
        with torch.no_grad():
            mu_z, sigma_z = self.encoder(Y[:, encoder_idx])
            if mean == True:
                return mu_z
            else:
                torch.manual_seed(seed)
                eps = torch.randn_like(mu_z)
                z = mu_z + sigma_z * eps
                return z
    
    def get_observables(self, Y, C, encoder_idx, seed = 13):
        with torch.no_grad():
            z = self.get_latent_vars(Y, encoder_idx, seed)
            Y_pred = self.obs_decoder.forward(z, C)
            
            x_dim = self.obs_decoder.x_dim
            c_dim = self.obs_decoder.c_dim
            
            x_pred, c_pred_mu = torch.sigmoid(Y_pred[:, :x_dim]), Y_pred[:, x_dim:x_dim+c_dim]
            
            return (x_pred, c_pred_mu)

    def optimize(self, data_loader, n_epochs, y_idx, cov_idx, encoder_idx, logging_freq=10, verbose=True, data_loader_val=None,
                 max_wait=20):
        # sample size
        N = len(data_loader.dataset)

        # scaling for loglikelihood terms
        batch_size = data_loader.batch_size
        batch_scale = N / batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0
            
        if verbose:
            print(f"\tData set size {N}, batch size {batch_size}.\n")
        
        train_loss = {"train_survival_loss":[], 'train_obs_con_loss':[], 'train_obs_bin_loss':[], 'train_kl_loss':[], 'train_total_loss':[]}
        val_loss = {"val_survival": [], "val_obs_con": [], "val_obs_bin":[] , 'val_kl_loss': [], 'val_total_loss':[]}
        
        for epoch in range(n_epochs):
            
            train_survival_loss = 0.0
            train_obs_con_loss = 0.0
            train_obs_bin_loss = 0.0
            train_kl_loss = 0.0
            train_total_loss = 0.0
            
            for batch_idx, (data_subset, ) in enumerate(data_loader):
                Y_subset, C_subset = data_subset[:, y_idx], data_subset[:, cov_idx]
                mu_z, sigma_z, (loss, survival_loss, obs_con_loss, obs_bin_loss, VAE_KL_loss) = self.forward(Y_subset.to(self.encoder.device), C_subset.to(self.encoder.device), encoder_idx)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                                
                train_total_loss += loss.item()
                train_survival_loss += survival_loss.item()
                train_obs_con_loss += obs_con_loss.item()
                train_obs_bin_loss += obs_bin_loss.item()
                train_kl_loss += VAE_KL_loss.item()
                
            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_total_loss:11.2f}")
                if data_loader_val is not None:
                    
                    val_survival_loss = 0.0
                    val_obs_con_loss = 0.0
                    val_obs_bin_loss = 0.0
                    val_kl_loss = 0.0
                    val_total_loss = 0.0
                    
                    for batch_idx, (data_subset, ) in enumerate(data_loader_val):
                        Y_subset, C_subset = data_subset[:, y_idx], data_subset[:, cov_idx]
                        mu_z, sigma_z, (validation_loss, survival_loss, obs_con_loss, obs_bin_loss, VAE_KL_loss) = self.forward(Y_subset.to(self.encoder.device), C_subset.to(self.encoder.device), encoder_idx)

                        val_total_loss += validation_loss.item()
                        val_survival_loss += survival_loss.item()
                        val_obs_con_loss += obs_con_loss.item()
                        val_obs_bin_loss += obs_bin_loss.item()
                        val_kl_loss += VAE_KL_loss.item()

                        if val_total_loss < best_val_loss:
                            best_val_loss = val_total_loss
                            wait = 0
                            print(f"best_epoch: {epoch}")
                            torch.save(self.state_dict(), "low_")
                        else:
                            wait += 1

                        if wait > max_wait:
                            state_dict = torch.load("low_")
                            self.load_state_dict(state_dict)
                            return train_loss, val_loss

                    val_loss['val_survival'].append(val_survival_loss)
                    val_loss['val_obs_con'].append(val_obs_con_loss)
                    val_loss['val_obs_bin'].append(val_obs_bin_loss)
                    val_loss['val_kl_loss'].append(val_kl_loss)
                    val_loss['val_total_loss'].append(val_total_loss)
                    
                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_total_loss:11.2f}")
                    
            train_loss['train_survival_loss'].append(train_survival_loss)
            train_loss['train_obs_con_loss'].append(train_obs_con_loss)
            train_loss['train_obs_bin_loss'].append(train_obs_bin_loss)
            train_loss['train_kl_loss'].append(train_kl_loss)
            train_loss['train_total_loss'].append(train_total_loss)

        if data_loader_val is not None:
            print("loading low_")
            state_dict = torch.load("low_")
            self.load_state_dict(state_dict)
        
        return train_loss, val_loss
        