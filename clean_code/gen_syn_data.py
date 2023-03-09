import torch
from torch.nn.functional import softplus
import torch.distributions as dist
import numpy as np
import pandas as pd


class DGP_v3:
    def __init__(self, ratio = 0.3):
        self.a_i = torch.distributions.Uniform(0, 1)
        self.pi_i = torch.distributions.Categorical(torch.tensor([1/3, 1/3, 1/3]))
        self.beta = torch.distributions.Uniform(-.5, .5)
        
        
        self.rho_0 = torch.tensor([0.8, 0.5, 0.1])
        self.rho_1 = torch.tensor([0.8, 0.1, 0.5])
        self.rho_2 = torch.tensor([0.1, 0.5, 0.8])
        
        self.mu_0 = torch.tensor([1., 1.])
        self.mu_1 = torch.tensor([-1., -1.])
        self.mu_2 = torch.tensor([1., -1.])
        
        self.rhos = [self.rho_0, self.rho_1, self.rho_2]
        self.mus = [self.mu_0, self.mu_1, self.mu_2]
        self.alphas = torch.tensor([-3., -1.5, -0.5])
        self.betas = [self.beta.sample((5,)) for _ in range(3)]
        
        self.ratio = ratio
        
    def generate_samples(self, N, p_censor = 0.4,  random_state = 13, baseline = 10):
        torch.manual_seed(random_state)
        samples = torch.zeros((N, 9))

        for _ in range(N):
            a = self.a_i.sample((1, ))
            pi = self.pi_i.sample((1,))
            
            idx = int(pi.detach().item())
            b = self.betas[idx]
            x_id = torch.distributions.Bernoulli(self.rhos[idx]).sample((1,))
            c_ij = torch.distributions.MultivariateNormal(self.mus[idx], torch.eye(2)).sample((1,))

            scale = torch.exp( self.ratio * (b * torch.hstack([x_id, abs(c_ij)])).sum() + (1 - self.ratio) * self.alphas[idx] * a)
            T_i = torch.distributions.Exponential(baseline * scale).sample((1,))
            u_i = torch.distributions.Exponential(baseline * 0.5 * scale).sample((1,))
            
            if T_i < u_i:
                s_i = torch.ones((1,))
                t_i = T_i
            else:
                s_i = torch.zeros((1,))
                t_i = u_i

            samples[_, :] = torch.hstack([t_i, s_i[:, None], x_id, c_ij, a[:, None], pi[:, None]])
            
        return samples
    
    def get_weights(self):
        return [torch.hstack([self.ratio * self.betas[i], (1-self.ratio) * self.alphas[i]]) for i in range(3)]
    
    @staticmethod
    def generate_pdf(target, dataset, weights, i, baseline = 10):
        assert type(target) == np.ndarray, "target must be numpy.ndarray"
        N = len(target)
        w = int(dataset[i, -1].item())
        covar = torch.hstack([dataset[i, 2:5], abs(dataset[i, 5:7]), dataset[i, 7:8]])
        
        pdfs = []

        for i in range(target.size):
            scale = torch.exp( (weights[w] * covar).sum() )
            pdfs.append(torch.exp(dist.Exponential(rate = baseline * scale).log_prob(torch.tensor(target[i]))))
        
        return torch.tensor(pdfs)
    
    @staticmethod
    def generate_cdf(target, dataset, weights, i, baseline = 10):
        cdfs = []
        w = int(dataset[i, -1].item())
        covar = torch.hstack([dataset[i, 2:5], abs(dataset[i, 5:7]), dataset[i, 7:8]])
        
        pdfs = []
        
        for i in range(target.size):
            scale = torch.exp( (weights[w] * covar).sum() )
            cdfs.append(dist.Exponential(rate = baseline * scale).cdf(torch.tensor(target[i])))
        
        
        return torch.tensor(cdfs)
    
    @staticmethod
    def predict(dataset, weights, T = 500, baseline = 10):
        # generate T * N pd.dataframe (t,n)entry corresponds to 1 - cdf(t) of subject n
        N = dataset.shape[0]
        t_eval = np.linspace(dataset[:, 0].min().item(), dataset[:, 0].max().item(), T)
        output = torch.zeros((T, N), dtype=torch.float32)
        
        for i in range(N):
            w = int(dataset[i, -1].item())
            covar = torch.hstack([dataset[i, 2:5], abs(dataset[i, 5:7]), dataset[i, 7:8]])
            
            scale = torch.exp( (weights[w] * covar).sum() )
            output[:, i] = dist.Exponential(rate = baseline * scale).cdf(torch.tensor(t_eval))
            
        return pd.DataFrame(output.detach().numpy(), index = t_eval)
    
    @staticmethod
    def calculate_test_loss(dataset, weights, baseline = 10):
        
        K = len(weights)
        D = weights[0].shape[0]
        np_weights = np.zeros((K, D))
        for k in range(K):
            np_weights[k] = weights[k].detach().numpy()
        
        t = dataset[:, 0]
        s = dataset[:, 1]
        w = dataset[:, -1]
        covar = torch.hstack([dataset[:, 2:5], abs(dataset[:, 5:7]), dataset[:, 7:8]])
        
        cens_ids = torch.nonzero(torch.eq(s,0))[:,0]
        ncens = cens_ids.size()[0]
        uncens_ids = torch.nonzero(torch.eq(s,1))[:,0]
        
        eps = 1e-8
        

        censterm = 0
        if torch.numel(cens_ids) != 0:
            scales = baseline * torch.exp(torch.from_numpy(np.multiply(np_weights[w[cens_ids].long().detach().numpy()], covar[cens_ids].detach().numpy() )).sum(axis=1) )
            cens_dist = dist.Exponential(rate = scales[:, None])
            cdf_cens = cens_dist.cdf(t[cens_ids][:, None]).squeeze()
            s_cens = 1 - cdf_cens
            censterm = torch.log(s_cens + eps).sum()
            
        uncensterm = 0
        if torch.numel(uncens_ids) != 0:
            unscales = baseline * torch.exp(torch.from_numpy(np.multiply(np_weights[w[uncens_ids].long().detach().numpy()], covar[uncens_ids].detach().numpy() )).sum(axis=1) )
            uncens_dist = dist.Exponential(rate = unscales[:, None])
#             cdf_uncens = uncens_dist.cdf(t[uncens_ids][:, None]).squeeze()
            dudt_uncens = uncens_dist.log_prob(t[uncens_ids][:, None] + eps).squeeze() 
            
            uncensterm = dudt_uncens.sum()

        return censterm, uncensterm, censterm + uncensterm
    

