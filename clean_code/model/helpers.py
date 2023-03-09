#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.distributions.normal import Normal
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli

def KL_standard_normal(mu, sigma):
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    q = Normal(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))

def normalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized log probabilities or normalized probabilities from unnormalized log probabilities.
    :param log_probs: unnormalized log probabilities
    :return: normalized log probabilities and normalized probabilities
    """
    # use the logsumexp trick to compute the normalizing constant
    log_norm_constant = torch.logsumexp(log_probs, dim = 1, keepdim= True)

    # subtract the normalizing constant from the unnormalized log probabilities to get the normalized log probabilities
    norm_log_probs = log_probs - log_norm_constant

    return norm_log_probs, torch.exp(norm_log_probs)

def approximate_KLqp(logitsp, logitsq):
    pp = torch.sigmoid(logitsp)
    qq = torch.sigmoid(logitsq)
    return (qq * (torch.log(qq) - torch.log(pp)) + (1 - qq) * (torch.log(1 - qq) - torch.log(1 - pp))).sum()

def rsample_RelaxedBernoulli(temperature, logits):
    p = LogitRelaxedBernoulli(temperature, logits=logits)
    return torch.sigmoid(p.rsample())

def expand_grid(a, b):
    nrow_a = a.size()[0]
    nrow_b = b.size()[0]
    ncol_b = b.size()[1]
    x = a.repeat(nrow_b, 1)
    y = b.repeat(1, nrow_a).view(-1, ncol_b)
    return x, y
