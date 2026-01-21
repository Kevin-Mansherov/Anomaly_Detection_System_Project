"""
PURPOSE:
Defines the architecture of the Restricted Boltzmann Machine (RBM)[cite: 134]. 
This generative model consists of a visible layer (input) and a hidden layer (feature representation)[cite: 136].
It is designed to learn the statistical distribution of "normal" network traffic and user behavior 
by performing Gibbs Sampling and calculating reconstruction error[cite: 139, 304, 310].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super(RBM, self).__init__()
        # Initialize weights with small standard deviation to improve convergence
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden)) # Bias for the hidden layer
        self.v_bias = nn.Parameter(torch.zeros(n_visible)) # Bias for the visible layer
        self.k = k # Number of Gibbs sampling steps

    def sample_h(self, v):
        """Sample the hidden layer given the visible layer (Positive Phase)"""
        prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        # Bernoulli sampling adds stochastic noise to improve feature learning
        char_h = torch.bernoulli(prob_h)
        return prob_h, char_h

    def sample_v(self, h):
        """Reconstruct the visible layer from the hidden (Negative Phase)"""
        prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob_v, torch.bernoulli(prob_v)
    
    def free_energy(self, v):
        """Calculate the free energy: F(v) = -a^T v - \ sum(log(1 + exp(W v + b)))"""
        v_bias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (- hidden_term - v_bias_term).mean()

    def forward(self, v):
        """Perform Gibbs Sampling for energy based model training"""
        v_pos = v
        v_neg = v

        for _ in range(self.k):
            _, h_neg = self.sample_h(v_neg)
            _, v_neg = self.sample_v(h_neg)

        return v_pos, v_neg
    
