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
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        # Initialize weights with small standard deviation to improve convergence
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden)) # Bias for the hidden layer
        self.v_bias = nn.Parameter(torch.zeros(n_visible)) # Bias for the visible layer

    def sample_h(self, v):
        """Sample the hidden layer given the visible layer (Positive Phase)"""
        prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        # Bernoulli sampling adds stochastic noise to improve feature learning
        char_h = torch.bernoulli(prob_h)
        return prob_h, char_h

    def sample_v(self, h):
        """Reconstruct the visible layer from the hidden (Negative Phase)"""
        prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob_v

    def forward(self, v):
        """Perform one step of Gibbs Sampling for reconstruction"""
        prob_h, char_h = self.sample_h(v)
        v_reconstructed = self.sample_v(char_h)
        return v_reconstructed, prob_h