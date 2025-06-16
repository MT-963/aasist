"""
AASIST-Robust: An enhanced version of AASIST with defensive mechanisms
against adversarial attacks.

Based on the original AASIST model.
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import components from the original AASIST model
from models.AASIST import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool, CONV, Residual_block


class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation of the noise. Default: 0.1
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. Default: True
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0.))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach().std() if self.is_relative_detach else self.sigma * x.std()
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class FeatureDenoising(nn.Module):
    """
    Feature denoising module to reduce the effect of adversarial perturbations.
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Non-local means denoising
        self.g = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.theta = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.W = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        """
        x: (batch_size, in_channels, length)
        """
        batch_size, _, length = x.size()
        
        # Apply 1x1 convolutions for feature transformation
        g_x = self.g(x).view(batch_size, self.out_channels, -1)
        theta_x = self.theta(x).view(batch_size, self.out_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.out_channels, -1)
        
        # Compute attention map
        f = torch.matmul(theta_x.permute(0, 2, 1), phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        # Apply attention and transform back
        y = torch.matmul(g_x, f_div_C.permute(0, 2, 1))
        y = y.view(batch_size, self.out_channels, length)
        
        # Residual connection
        W_y = self.W(y)
        z = self.bn(W_y) + x
        
        return z


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        
        # Input feature extraction
        self.conv = CONV(
            out_channels=d_args['first_conv'],
            kernel_size=1024,
            in_channels=1,
            stride=256,
            padding=0,
        )
        
        # Extract filter configurations
        filts = d_args['filts']
        
        # Residual blocks for feature extraction - fixed initialization
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4]))
        )
        
        # First batch normalization
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        # Gaussian noise for adversarial defense
        self.gaussian_noise = GaussianNoise(sigma=0.1)
        
        # Feature denoising modules
        self.denoising = FeatureDenoising(in_channels=filts[-1][-1])
        
        # Position embedding
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        
        # Graph attention layers
        self.GAT_layer_S = GraphAttentionLayer(
            in_dim=filts[-1][-1],
            out_dim=d_args['gat_dims'][0],
            temperature=d_args['temperatures'][0]
        )
        
        self.GAT_layer_T = GraphAttentionLayer(
            in_dim=filts[-1][-1],
            out_dim=d_args['gat_dims'][0],
            temperature=d_args['temperatures'][1]
        )
        
        # Master nodes
        self.master1 = nn.Parameter(torch.randn(1, 1, d_args['gat_dims'][0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, d_args['gat_dims'][0]))
        
        # Heterogeneous graph attention layers
        self.HtrgGAT_layer_ST1 = HtrgGraphAttentionLayer(
            in_dim=d_args['gat_dims'][0],
            out_dim=d_args['gat_dims'][1],
            temperature=d_args['temperatures'][2]
        )
        
        self.HtrgGAT_layer_ST2 = HtrgGraphAttentionLayer(
            in_dim=d_args['gat_dims'][1],
            out_dim=d_args['gat_dims'][1],
            temperature=d_args['temperatures'][3]
        )
        
        # Graph pooling layers
        self.pool_S = GraphPool(
            k=d_args['pool_ratios'][0],
            in_dim=d_args['gat_dims'][0],
            p=0.3
        )
        
        self.pool_T = GraphPool(
            k=d_args['pool_ratios'][1],
            in_dim=d_args['gat_dims'][0],
            p=0.3
        )
        
        self.pool_hS = GraphPool(
            k=d_args['pool_ratios'][2],
            in_dim=d_args['gat_dims'][1],
            p=0.3
        )
        
        self.pool_hT = GraphPool(
            k=d_args['pool_ratios'][3],
            in_dim=d_args['gat_dims'][1],
            p=0.3
        )
        
        # Dropout layers
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        
        # Output layers with dropout for regularization
        self.out_layer = nn.Linear(4 * d_args['gat_dims'][1], 2)
        
        # Ensemble output layer for improved robustness
        self.aux_out_layer = nn.Linear(filts[-1][-1], 2)
        
        # Ensemble weight (learnable)
        self.ensemble_weight = nn.Parameter(torch.tensor([0.8, 0.2]))

    def forward(self, x, Freq_aug=False):
        """
        x: (batch_size, 1, length)
        """
        # Apply Gaussian noise during training for adversarial defense
        if self.training:
            x = self.gaussian_noise(x)
        
        # Handle different input shapes
        if len(x.shape) == 4:
            x = x.squeeze(1).squeeze(1)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Feature extraction
        x = self.conv(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        
        # Get embeddings using encoder
        e = self.encoder(x)
        
        # Store features for auxiliary output
        e_flat = e.mean(dim=(2, 3))  # Global average pooling
        
        # Apply feature denoising if in training mode
        if self.training:
            # Extract a feature map for denoising
            e_denoise, _ = torch.max(torch.abs(e), dim=2)  # max along freq
            e_denoise = self.denoising(e_denoise)
            # Add back to the original features with a residual connection
            e = e + e_denoise.unsqueeze(2)
        
        # Spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S
        
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        # Temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)
        
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # Learnable master node
        master = self.master1.expand(x.size(0), -1, -1)
        
        # Heterogeneous graph attention
        out_T, out_S, master = self.HtrgGAT_layer_ST1(out_T, out_S, master=master)
        
        # Apply pooling
        out_S = self.pool_hS(out_S)
        out_T = self.pool_hT(out_T)
        
        # Second heterogeneous graph attention
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST2(out_T, out_S, master=master)
        
        # Residual connections
        out_T = out_T + out_T_aug
        out_S = out_S + out_S_aug
        master = master + master_aug
        
        # Apply dropout
        out_T = self.drop_way(out_T)
        out_S = self.drop_way(out_S)
        master = self.drop_way(master)
        
        # Global pooling
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        M = master.squeeze(1)
        
        # Concatenate features
        out = torch.cat([T_max, T_avg, S_max, S_avg], dim=1)
        out = self.drop(out)
        
        # Main output
        logits = self.out_layer(out)
        
        # Auxiliary output for ensemble
        aux_logits = self.aux_out_layer(e_flat)
        
        # Normalize ensemble weights
        norm_weights = F.softmax(self.ensemble_weight, dim=0)
        
        # Ensemble prediction (weighted average)
        if self.training:
            # During training, use main output
            ensemble_logits = logits
        else:
            # During inference, use weighted ensemble
            ensemble_logits = norm_weights[0] * logits + norm_weights[1] * aux_logits
        
        return ensemble_logits, logits 