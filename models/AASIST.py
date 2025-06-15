"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # att_map = torch.matmul(att_map, self.att_weight12)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)

        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class SpeakerConditioningModule(nn.Module):
    """
    Module for speaker-aware conditioning in AASIST.
    Can condition at the frame level or utterance level.
    """
    def __init__(self, 
                 spk_emb_dim: int,
                 target_dim: int,
                 conditioning_level: str = "frame",
                 use_attention: bool = True):
        """
        Args:
            spk_emb_dim (int): Dimension of speaker embedding vector
            target_dim (int): Dimension of features to condition
            conditioning_level (str): Either 'frame' or 'utterance'
            use_attention (bool): Whether to use attention for conditioning
        """
        super().__init__()
        self.conditioning_level = conditioning_level
        self.use_attention = use_attention
        
        # Projection to match dimensions
        self.proj = nn.Linear(spk_emb_dim, target_dim)
        
        if use_attention:
            # Attention mechanism for better integration
            self.attention = nn.Sequential(
                nn.Linear(target_dim * 2, target_dim),
                nn.Tanh(),
                nn.Linear(target_dim, 1),
                nn.Softmax(dim=1)
            )
            
            # Final fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(target_dim * 2, target_dim),
                nn.ReLU()
            )
        else:
            # Simple concatenation and projection
            self.fusion = nn.Sequential(
                nn.Linear(target_dim * 2, target_dim),
                nn.ReLU()
            )
    
    def forward(self, features, speaker_embedding):
        """
        Apply speaker conditioning to features
        
        Args:
            features: Features to condition (batch_size, seq_len, dim) for frame-level
                     or (batch_size, dim) for utterance-level
            speaker_embedding: Speaker embedding vectors (batch_size, spk_emb_dim)
        
        Returns:
            Conditioned features with the same shape as input features
        """
        # Project speaker embedding to match feature dimensions
        spk_proj = self.proj(speaker_embedding)  # (batch_size, target_dim)
        
        if self.conditioning_level == "frame":
            # Frame-level conditioning
            batch_size, seq_len, dim = features.size()
            
            # Expand speaker embedding to match sequence length
            spk_proj = spk_proj.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, target_dim)
            
            if self.use_attention:
                # Compute attention weights
                concat_feats = torch.cat([features, spk_proj], dim=2)
                attn_weights = self.attention(concat_feats)  # (batch_size, seq_len, 1)
                
                # Apply attention weights
                spk_context = attn_weights * spk_proj
                
                # Fuse features with speaker context
                conditioned_features = self.fusion(torch.cat([features, spk_context], dim=2))
            else:
                # Simple concatenation and projection
                conditioned_features = self.fusion(torch.cat([features, spk_proj], dim=2))
                
        else:  # utterance-level
            # Utterance-level conditioning
            if features.dim() == 3:
                # If features are sequential, pool them to get utterance-level representation
                features = torch.mean(features, dim=1)  # (batch_size, dim)
                
            # Simple concatenation and projection for utterance level
            conditioned_features = self.fusion(torch.cat([features, spk_proj], dim=1))
            
        return conditioned_features


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Res2NetBlock(nn.Module):
    """Res2Net block with squeeze-and-excitation layer"""
    def __init__(self, nb_filts, first=False, width=14, scale=8):
        super().__init__()
        self.first = first
        self.width = width
        self.scale = scale
        
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        
        # Res2Net structure with width and scale
        self.nums = self.width
        self.convs = nn.ModuleList()
        
        # Split convolutions
        for i in range(self.nums):
            self.convs.append(
                nn.Conv2d(in_channels=nb_filts[0] // self.width + (1 if i == 0 else 0),
                          out_channels=nb_filts[1] // self.width,
                          kernel_size=(2, 3),
                          padding=(1, 1),
                          stride=1)
            )
        
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        
        # Final convolution after concatenation
        self.conv_cat = nn.Conv2d(in_channels=nb_filts[1],
                                 out_channels=nb_filts[1],
                                 kernel_size=(2, 3),
                                 padding=(0, 1),
                                 stride=1)
        
        # Squeeze and excitation layer
        self.se = SELayer(channel=nb_filts[1], reduction=16)
        
        # Downsample if needed
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                           out_channels=nb_filts[1],
                                           padding=(0, 1),
                                           kernel_size=(1, 3),
                                           stride=1)
        else:
            self.downsample = False
            
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        
        # Initial batch norm if not first block
        if not self.first:
            x = self.bn1(x)
            x = self.selu(x)
        
        # Split and process with Res2Net structure
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        outputs = []
        
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i] if i % self.scale == 0 else spx[i]
            sp = self.convs[i](sp)
            outputs.append(sp)
            
        # Concatenate outputs
        out = torch.cat(outputs, dim=1)
        
        # Apply batch norm and second convolution
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv_cat(out)
        
        # Apply SE module
        out = self.se(out)
        
        # Downsample residual if needed
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        # Add residual connection
        out += identity
        
        # Max pooling
        out = self.mp(out)
        
        return out


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]
        
        # Res2Net parameters
        res2net_width = d_args.get("res2net_width", 14)
        res2net_scale = d_args.get("res2net_scale", 8)
        
        # Speaker conditioning parameters
        self.use_speaker_conditioning = d_args.get("speaker_conditioning", False)
        if self.use_speaker_conditioning:
            self.spk_emb_dim = d_args.get("spk_emb_dim", 256)
            self.conditioning_level = d_args.get("conditioning_level", "frame")
            self.use_attention = d_args.get("use_attention", True)
            
            # Create speaker conditioning modules for different features
            self.spk_cond_gat = SpeakerConditioningModule(
                spk_emb_dim=self.spk_emb_dim,
                target_dim=gat_dims[1],
                conditioning_level=self.conditioning_level,
                use_attention=self.use_attention
            )

        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=d_args["first_conv"],
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Res2NetBlock(nb_filts=filts[1], first=True, width=res2net_width, scale=res2net_scale)),
            nn.Sequential(Res2NetBlock(nb_filts=filts[2], width=res2net_width, scale=res2net_scale)),
            nn.Sequential(Res2NetBlock(nb_filts=filts[3], width=res2net_width, scale=res2net_scale)),
            nn.Sequential(Res2NetBlock(nb_filts=filts[4], width=res2net_width, scale=res2net_scale)),
            nn.Sequential(Res2NetBlock(nb_filts=filts[4], width=res2net_width, scale=res2net_scale)),
            nn.Sequential(Res2NetBlock(nb_filts=filts[4], width=res2net_width, scale=res2net_scale)))

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, Freq_aug=False, speaker_embedding=None):
        """
        Forward pass with optional speaker conditioning
        
        Args:
            x: Input audio
            Freq_aug: Whether to use frequency augmentation
            speaker_embedding: Optional speaker embedding for conditioning (batch_size, spk_emb_dim)
        """
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # get embeddings using encoder
        # (#bs, #filt, #spec, #seq)
        e = self.encoder(x)

        # spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        
        # Apply speaker conditioning if enabled and speaker embedding is provided
        if self.use_speaker_conditioning and speaker_embedding is not None:
            # Apply conditioning to the fused features before the final layer
            if self.conditioning_level == "frame":
                # Apply frame-level conditioning to GAT outputs
                out_T = self.spk_cond_gat(out_T, speaker_embedding)
                out_S = self.spk_cond_gat(out_S, speaker_embedding)
        
        # Extract statistics
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
            
        # Apply utterance-level speaker conditioning if enabled
        if self.use_speaker_conditioning and speaker_embedding is not None:
            if self.conditioning_level == "utterance":
                # Apply utterance-level conditioning to final representation
                last_hidden = self.spk_cond_gat(last_hidden, speaker_embedding)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output
