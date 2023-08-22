## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import numpy as np
import matplotlib.pyplot as plt
#from common.Modules import ScaledDotProductAttention
from torch.nn import MultiheadAttention

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5 , attn_dropout=dropout)

#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        

#     def forward(self, q, k, v, mask=None):

#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

#         residual = q

#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

#         if mask is not None:
#             mask = mask.unsqueeze(1)   # For head axis broadcasting.
#  #       print("mask is ",mask)
#         q, attn = self.attention(q, k, v, mask=mask)

#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual

#         q = self.layer_norm(q)

#         return q, attn
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

##################################################################################################################
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,mask = None):
        super().__init__()

#        self.slf_attn = MultiHeadAttention(num_heads, dim, dim//num_heads, dim//num_heads, dropout=attn_drop)
        self.slf_attn = MultiheadAttention(dim,num_heads,dropout=attn_drop,batch_first=True)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_ffn = PositionwiseFeedForward(dim, mlp_hidden_dim, dropout=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x,attn = self.slf_attn(x,x,x)
        x = self.pos_ffn(x)
        return x

class Block_Decoder(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,mask = None):
        super().__init__()
  
        mlp_hidden_dim = int(dim * mlp_ratio)
     
#        self.slf_attn = MultiHeadAttention(num_heads, dim, dim//num_heads, dim//num_heads, dropout=attn_drop)
#        self.enc_attn = MultiHeadAttention(num_heads, dim, dim//num_heads, dim//num_heads, dropout=attn_drop)
        self.slf_attn = MultiheadAttention( dim, num_heads, dropout=attn_drop,batch_first=True)
        self.enc_attn = MultiheadAttention( dim, num_heads, dropout=attn_drop,batch_first=True)

        self.pos_ffn = PositionwiseFeedForward(dim, mlp_hidden_dim, dropout=drop)
        
        
    def get_subsequent_mask(self,seq):
        ''' For masking out the subsequent info. '''
        sz_b, len_s,dim_l = seq.size()
        subsequent_mask = (torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool().repeat(sz_b*8,1,1)
        return subsequent_mask

    def forward(self, enc_output, x):
        trg_mask = self.get_subsequent_mask(x)
        x, dec_slf_attn = self.slf_attn(x, x, x, attn_mask=trg_mask)
        x, dec_enc_attn = self.enc_attn(x, enc_output, enc_output)
        x = self.pos_ffn(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    
    
class Decoder(nn.Module):
    def __init__(self,num_frame_d=9, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None,mask = 1):

        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            enc_output: output of the encoder layer
            num_frame_d (int, tuple): input frame number for decoder
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame_d, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.Spatial_blocks = nn.ModuleList([
             Block(
                 dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
             for i in range(depth)])
        
        self.temporal_blocks = nn.ModuleList([
            Block_Decoder(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,mask = 1)
            for i in range(depth)])  
        
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        self.Enc_dec_norm = norm_layer(embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim_ratio, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
                
        ####### A easy way to implement weighted mean

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        self.position_enc = PositionalEncoding(embed_dim, n_position=200)
        
    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.norm1(self.pos_drop(x))

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def temporal_forward_features(self,enc_output, x):
        b,d,e  = x.shape
        x += self.Temporal_pos_embed[0][0:d].reshape(1,d,e)
#        x += self.Temporal_pos_embed
        
 #       x = self.position_enc(x)
        x = self.norm2(self.pos_drop(x))
        for blk in self.temporal_blocks:
            x = blk(enc_output,x)
        
        return x

                                                                                                                                 
    def forward(self, fut_seq, enc_output):
        x = fut_seq
        b, e, j, d = x.shape
        x = x.permute(0, 3, 1, 2)

        x = self.Spatial_forward_features(x)
        x = self.temporal_forward_features(enc_output,x)

        x = self.head(x)
        
        return x
        
              
class Encoder(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.temporal_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim_ratio, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.position_enc = PositionalEncoding(embed_dim, n_position=200)

    def Spatial_forward_features(self, x):

        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.norm1(self.pos_drop(x))

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def temporal_forward_features(self, x):
        b,d,e  = x.shape        
#        x = self.position_enc(x)
        x += self.Temporal_pos_embed[0][0:d].reshape(1,d,e)
 #       x +=self.Temporal_pos_embed
 
        x = self.norm2(self.pos_drop(x))
        
        for blk in self.temporal_blocks:
            x = blk(x)

        return x

    def forward(self, prv_seq):
        x = prv_seq
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        x = self.Spatial_forward_features(x)
        x = self.temporal_forward_features(x)

        return x


class My_Transformer(nn.Module):
    def __init__(self, num_frame=9, num_frame_d= 9, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4,
             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None):

        super().__init__()
        
# =============================================================================
#         embed_dim_ratio = 32
#         drop_rate=0.0
#         attn_drop_rate=0.0
#         drop_path_rate=0.0
#         depth=4
# =============================================================================
        
        self.encoder = Encoder(num_frame, num_joints=num_joints, in_chans=3, embed_dim_ratio=embed_dim_ratio, depth=depth,
                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,  norm_layer=None)
        
        self.decoder = Decoder(num_frame_d, num_joints=num_joints, in_chans=3, embed_dim_ratio=embed_dim_ratio, depth=depth,
                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,  norm_layer=None, mask = 1)
                
    def forward(self, prv_seq, fut_seq):
        gt = fut_seq
        b,g,j,d = fut_seq.shape
        enc_output = self.encoder(prv_seq) 
        
#        fut_seq = fut_seq.permute(1,0,2,3)
#        fut_seq = fut_seq[0:12].permute(1,0,2,3)
#        fut_ext = torch.zeros(b,1,j,d).cuda()
#        fut_seq2 = gt.permute(1,0,2,3)[13:].permute(1,0,2,3)
#        fut_seq = torch.cat((fut_seq,fut_ext,fut_seq2),dim=1)

        dec_output = self.decoder(fut_seq , enc_output)

        dec_output = dec_output.view(b,g,j,d)

# =============================================================================
#         dec=[]
#         fut=[]
#         for i in range(20):
#             fut.append(fut_seq[0][i][11][0])
#             dec.append(dec_output[0][i][11][0])
#         plt.figure()
#         plt.plot(fut)
#         plt.plot(dec)
#         plt.show()
# =============================================================================
        
        return dec_output
    
class Inference(nn.Module):
    def __init__(self, num_frame=9, num_frame_d= 9, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4,
             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,  norm_layer=None):
        
        super().__init__()
        
        self.estimation_field = num_frame_d
        self.encoder = Encoder(num_frame, num_joints=num_joints, in_chans=3, embed_dim_ratio=embed_dim_ratio, depth=depth,
                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,  norm_layer=None)
        
        self.decoder = Decoder(num_frame_d, num_joints=num_joints, in_chans=3, embed_dim_ratio=embed_dim_ratio, depth=depth,
                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,  norm_layer=None,mask = 1)      

    def forward(self, prv_seq,gt):
        b,g,j,d = prv_seq.shape
        _,e,_,_ = gt.shape
        enc_output = self.encoder(prv_seq)
        fut_seq = prv_seq.permute(1,0,2,3)
        fut_seq = fut_seq[-1].reshape(b,1,j,d)
    
        
        for i in range(1,e):
            dec_output = self.decoder(fut_seq , enc_output)
            _,e,_ = dec_output.shape
            dec_output = dec_output.view(b,e,j,d)
            dec_output = dec_output.permute(1,0,2,3)
            dec_output = dec_output[i-1].reshape(b,1,j,d)
            fut_seq = torch.cat((fut_seq,dec_output),dim=1)

        return fut_seq 
        

    
        
        
