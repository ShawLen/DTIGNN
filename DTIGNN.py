import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from common.utils import norm_Adj, re_normalization
from prepareData import onehot_to_phase, generate_actphase, revise_unknown
import sys
sys.path.append("..")


def clones(module, N):
    '''
    produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 1 means reachable; 0 means unreachable
    return torch.from_numpy(subsequent_mask) == 0


# class spatialGCN(nn.Module):
#     def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
#         super(spatialGCN, self).__init__()
#         self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.Theta = nn.Linear(in_channels, out_channels, bias=False)

#     def forward(self, x):
#         '''
#         spatial graph convolution operation
#         x: (batch_size, N, T, F_in)
#         :return: (batch_size, N, T, F_out)
#         '''
#         batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

#         x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

#         return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)

        # the sum of each row is 1; (b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


# class spatialAttentionGCN(nn.Module):
#     def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
#         super(spatialAttentionGCN, self).__init__()
#         self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.Theta = nn.Linear(in_channels, out_channels, bias=False)
#         self.SAt = Spatial_Attention_layer(dropout=dropout)

#     def forward(self, x):
#         '''
#         spatial graph convolution operation
#         x: (batch_size, N, T, F_in)
#         :return: (batch_size, N, T, F_out)
#         '''

#         batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

#         spatial_attention = self.SAt(x)  # (batch, T, N, N)

#         x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

#         spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)
#         # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)
#         return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, DEVICE, sym_norm_Adj_matrix, mask_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.mask_matrix = mask_matrix
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)
        self.DEVICE = DEVICE

    def forward(self, x, phaseAct_matrix):
        '''
        spatial graph convolution operation,including imputation
        :param x: (batch_size, N, T, F_in)
        :param phaseAct_matrix: (B, T, N, N)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        for t in range(1,num_of_timesteps):
            # scaled self attention: (B, T, N, N)
            spatial_attention = self.SAt(x) / math.sqrt(in_channels)
            # (B,T,N,N)-permute->(B,T,N,N)
            sat_act = (phaseAct_matrix * spatial_attention).permute(0, 1, 3, 2)
            # (B,T,N,N)(B,T,N,F)->(B,T,N,F)-permute->(B,N,F,T)
            x_predict = torch.matmul(sat_act[:, 1:], x.permute(0, 2, 1, 3)[:, :num_of_timesteps-1]).permute(0, 2, 3, 1)
            # (B,N,F,T)-permute->(B,N,T,F)
            x = (revise_unknown(x.permute(0, 1, 3, 2), x_predict, self.mask_matrix).to(self.DEVICE)).permute(0, 1, 3, 2)
        
        spatial_attention = (self.SAt(x) / math.sqrt(in_channels)).reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        phaseAct_adj = (phaseAct_matrix * self.sym_norm_Adj_matrix).reshape(-1,num_of_vertices, num_of_vertices)

        satAct_adj = phaseAct_adj.mul(spatial_attention)
        # (B,N,T,F)->permute->(B,T,N,F)-reshape->(B*T,N,F)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))

        # (B*T,N,F_in)->(B*T,N, F_out)-reshape->(B,T,N,F_out)->(B,N,T,F_out)
        return F.relu(self.Theta(torch.matmul(satAct_adj.permute(0, 2, 1), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        # 在模型中定义一个常量，.step时不会被更新
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in(64))
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            # (batch_size, N, T, F_in) + (1,1,T,d_model)
            x = x + self.pe[:, :, self.lookup_index, :]
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, layernum,  phaseAct_matrix, x, sublayer):
        '''
        x: (batch, N, T, d_model)
        sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            if layernum != 2:
                return x + self.dropout(sublayer(self.norm(x)))
            else:
                gcn = sublayer(self.norm(x), phaseAct_matrix)
                return x + self.dropout(gcn)
        if self.residual_connection and (not self.use_LayerNorm):
            if layernum != 2:
                return x + self.dropout(sublayer(x))
            else:
                gcn = sublayer(x, phaseAct_matrix)
                return x + self.dropout(gcn)
        if (not self.residual_connection) and self.use_LayerNorm:
            if layernum != 2:
                return self.dropout(sublayer(self.norm(x)))
            else:
                gcn = sublayer(self.norm(x), phaseAct_matrix)
                return self.dropout(gcn)


class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phaseAct_matrix):
        '''
        :param x: (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        gcn = self.gcn(x, phaseAct_matrix)
        return self.dropout(F.relu(gcn))


def attention(query, key, value, mask=None, dropout=None):
    '''
    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / \
        math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        # -1e9 means attention scores=0
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention( query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


# key causal; query causal;
class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):
    def __init__(self, nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_mhalf * points_per_mhalf

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
    
            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.h_length, :], key[:, :, :self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[
                :, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.h_length, :].permute(
                    0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 1d conv on query, 1d conv on key
class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_mhalf * points_per_mhalf

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        query=key=value
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.h_length > 0:
                # l:Conv2d
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context,
                                                                                                                                                       (query[:, :, :self.h_length, :], key[:, :, : self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)
            # (batch, N, h, T, d_k)
            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(
                0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous(
            ).view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.h_length, :].permute(
                    0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


# query: causal conv; key 1d conv
class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1)//2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_mhalf * points_per_mhalf

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(query[:, :, : self.h_length, :].permute(
                    0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, : self.h_length, :].permute(
                    0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, : self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    def __init__(self, PAct, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.PAct = PAct
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, src, trg, _mean, _std):
        '''
        :param src:  (batch_size, N, T_in, F_in)
        :param trg: (batch, N, T_out, F_out)
        :param _mean: (1, 1, F, 1)
        :param _std: (1, 1, F, 1)
        '''
        encoder_output, encoder_refill = self.encode(src, _mean, _std)  # (batch_size, N, T_in, d_model)

        return self.decode(trg, encoder_output), encoder_refill

    def encode(self, src, _mean, _std):
        '''
        :param src: (batch_size, N, T_in, F_in)
        :param _mean: (1, 1, F, 1)
        :param _std: (1, 1, F, 1)
        : returns: encoder_output:(B, N, T, F)
        '''
        phaseAct_matrix = self.PAct(src, _mean, _std)  # (b,T,N,N)
        phaseAct_matrix = torch.from_numpy(phaseAct_matrix).type(torch.FloatTensor).to(self.DEVICE)
        encoder_output = self.encoder(self.src_embed(src), phaseAct_matrix)
        return encoder_output,self.prediction_generator(encoder_output)

    def decode(self, trg, encoder_output):
        '''
        :param trg:(batch_size, N, T, F(3))
        :param encoder_output: (B, N, T, F)
        :return: (B, N, T, F)
        '''
        batch_size, N, T, _ = trg.shape
        phaseAct_matrix = torch.ones(batch_size, T, N, N).type(torch.FloatTensor).to(self.DEVICE)
        h = self.trg_embed(trg)
        output = self.decoder(h, encoder_output, phaseAct_matrix)
        return self.prediction_generator(output)


class EncoderLayer(nn.Module):
    def __init__(self, size, sat_act, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.sat_act = sat_act
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(
                size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x, phaseAct_matrix):
        '''
        :param x: (B, N, T_in, F_in)
        :param phaseAct_matrix: (B, T, N, N)
        :return: (B, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](0, phaseAct_matrix, x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](2, phaseAct_matrix, x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x, phaseAct_matrix)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, phaseAct_matrix):
        '''
        :param x: (B, N, T_in, F_in)
        :param phaseAct_matrix: (B, T, N, N)
        :return: (B, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x, phaseAct_matrix)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory, phaseAct_matrix):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            # self.self_attn: captures the correlation in the decoder sequence
            x = self.sublayer[0](0, phaseAct_matrix, x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            # self.src_attn: capture the correlations between the decoder sequence (queries) and the encoder output sequence(keys)
            x = self.sublayer[1](1, phaseAct_matrix, x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
            # output:  (batch, N, T', d_model)
            return self.sublayer[2](2, phaseAct_matrix, x, self.feed_forward_gcn)
        else:
            # output: (batch, N, T', d_model)
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)
            # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)
            # output:  (batch, N, T', d_model)
            return self.feed_forward_gcn(x, phaseAct_matrix)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, phaseAct_matrix):
        '''
        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory, phaseAct_matrix)
        return self.norm(x)


def search_index(max_len, num_of_depend, num_for_predict, points_per_mhalf):
    '''
    :param max_len: int, length of all encoder input
    :param num_of_depend: int,
    :param num_for_predict: int, the number of points will be predicted for each sample
    :param points_per_mhalf: int, number of points per hour, depends on data
    :return: list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_mhalf * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


class Phase_Act_layer(nn.Module):
    def __init__(self, adj_mx, adj_phase):
        super(Phase_Act_layer, self).__init__()
        self.adj_mx = adj_mx
        # (N,N)
        self.adj_phase = adj_phase

    def forward(self, x, _mean, _std):
        '''
        :param x:(b,N,T,F)
        :param _mean:(1,1,F(11),1)
        :param _std:(1,1,F(11),1)
        '''
        _, N, _, _ = x.shape
        x_renor = re_normalization(x.cpu().numpy().transpose(0, 1, 3, 2), _mean, _std)[:, :, 3:]
        x_renor_ = np.where((np.abs(x_renor-1.) >= 1e-6), 0, 1)
        # (B,T,N,2)
        onehot2phase = onehot_to_phase(x_renor_)
        # (B,T,N,N)
        # compute phase_act matrix of each time according to adj_phase,x_next_phase
        phase_matrix = generate_actphase(onehot2phase, self.adj_mx, self.adj_phase)
        # add self_loop
        phase_matrix = phase_matrix + np.eye(N)
        return phase_matrix


def make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, adj_phase, mask_matrix, nb_head,
               num_of_mhalf, points_per_mhalf, num_for_predict, len_input, dropout=.0, aware_temporal_context=True,
               SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):

    c = copy.deepcopy

    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)

    num_of_vertices = norm_Adj_matrix.shape[0]

    src_dense = nn.Linear(encoder_input_size, d_model)
    PAct = Phase_Act_layer(adj_mx, adj_phase)

    sat_act = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(DEVICE, norm_Adj_matrix, mask_matrix, d_model, d_model), dropout=dropout)

    position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(DEVICE, norm_Adj_matrix, mask_matrix, d_model, d_model), dropout=dropout)

    # target input projection
    trg_dense = nn.Linear(decoder_input_size, d_model)

    # encoder temporal position embedding
    max_len = num_of_mhalf * points_per_mhalf

    h_index = search_index(max_len, num_of_mhalf, len_input, points_per_mhalf)
    en_lookup_index = h_index

    print('TemporalPositionalEncoding max_len:', max_len)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    if aware_temporal_context:  # employ temporal trend-aware attention
        attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size, dropout=dropout)  # encoder的trend-aware attention用一维卷积
        attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size, dropout=dropout)
        att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size, dropout=dropout)  # decoder的trend-aware attention用因果卷积
    else:  # employ traditional self attention
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and (TE):
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)

    encoderLayer = EncoderLayer(d_model, c(sat_act), attn_ss, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    decoder = Decoder(decoderLayer, num_layers)

    generator = nn.Linear(d_model, decoder_input_size)

    model = EncoderDecoder(PAct,
                           encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE)
    # param init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
