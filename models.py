import torch
from torch import nn
from torch_geometric.nn import GINConv, GCN2Conv, TransformerConv

from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
# from torch_geometric.utils import softmax
from torch.nn.modules.normalization import LayerNorm
import sys

class ToyMPNN(nn.Module):
    def __init__(self, CONV_OP, nlayer, inch, outch, hidch, is_graph_level=False, pedim=0, trans_nhead=8, trans_nlayer=2, wo_dee=False, edge_dim=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.CONV_OP = CONV_OP
        if self.CONV_OP == GINConv:
            self.enc = CONV_OP(MLP_batchnorm(inch, hidch, hidch))
            self.dec = CONV_OP(MLP_batchnorm(2*hidch, outch, 2*hidch))
            self.net = nn.ModuleList([CONV_OP(MLP_batchnorm(2*hidch, hidch, 2*hidch)) for i in range(nlayer)])
        else:
            self.enc = CONV_OP(inch, hidch)
            self.dec = CONV_OP(2*hidch, outch)
            self.net = nn.ModuleList([CONV_OP(2*hidch, hidch) for i in range(nlayer)])
        self.nlayer = nlayer
        self.is_graph_level = is_graph_level
        if is_graph_level:
            self.pooling = BatchPooling()

    def forward(self, x, edge_index0, de_edge_index0=None, de_x=None, skip_connect=True, batch=None):
        if self.CONV_OP == GCN2Conv:
            x = torch.cat([self.enc(x, x, edge_index0), self.enc(x, x, edge_index0)], -1)
        else:
            x = torch.cat([self.enc(x, edge_index0), self.enc(x, edge_index0)], -1)
        x0 = x
        for i, net in enumerate(self.net):
            edge_index = edge_index0
            if self.CONV_OP == GCN2Conv:
                x = torch.cat([net(x, x0, edge_index), net(x, x0, edge_index)], -1)
            else:
                x = torch.cat([net(x, edge_index), net(x, edge_index)], -1)
            # if self.CONV_OP == GCN2Conv:
            #     x = net(x, x0, edge_index)
            # else:
            #     x = net(x, edge_index) 
            x = torch.relu(x)
            if skip_connect:
                x = x0 + x
        if self.CONV_OP == GCN2Conv:
            x = self.dec(x, x0, edge_index0)
        else:
            x = self.dec(x, edge_index0)
        if self.is_graph_level:
            # x = x.max(-2)
            x = self.pooling(x, batch)
        return x
    
class BatchPooling(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x, batch):
        out = []
        for bi in range(batch.max()+1):
            out.append(x[batch==bi].max(0)[0])
            # out.append(x[batch==bi].mean(0))
        return torch.stack(out)

class ToyMDNN(nn.Module):
    def __init__(self, CONV_OP, nlayer, inch, outch, hidch, is_graph_level=False, pedim=0, trans_nhead=1, trans_nlayer=2, wo_dee=False, edge_dim=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.CONV_OP = CONV_OP
        if self.CONV_OP == GINConv:
            self.enc = CONV_OP(MLP_batchnorm(inch, hidch, hidch))
            self.dec = CONV_OP(MLP_batchnorm(2*hidch, outch, 2*hidch))
            self.net = nn.ModuleList([CONV_OP(MLP_batchnorm(2*hidch, hidch, 2*hidch)) for i in range(nlayer)])
        else:
            self.enc = CONV_OP(inch, hidch)
            self.dec = CONV_OP(2*hidch, outch)
            self.net = nn.ModuleList([CONV_OP(2*hidch, hidch) for i in range(nlayer)])
        self.nlayer = nlayer
        self.is_graph_level = is_graph_level
        if is_graph_level:
            self.pooling = BatchPooling()
            
        self.transformer = DetourTransformer(trans_nlayer, 2*inch+1, hidch, heads=trans_nhead, edge_dim=edge_dim)
        self.pe_lin = nn.Linear(pedim, hidch)
        if not wo_dee:
            self.dee_lin = nn.Linear(1, hidch)
        self.wo_dee = wo_dee
        self.lin_identifier = Linear(1, hidch)
        self.lin_in = nn.Sequential(nn.Linear(inch, hidch), nn.BatchNorm1d(hidch, affine=True))
        self.lin_out = nn.Sequential(nn.Linear(2*hidch, 2*hidch), nn.BatchNorm1d(2*hidch, affine=True), nn.Dropout(0.3),
                                     nn.Linear(2*hidch, 2*hidch), nn.BatchNorm1d(2*hidch, affine=True), nn.Dropout(0.3),
                                     nn.Linear(2*hidch, outch)
                                    )

    def den_weighted_forward(self, x, edge_index0, de_edge_index0, skip_connect=True, batch=None):
        de_edge_index0 = torch.sparse_coo_tensor(edge_index0, de_edge_index0[0], (x.shape[0], x.shape[0])).to_sparse_csr()
        if self.CONV_OP == GCN2Conv:
            x = torch.cat([self.enc(x, x, edge_index0), self.enc(x, x, de_edge_index0)], -1)
        else:
            x = torch.cat([self.enc(x, edge_index0), self.enc(x, de_edge_index0)], -1)
        x0 = x
        for i, net in enumerate(self.net):
            edge_index = edge_index0
            de_edge_index = de_edge_index0
            if self.CONV_OP == GCN2Conv:
                x = torch.cat([net(x, x0, edge_index), net(x, x0, de_edge_index)], -1)
            else:
                x = torch.cat([net(x, edge_index), net(x, de_edge_index)], -1)
            x = torch.relu(x)
            if skip_connect:
                x = x0 + x
        if self.CONV_OP == GCN2Conv:
            x = (self.dec(x, x0, edge_index0) + self.dec(x, x0, de_edge_index0)) / 2
        else:
            x = (self.dec(x, edge_index0) + self.dec(x, de_edge_index0)) / 2
        if self.is_graph_level:
            # x = x.max(-2)
            x = self.pooling(x, batch)
        return x

    def mdnn_forward(self, x, edge_index0, de_x, skip_connect=True, batch=None):
        trans_x, pe, dee, ID, pad_mask = de_x
        pe = self.pe_lin(pe)
        if self.wo_dee:
            dee = None
        else:
            dee = self.dee_lin(dee)
        ID = self.lin_identifier(ID)
        ## Only transformer
        # x = torch.relu(self.lin_in(x))
        # x0 = x
        # x = torch.cat([self.transformer(trans_x, pad_mask, [pe, None, ID]), self.transformer(trans_x, pad_mask, [pe, dee, ID])], -1)
        # if skip_connect:
        #     x = torch.cat([x0, x0], -1) + x
        # x = self.lin_out(x)
        ## Combine with MPNN
        if self.CONV_OP == GCN2Conv:
            x = torch.cat([self.enc(x, x, edge_index0), self.enc(x, x, edge_index0)], -1)
        else:
            x = torch.cat([self.enc(x, edge_index0), self.enc(x, edge_index0)], -1)
        x0 = x
        trans_x = self.transformer(trans_x, pad_mask, [pe, dee, ID]) # x0[..., :int(x0.shape[-1]/2)] + 
        for i, net in enumerate(self.net):
            edge_index = edge_index0
            if self.CONV_OP == GCN2Conv:
                x = torch.cat([net(x, x0, edge_index), trans_x], -1)
            else:
                x = torch.cat([net(x, edge_index), trans_x], -1)
            x = torch.relu(x)
            if skip_connect:
                x = x0 + x
        ## Predict head
        if self.CONV_OP == GCN2Conv:
            x = self.dec(x, x0, edge_index0)
        else:
            x = self.dec(x, edge_index0)
        # x = self.lin_out(x)
        if self.is_graph_level:
            x = self.pooling(x, batch)
        return x


    def forward(self, x, edge_index0, de_edge_index0=None, de_x=None, skip_connect=True, batch=None):
        if de_edge_index0 is not None:
            return self.den_weighted_forward(x, edge_index0, de_edge_index0, skip_connect=skip_connect, batch=batch)
        else:
            return self.mdnn_forward(x, edge_index0, de_x, skip_connect=skip_connect, batch=batch)
    
    

class MLP_batchnorm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden=1, output_activation='relu',batchnorm_affine=True, device='cpu', dtype=torch.float32):
        super().__init__()
        # Inputs to hidden layer linear transformation
        assert num_hidden > 0
        self.num_hidden = num_hidden
        self.dtype = dtype
        
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.linears.append(nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype))
        self.bns.append(nn.BatchNorm1d(hidden_dim, affine=batchnorm_affine, device=device, dtype=dtype))
        
        for layer in range(num_hidden-1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype))
            self.bns.append(nn.BatchNorm1d(hidden_dim, affine=batchnorm_affine, device=device, dtype=dtype))
        self.linears.append(nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype))
        self.activation = nn.functional.relu
        if output_activation == 'relu':
            self.bns.append(nn.BatchNorm1d(output_dim, affine=batchnorm_affine, device=device, dtype=dtype)) 
            self.output_activation = nn.functional.relu
        elif output_activation == 'linear':
            self.output_activation = None
        else:
            raise 'unknown activation for output layer of MLP'
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        
        for layer in range(self.num_hidden):
            x = self.linears[layer](x)
            x = self.bns[layer](x)
            x = self.activation(x)
            
        x = self.linears[-1](x)
        if not (self.output_activation) is None:
            x = self.bns[-1](x)
            x = self.output_activation(x)
        return x
    


class DetourTransformer(nn.Module):

    def __init__(self, 
        nlayer,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.ModuleList([
            DetourTransformerLayer(out_channels, int(out_channels/heads), heads, concat, beta, dropout, edge_dim, bias, root_weight)
            for _ in range(nlayer)])
        self.nlayer = nlayer
        self.in_bn = nn.BatchNorm1d(out_channels, affine=True)
        self.lin_in = nn.Linear(in_channels, out_channels)
        # self.lin_out = nn.Sequential(nn.BatchNorm1d(in_channels, affine=True), nn.Linear(in_channels, out_channels))
        # self.lin_out = Linear(in_channels, out_channels)
    
    # def reset_parameters(self):
    #     self.lin_identifier.reset_parameters()

    def forward(self, x, pad_mask, extra_encodings=[], edge_attr=None):
        # if DeE is not None:
        #     self_loop_dee = DeE if edge_attr is None else torch.cat([DeE, edge_attr], 1)
        #     self_loop_attr = 0 if edge_attr is None else torch.cat([torch.zeros(len(x0), DeE.shape[1]).to(x0.device), x], 1)
        #     edge_index, self_loop_attr = add_self_loops(edge_index, self_loop_dee, fill_value=self_loop_attr, num_nodes=x0.shape[0])
        # else:
        #     edge_index, self_loop_attr = add_self_loops(edge_index, edge_attr, num_nodes=x0.shape[0], fill_value=x0)
        # sortid = edge_index[0].argsort()
        # edge_index = edge_index[:, sortid]
        # if edge_attr is not None:
        #     # edge_attr = self_loop_attr[:, DeE.shape[1]:]
        #     edge_attr = edge_attr[sortid]
        # if DeE is not None:
        #     # DeE = self_loop_attr[:, :DeE.shape[1]]
        #     DeE = DeE[sortid]
        # id_mask = edge_index[0] == edge_index[1]
        # ID = self.lin_identifier(id_mask.float()[:, None])
        # LapPE = LapPE[edge_index[1]]
        # if DeE is None:
        #     DeE = torch.zeros(edge_index.shape[1], x0.shape[1]).to(x0.device)
        # x = torch.stack([x0[edge_index[0]], x0[edge_index[1]], DeE], -1)
        # assert id_mask[-1], id_mask[-100:]
        # x = x + DeE + LapPE + ID 
        x = self.lin_in(x)
        x = self.in_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        for encoding in extra_encodings:
            if encoding is not None:
                x = x + encoding
        for i in range(self.nlayer):
            x = self.net[i](x, pad_mask)
            # x = torch.stack([x0[edge_index[0]], x0[edge_index[1]], DeE], -1)
        x = x[:, 0]
        return x

class DetourTransformerLayer(nn.Module):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        # kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.bn_key = nn.BatchNorm1d(heads * out_channels, affine=True)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.bn_query = nn.BatchNorm1d(heads * out_channels, affine=True)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.bn_value = nn.BatchNorm1d(heads * out_channels, affine=True)
        self.layer_norm = LayerNorm(heads * out_channels)
        # self.lin_ff1 = Linear(heads * out_channels, heads * out_channels)
        self.lin_ff = Linear(heads * out_channels, heads * out_channels)
        self.bn_ff = nn.BatchNorm1d(heads * out_channels, affine=True)
        # self.lin_detour = Linear(3 if edge_dim is None else 4, 1)
        # if edge_dim is not None:
        #     self.lin_edge_detour = nn.Sequential(Linear(edge_dim , in_channels[0]), nn.BatchNorm1d(in_channels[0], affine=True))
        #     self.lin_edge = nn.Sequential(Linear(edge_dim, heads * out_channels, bias=False), nn.BatchNorm1d(heads * out_channels, affine=True))
        # else:
        #     self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            self.bn_skip = nn.BatchNorm1d(heads * out_channels, affine=True)
                                   
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            self.bn_skip = nn.BatchNorm1d(out_channels, affine=True)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     # super().reset_parameters()
    #     self.layer_norm.reset_parameters()
    #     self.lin_key.reset_parameters()
    #     self.lin_query.reset_parameters()
    #     self.lin_value.reset_parameters()
    #     self.lin_detour.reset_parameters()
    #     self.lin_ff.reset_parameters()
    #     if self.edge_dim:
    #         self.lin_edge_detour.reset_parameters()
    #         self.lin_edge.reset_parameters()
    #     self.lin_skip.reset_parameters()
    #     if self.beta:
    #         self.lin_beta.reset_parameters()

    def forward(self, x, pad_mask):#: OptTensor = None, return_attention_weights=None):
        r"""Runs the forward pass of the module.

        Args:
            LapPE: node-wise positional encoding
            DeE: edge-wise detour number encoding
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        node_num, seq_len = x.shape[:2]
        assert H*C == x.shape[-1], x.shape

        # if edge_attr is not None:
        #     edge_attr = self.lin_edge(edge_attr)
        #     x = torch.cat([x, edge_attr[..., None]], -1)
        # x = ID + LapPE + self.lin_detour(x)[..., 0]
        x: PairTensor = (x, x)
        
        query = self.lin_query(x[1])
        query = self.bn_query(query.permute(0, 2, 1)).permute(0, 2, 1).view(node_num, seq_len, H, C) # N S H C
        key = self.lin_key(x[0])
        key = self.bn_key(key.permute(0, 2, 1)).permute(0, 2, 1).view(node_num, seq_len, H, C) # N S H C
        value = self.lin_value(x[0])
        value = self.bn_value(value.permute(0, 2, 1)).permute(0, 2, 1).view(node_num, seq_len, H, C) # N S H C
        # if edge_attr is not None:
        #     value = value + edge_attr

        alpha = query.permute(0, 2, 1, 3) @ key.permute(0, 2, 3, 1) # N, H, S, S
        # while len(pad_mask.shape) < len(alpha.shape):
        pad_mask = pad_mask[:, None]
        alpha = alpha * pad_mask # NHSS * N1S1 = NHSS
        value_ = alpha.softmax(dim=-1) @ value.permute(0, 2, 1, 3) # N, H, S, C
        value_ = value_.permute(0, 2, 1, 3) # N, S, H, C
        if self.concat:
            value_ = value_.reshape(node_num, seq_len, H * C) # N, S, H*C
        else:
            value_ = value_.mean(dim=-2) # N, S, C
        ## Add & Norm
        out = self.layer_norm(value_ + x[0]) # N, S, H*C
        ## Feed-forward & Add & Norm
        ff = self.bn_ff(self.lin_ff(out).permute(0, 2, 1)).permute(0, 2, 1)
        out = self.layer_norm(out + ff) # N, S, H*C

        # out = self.propagate(x[0], torch.where(id_mask)[0], query=query, key=key, value=value)
        ## Re-sort the output
        # out = out[edge_index[0, id_mask]]

        if self.root_weight:
            x_r = self.lin_skip(x[0])
            x_r = self.bn_skip(x_r.permute(0, 2, 1)).permute(0, 2, 1)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r
        return out
        # if isinstance(return_attention_weights, bool):
        #     assert alpha is not None
        #     if isinstance(edge_index, Tensor):
        #         return out, (edge_index, alpha)
        #     elif isinstance(edge_index, SparseTensor):
        #         return out, edge_index.set_value(alpha, layout='coo')
        # else:
        #     return out

    def propagate(self, x, segment, query, key, value) -> Tensor:
        # segment starts with 0
        segment = torch.cat([torch.zeros(1).to(segment.device).long(), segment])
        segment[1:] = segment[1:] + 1
        # out = torch.zeros_like(x).view(-1, self.heads, self.out_channels)
        out = []
        ## Multi-head Self-Attention
        for i in range(len(segment)-1):
            Q = query[segment[i]:segment[i+1]]
            K = key[segment[i]:segment[i+1]]
            V = value[segment[i]:segment[i+1]]
            alpha = Q.permute(1,0,2) @ K.permute(1,2,0) # H, Ne, Ne
            V_ = alpha.softmax(dim=-1) @ V.permute(1,0,2) # H, Ne, C
            ## self loop is added to the last of sequence
            # out[segment[i]:segment[i+1]] = V_.permute(1,0,2)
            out.append(V_[:, -1])

        # out = torch.cat(out, 1)[:, segment[1:]-1] # Nv, H, C
        out = torch.stack(out)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels) # Nv, H*C
        else:
            out = out.mean(dim=1).view(self.out_channels) # Nv, C

        ## Add & Norm
        out = self.layer_norm(out + x[segment[1:]-1]) # Nv, H*C
        ## Feed-forward & Add & Norm
        out = self.layer_norm(out + self.lin_ff(out)) # Nv, H*C
        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')




class DetourTransformerEncoder(nn.Module):

    def __init__(self, 
        nlayer,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        pedim: int = 12,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.ModuleList([
            DetourTransformerLayer(out_channels, int(out_channels/heads), heads, concat, beta, dropout, edge_dim, bias, root_weight)
            for _ in range(nlayer)])
        self.nlayer = nlayer
        self.in_bn = nn.BatchNorm1d(out_channels, affine=True)
        self.lin_in = nn.Linear(in_channels, out_channels)
        self.lin_extra = nn.ModuleList([
            nn.Linear(pedim, out_channels),
            nn.Linear(1, out_channels),
        ])
        # self.lin_out = nn.Sequential(nn.BatchNorm1d(in_channels, affine=True), nn.Linear(in_channels, out_channels))
        self.lin_out = Linear(out_channels, 1)
    
    # def reset_parameters(self):
    #     self.lin_identifier.reset_parameters()

    def forward(self, x, pad_mask, extra_encodings=[], edge_attr=None):
        x = self.lin_in(x)
        x = self.in_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        enci = 0
        for encoding in extra_encodings:
            if encoding is not None:
                encoding = self.lin_extra[enci](encoding)
                enci += 1
                x = x + encoding
        for i in range(self.nlayer):
            x = self.net[i](x, pad_mask)
            # x = torch.stack([x0[edge_index[0]], x0[edge_index[1]], DeE], -1)
        x = self.lin_out(x)#.relu()
        return x

class TopkEncoder(nn.Module):
    def __init__(self, topk, out_channel,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin = nn.Sequential(
            nn.Linear(topk*out_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel)
        )

    def forward(self, x):
        x = self.lin(x.reshape(x.shape[0], -1))
        return x