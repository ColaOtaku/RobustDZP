from common import series_decomp,ResBlock,match_hidden_dim,MLP
import torch
from torch import nn

from torch.nn import functional as F
from typing import Optional
from torch import nn, Tensor

class OCAP(nn.Module):
    def __init__(self, configs):
        super(OCAP, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.dy_input_dim = configs['dy_input_dim']
        self.dy_embed_dim = configs['dy_embed_dim']
        self.kernel_size = configs['avg_kernel_size']
        self.hidden_dim = configs['hidden_dim']
        device = configs['device']
        self.group = configs['group']
        self.decompsition = series_decomp(self.kernel_size)
        self.end_dim = self.dy_embed_dim + 2

        self.dy_embed = nn.Sequential(
            nn.Linear(self.dy_input_dim,self.dy_embed_dim),
            nn.ReLU() 
        )
        self.seasonal_list = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len).to(device) for i in range(self.group)])
        self.trend_list = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len).to(device) for i in range(self.group)])
        self.residual_list = nn.ModuleList([nn.Linear(self.seq_len,self.pred_len).to(device) for i in range(self.group)])
        # self.end_list = nn.ModuleList([nn.Linear(self.end_dim,1).to(device) for i in range(self.group)])
        self.end_list = nn.ModuleList([MLP(self.end_dim, self.hidden_dim, 1).to(device) for i in range(self.group)])
        self.atten_proj = nn.Sequential(
            nn.Linear(self.hidden_dim ,1),
            nn.Sigmoid(), # ensure normalized \eta falls into (0,1) 
        )
        self.gru = nn.GRU(input_size = self.dy_embed_dim + 2, hidden_size = self.hidden_dim, batch_first = True)

    def forward(self, x, dy, gi):
        b,n,_ = x.shape

        global_in, global_out = [],[]
        for i in range(self.group):
            raw = x[:,torch.nonzero(gi == i).reshape(-1)].mean(axis=1)
            global_in.append(raw)
            global_out.append(self.residual_list[i](raw))

        x = x.transpose(1,2)
        dy = self.dy_embed(dy)
        hidden_former = torch.zeros(b,n,self.seq_len,self.end_dim).to(x.device)
        hidden_after = torch.zeros(b,n,self.pred_len,self.end_dim).to(x.device)
        output_order_former = torch.zeros(b,n,self.seq_len).to(x.device)
        output_order_after = torch.zeros(b,n,self.pred_len).to(x.device)

        for i in range(self.group): #if AOIs are pre_grouped
            index = torch.nonzero(gi == i).reshape(-1)
            raw=x[...,index]

            seasonal_init, trend_init = self.decompsition(raw)
            seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

            seasonal_output = self.seasonal_list[i](seasonal_init)
            trend_output = self.trend_list[i](trend_init)
        
            decomp_former,decomp_after = torch.stack([seasonal_init,trend_init],dim=-1),torch.stack([seasonal_output,trend_output],dim=-1)
            hidden_former_sp,hidden_after_sp = torch.cat([decomp_former,dy[...,index,:-self.pred_len,:]],dim = -1),torch.cat([decomp_after,dy[...,index,-self.pred_len:,:]],dim = -1)

            output_order_former_sp = self.end_list[i](hidden_former_sp).squeeze(-1)+ global_in[i].unsqueeze(1)
            output_order_after_sp = self.end_list[i](hidden_after_sp).squeeze(-1)+ global_out[i].unsqueeze(1)

            hidden_former[:,index] = hidden_former_sp
            hidden_after[:,index] = hidden_after_sp
            output_order_former[:,index] = output_order_former_sp
            output_order_after[:,index] = output_order_after_sp

        atten_former ,_ = self.gru(hidden_former.mean(axis=1).squeeze(1))
        atten_after ,_ = self.gru(hidden_after.mean(axis=1).squeeze(1))

        output_atten_former,output_atten_after = self.atten_proj(atten_former).squeeze(-1),self.atten_proj(atten_after).squeeze(-1)

        return output_order_former, output_atten_former, output_order_after,output_atten_after


class TiDE(nn.Module):
    def __init__(self, param, bias = True): 
        super().__init__()
        L = param['seq_len']
        H = param['pred_len']
        dy_dim = param['dy_dim']
        st_dim = param['st_dim']
        dy_embed_dim = param['dy_embed_dim']
        decode_dim = param['decode_dim']
        hidden_dim = param['hidden_dim']
        enc_layer_nums = param['enc_layer_nums']
        dec_layer_nums = param['dec_layer_nums']
        dropout = param['dropout']

        flatten_dim = L +  (L+H) * dy_embed_dim 
        self.decode_dim = decode_dim
        stack_dim = decode_dim + dy_embed_dim 
        self.L = L
        self.H = H
        self.dy_encoder = ResBlock(dy_dim, match_hidden_dim(dy_dim,dy_embed_dim), dy_embed_dim, dropout, bias)
        self.encoders = TiDE.create_ResBlock(flatten_dim, match_hidden_dim(flatten_dim,hidden_dim), hidden_dim, enc_layer_nums, dropout, bias)
        self.decoders = TiDE.create_ResBlock(hidden_dim, match_hidden_dim(hidden_dim,decode_dim * H), decode_dim * H, dec_layer_nums, dropout, bias)
        self.final_decoder = ResBlock(stack_dim, stack_dim, 1, dropout, bias)
        self.residual_proj = nn.Linear(L, H, bias=bias)

        
    def forward(self, lookback, dynamic):
        # lookback: b,n,t
        # dynamic: b,n,t,d
        # static: b,n,d
        b,n,_ = lookback.shape
        embed_dy = self.dy_encoder(dynamic)
        feature = torch.cat([lookback, embed_dy.reshape(b, n, -1)], dim=-1)

        hidden = self.encoders(feature)
        decoded = self.decoders(hidden).reshape(b, n, self.H, self.decode_dim)
        
        prediction = self.final_decoder(torch.cat([embed_dy[:,:,self.L:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(lookback)
        # prediction = prediction.mean(axis=1)
        return prediction
    
    @staticmethod
    def create_ResBlock(input_dim,hidden_dim, output_dim, layer_nums, dropout, bias):
        ret = []
        if layer_nums == 1:
            ret.append(ResBlock(input_dim, hidden_dim, output_dim,  dropout, bias))
        elif layer_nums ==2:
            ret.append(ResBlock(input_dim, hidden_dim, hidden_dim,  dropout, bias))
            ret.append(ResBlock(hidden_dim, hidden_dim, output_dim,  dropout, bias))
        else:
            ret.append(ResBlock(input_dim, hidden_dim, hidden_dim,  dropout, bias))
            for i in range(layer_nums-2):
                ret.append(ResBlock(hidden_dim, hidden_dim, hidden_dim,  dropout, bias))
            ret.append(ResBlock(hidden_dim, hidden_dim, output_dim,  dropout, bias))
        return nn.Sequential(*ret) 
    
class Dlinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs,):
        super(Dlinear, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
    def forward(self, x):

        # x: [B,N,T]
        x = x.transpose(1,2)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        output = seasonal_output + trend_output

        return output
    


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.pred_len = args['pred_len']
        self.d_model = args['hidden'] 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=2,
            dim_feedforward= self.d_model,
            batch_first=True,
            dropout=0.1,
        )
        self.input_fc = nn.Linear(args['input_size'],self.d_model )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Conv1d(args['seq_len'],args['pred_len'],1)
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.ReLU(),
            nn.Linear(self.d_model//2,1),
        )

    def forward(self, x,tcov):
        # b,n,t
        inputs = torch.cat([x.unsqueeze(-1),tcov],dim=-1)
        b,n,t,_ = inputs.shape
        x = inputs.reshape(b*n,t,-1)
        x = self.input_fc(x)  
        x = self.encoder(x)

        x = self.fc1(x)
        out = self.fc2(x).squeeze(-1)

        return out.reshape(b,n,-1)
    

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

# Fetch activation function
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

# Vanilla Pos Enc used in original transformers architecture    
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

# Custom Enc
def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        # pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

# Custom Enc
def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

# Apply for model
def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=24, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]    
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='sincos', learn_pe=False, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
    

class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class PatchTST(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=24, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters

        if configs['embed']>0:
            c_in = configs['embed']
            
        else:
            c_in = configs['enc_in']
        context_window = configs['seq_len']
        target_window = configs['pred_len']
        
        n_layers = 2
        n_heads = 2
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        # dropout = 0.1
        # fc_dropout = 0.1
        # head_dropout = 0.1
        
        dropout = 0 # appears with better perf
        fc_dropout = 0
        head_dropout = 0

        individual = False
    
        patch_len = 3
        stride = 3
        padding_patch = 1
        
        # model
        self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
                                  fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,pretrain_head=pretrain_head, 
                                  head_type=head_type, individual=individual, verbose=verbose, **kwargs)
        
        self.embed = nn.Conv1d(configs['enc_in'],configs['embed'],1)

    def forward(self, x, type = 'atten'):   
        if type == 'atten': 
            x = self.embed(x)
            x = self.model(x)
            x = x.mean(axis=1)
        else:
            x = self.model(x)
        return x