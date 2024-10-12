import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import shift_dim, view_range


class NDimMultiHeadAttention(nn.Module):
    def __init__(
        self,
        shape,
        dim_q,
        dim_kv,
        n_head,
        n_layer,
        causal,
        attn_type,
        attn_kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.shape = shape
        self.n_dim = len(shape)

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = NDimLinear(dim_q, n_head * self.d_k, bias=False)
        self.w_ks = NDimLinear(dim_kv, n_head * self.d_k, bias=False)
        self.w_vs = NDimLinear(dim_kv, n_head * self.d_v, bias=False)

        self.fc = NDimLinear(n_head * self.d_v, dim_q, bias=True)
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == "full":
            self.attn = NDimFullAttention(shape, causal, **attn_kwargs)
        elif attn_type == "axial":
            assert not causal, "causal axial attention is not supported"
            self.attn = NDimAxialAttention(shape, **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        b, *spatial_dims, c = q.shape

        # Reshape input: [b, d1, d2, ..., dn, c] -> [b, d1*d2*...*dn, c]
        q = q.view(b, -1, c)
        k = k.view(b, -1, c)
        v = v.view(b, -1, c)

        # Linear projections
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # Split into multiple heads
        q = q.view(b, -1, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(b, -1, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(b, -1, self.n_head, self.d_v).transpose(1, 2)

        # Apply attention
        a = self.attn(q, k, v, decode_step, decode_idx)

        # Concatenate heads
        a = a.transpose(1, 2).contiguous().view(b, -1, self.n_head * self.d_v)

        # Final linear projection
        a = self.fc(a)

        # Reshape output: [b, d1*d2*...*dn, c] -> [b, d1, d2, ..., dn, c]
        return a.view(b, *spatial_dims, -1)


class NDimLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        *spatial_dims, last_dim = x.shape
        if last_dim != self.linear.in_features:
            x = x.view(-1, self.linear.in_features)
        else:
            x = x.view(-1, last_dim)
        x = self.linear(x)
        return x.view(*spatial_dims, -1)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# Attention
class NDimFullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.shape = shape

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(seq_len, seq_len))
            )

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            attn_dropout=self.attn_dropout,
            training=self.training,
        )

        return view_range(out, 2, 3, old_shape)


class NDimAxialAttention(nn.Module):
    def __init__(self, shape, axial_dim=-1):
        super().__init__()
        self.shape = shape
        self.n_dim = len(shape)
        if axial_dim < 0:
            axial_dim = self.n_dim + axial_dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


# Helper Functions
def scaled_dot_product_attention(
    q, k, v, mask=None, attn_dropout=0.0, training=True
):
    # Performs scaled dot-product attention over the second to last dimension

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x d

    return a
