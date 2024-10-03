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
        elif attn_type == "sparse":
            self.attn = NDimSparseAttention(
                shape, n_head, causal, **attn_kwargs
            )

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


class NDimStridedSparsityConfig:
    def __init__(self, shape, n_head, causal, block, num_local_blocks):
        self.shape = shape
        self.n_dim = len(shape)
        self.n_head = n_head
        self.causal = causal
        self.block = block
        self.num_local_blocks = num_local_blocks

        assert self.num_local_blocks >= 1, "Must have at least 1 local block"
        assert (
            self.seq_len % self.block == 0
        ), "seq len must be divisible by block size"

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    @property
    def seq_len(self):
        return np.prod(self.shape)

    @property
    def num_blocks(self):
        return self.seq_len // self.block

    def set_local_layout(self, layout):
        num_blocks = self.num_blocks
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                max(0, row - self.num_local_blocks),
                (row + 1 if self.causal else end),
            ):
                layout[:, row, col] = 1
        return layout

    def set_global_layout(self, layout):
        num_blocks = self.num_blocks
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[:, row, col] = 1

        return layout

    def make_layout(self):
        layout = torch.zeros(
            (self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64
        )
        layout = self.set_local_layout(layout)
        layout = self.set_global_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        block_layout = self.make_layout()
        assert (
            block_layout.shape[1] == block_layout.shape[2] == self.num_blocks
        )

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(self.n_head):
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j:  # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]]  # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1 :] = 0.0
        return block_row

    # Helper functions

    def _compute_block_shape(self):
        cum_prod = 1
        for i in range(self.n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)
        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(
            self._block_shape
        ), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)


class NDimSparseAttention(nn.Module):
    ops = dict()  # type: ignore
    attn_mask = dict()  # type: ignore
    block_layout = dict()  # type: ignore

    def __init__(
        self,
        shape,
        n_head,
        causal,
        num_local_blocks=4,
        block=32,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.sparsity_config = NDimStridedSparsityConfig(
            shape=shape,
            n_head=n_head,
            causal=causal,
            block=block,
            num_local_blocks=num_local_blocks,
        )

        if self.shape not in NDimSparseAttention.block_layout:
            NDimSparseAttention.block_layout[self.shape] = (
                self.sparsity_config.make_layout()
            )
        if causal and self.shape not in NDimSparseAttention.attn_mask:
            NDimSparseAttention.attn_mask[self.shape] = (
                self.sparsity_config.make_sparse_attn_mask()
            )

    def get_ops(self):
        try:
            from deepspeed.ops.sparse_attention import MatMul, Softmax
        except ImportError:
            raise Exception(
                "Error importing deepspeed. "
                "Please install using "
                "`DS_BUILD_SPARSE_ATTN=1 pip install deepspeed`"
            )
        if self.shape not in NDimSparseAttention.ops:
            sparsity_layout = self.sparsity_config.make_layout()
            sparse_dot_sdd_nt = MatMul(
                sparsity_layout,
                self.sparsity_config.block,
                "sdd",
                trans_a=False,
                trans_b=True,
            )

            sparse_dot_dsd_nn = MatMul(
                sparsity_layout,
                self.sparsity_config.block,
                "dsd",
                trans_a=False,
                trans_b=False,
            )

            sparse_softmax = Softmax(
                sparsity_layout, self.sparsity_config.block
            )

            NDimSparseAttention.ops[self.shape] = (
                sparse_dot_sdd_nt,
                sparse_dot_dsd_nn,
                sparse_softmax,
            )
        return NDimSparseAttention.ops[self.shape]

    def forward(self, q, k, v, decode_step, decode_idx):
        if self.training and self.shape not in NDimSparseAttention.ops:
            self.get_ops()

        NDimSparseAttention.block_layout[self.shape] = (
            NDimSparseAttention.block_layout[self.shape].to(q)
        )
        if self.causal:
            NDimSparseAttention.attn_mask[self.shape] = (
                NDimSparseAttention.attn_mask[self.shape].to(q).type_as(q)
            )
        attn_mask = (
            NDimSparseAttention.attn_mask[self.shape] if self.causal else None
        )

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        if decode_step is not None:
            mask = self.sparsity_config.get_non_block_layout_row(
                NDimSparseAttention.block_layout[self.shape], decode_step
            )
            out = scaled_dot_product_attention(
                q, k, v, mask=mask, training=self.training
            )
        else:
            if q.shape != k.shape or k.shape != v.shape:
                raise Exception("SparseAttention only support self-attention")
            sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = (
                self.get_ops()
            )
            scaling = float(q.shape[-1]) ** -0.5

            attn_output_weights = sparse_dot_sdd_nt(q, k)
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(
                    attn_mask == 0, float("-inf")
                )
            attn_output_weights = sparse_softmax(
                attn_output_weights, scale=scaling
            )

            out = sparse_dot_dsd_nn(attn_output_weights, v)

        return view_range(out, 2, 3, old_shape)


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
