import pytest
import torch
import numpy as np
from nd_vq_vae.attention_ndim import (
    NDimMultiHeadAttention,
    NDimFullAttention,
    NDimAxialAttention,
    scaled_dot_product_attention,
)


@pytest.fixture
def shape():
    return (4, 4, 4)


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def dim_q():
    return 64


@pytest.fixture
def dim_kv():
    return 64


@pytest.fixture
def n_head():
    return 4


@pytest.fixture
def n_layer():
    return 3


def test_ndim_multi_head_attention(
    shape, batch_size, dim_q, dim_kv, n_head, n_layer
):
    attn = NDimMultiHeadAttention(
        shape,
        dim_q,
        dim_kv,
        n_head,
        n_layer,
        causal=False,
        attn_type="full",
        attn_kwargs={"attn_dropout": 0.1},
    )

    q = torch.randn(batch_size, *shape, dim_q)
    k = torch.randn(batch_size, *shape, dim_kv)
    v = torch.randn(batch_size, *shape, dim_kv)

    output = attn(q, k, v)

    assert output.shape == (batch_size, *shape, dim_q)


def test_ndim_full_attention(shape, batch_size, n_head):
    seq_len = np.prod(shape)
    d_k = 16

    full_attn = NDimFullAttention(shape, causal=False, attn_dropout=0.1)

    q = torch.randn(batch_size, n_head, *shape, d_k)
    k = torch.randn(batch_size, n_head, *shape, d_k)
    v = torch.randn(batch_size, n_head, *shape, d_k)

    output = full_attn(q, k, v, decode_step=None, decode_idx=None)

    assert output.shape == (batch_size, n_head, *shape, d_k)


def test_ndim_axial_attention(shape, batch_size, n_head):
    d_k = 16

    axial_attn = NDimAxialAttention(shape, axial_dim=-1)

    q = torch.randn(batch_size, n_head, *shape, d_k)
    k = torch.randn(batch_size, n_head, *shape, d_k)
    v = torch.randn(batch_size, n_head, *shape, d_k)

    output = axial_attn(q, k, v, decode_step=None, decode_idx=None)

    assert output.shape == (batch_size, n_head, *shape, d_k)


def test_scaled_dot_product_attention(batch_size, n_head):
    seq_len = 16
    d_k = 32

    q = torch.randn(batch_size, n_head, seq_len, d_k)
    k = torch.randn(batch_size, n_head, seq_len, d_k)
    v = torch.randn(batch_size, n_head, seq_len, d_k)

    output = scaled_dot_product_attention(q, k, v)

    assert output.shape == (batch_size, n_head, seq_len, d_k)
