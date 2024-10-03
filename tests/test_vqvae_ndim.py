from typing import Dict
import pytest
import torch
import numpy as np
from nd_vq_vae.vqvae_ndim import (
    NDimVQVAE,
    NDimAttentionResidualBlock,
    Codebook,
    NDimEncoder,
    NDimDecoder,
    NDimSamePadConv,
    NDimSamePadConvTranspose,
    NDimBatchNorm,
)


@pytest.fixture
def args() -> Dict:
    args = dict(
        embedding_dim=64,
        n_codes=512,
        n_hiddens=128,
        n_res_layers=2,
        downsample=[8, 8, 8],
        codebook_beta=0.25,
        n_dims=3,
        input_shape=[32, 32, 32],
    )
    return args


@pytest.fixture
def batch_size():
    return 4


def test_ndim_vqvae(args, batch_size):
    model = NDimVQVAE(**args)
    x = torch.randn(batch_size, 3, *args["input_shape"])
    recon_loss, x_recon, vq_output = model(x)

    assert x_recon.shape == x.shape
    assert isinstance(recon_loss, torch.Tensor)
    assert isinstance(vq_output, dict)
    assert "encodings" in vq_output
    assert "embeddings" in vq_output
    assert "commitment_loss" in vq_output
    assert "perplexity" in vq_output


def test_ndim_attention_residual_block(args, batch_size):
    block = NDimAttentionResidualBlock(
        args["n_hiddens"], args["n_dims"], args["input_shape"]
    )
    x = torch.randn(
        batch_size, args["n_hiddens"], *[s // 8 for s in args["input_shape"]]
    )
    output = block(x)

    assert output.shape == x.shape


def test_codebook(args, batch_size):
    codebook = Codebook(
        args["n_codes"], args["embedding_dim"], args["codebook_beta"]
    )
    z = torch.randn(
        batch_size,
        args["embedding_dim"],
        *[s // 8 for s in args["input_shape"]]
    )
    output = codebook(z)

    assert "encodings" in output
    assert "embeddings" in output
    assert "commitment_loss" in output
    assert "perplexity" in output
    assert "safe_perplexity" in output


def test_ndim_encoder(args, batch_size):
    encoder = NDimEncoder(
        args["n_hiddens"],
        args["n_res_layers"],
        args["downsample"],
        args["n_dims"],
    )
    x = torch.randn(batch_size, 3, *args["input_shape"])
    output = encoder(x)

    expected_shape = [batch_size, args["n_hiddens"]] + [
        s // d for s, d in zip(args["input_shape"], args["downsample"])
    ]
    assert list(output.shape) == expected_shape


def test_ndim_decoder(args, batch_size):
    decoder = NDimDecoder(
        args["n_hiddens"],
        args["n_res_layers"],
        args["downsample"],
        args["n_dims"],
    )
    x = torch.randn(
        batch_size,
        args["n_hiddens"],
        *[s // d for s, d in zip(args["input_shape"], args["downsample"])]
    )
    output = decoder(x)

    assert output.shape == (batch_size, 3, *args["input_shape"])


def test_ndim_same_pad_conv(args, batch_size):
    conv = NDimSamePadConv(3, args["n_hiddens"], 3, args["n_dims"])
    x = torch.randn(batch_size, 3, *args["input_shape"])
    output = conv(x)

    assert output.shape == (
        batch_size,
        args["n_hiddens"],
        *args["input_shape"],
    )


def test_ndim_same_pad_conv_transpose(args, batch_size):
    convt = NDimSamePadConvTranspose(
        args["n_hiddens"], 3, 4, args["n_dims"], stride=2
    )
    x = torch.randn(
        batch_size, args["n_hiddens"], *[s // 2 for s in args["input_shape"]]
    )
    output = convt(x)

    assert output.shape == (batch_size, 3, *args["input_shape"])


def test_ndim_batch_norm(args, batch_size):
    bn = NDimBatchNorm(args["n_hiddens"], args["n_dims"])
    x = torch.randn(
        batch_size, args["n_hiddens"], *[s // 8 for s in args["input_shape"]]
    )
    output = bn(x)

    assert output.shape == x.shape


def test_vqvae_encode_decode(args, batch_size):
    model = NDimVQVAE(**args)
    x = torch.randn(batch_size, 3, *args["input_shape"])

    encodings = model.encode(x)
    x_recon = model.decode(encodings)

    assert x_recon.shape == x.shape


def test_vqvae_training_step(args, batch_size):
    model = NDimVQVAE(**args)
    x = torch.randn(batch_size, 3, *args["input_shape"])
    batch = {"data": x}

    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_vqvae_validation_step(args, batch_size):
    model = NDimVQVAE(**args)
    x = torch.randn(batch_size, 3, *args["input_shape"])
    batch = {"data": x}

    model.validation_step(batch, 0)
    # This test mainly checks if the validation_step runs without errors


def test_vqvae_configure_optimizers(args):
    model = NDimVQVAE(**args)
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Optimizer)
