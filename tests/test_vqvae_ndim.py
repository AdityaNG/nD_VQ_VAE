import pytest
import torch
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


import pytest


@pytest.fixture(params=[1, 2, 3])
def n_dims(request):
    return request.param


@pytest.fixture(params=[32])
def input_size(request):
    return request.param


@pytest.fixture(params=[128])
def embedding_dim(request):
    return request.param


@pytest.fixture(params=[256])
def n_codes(request):
    return request.param


@pytest.fixture(params=[128])
def n_hiddens(request):
    return request.param


@pytest.fixture(params=[2, 3])
def n_res_layers(request):
    return request.param


@pytest.fixture(params=[4, 8])
def downsample(request):
    return request.param


@pytest.fixture(params=[0.25])
def codebook_beta(request):
    return request.param


@pytest.fixture(
    params=[
        1,
        4,
    ]
)
def batch_size(request):
    return request.param


@pytest.fixture
def args(
    n_dims,
    input_size,
    embedding_dim,
    n_codes,
    n_hiddens,
    n_res_layers,
    downsample,
    codebook_beta,
):
    input_shape = [input_size] * n_dims
    return {
        "embedding_dim": embedding_dim,
        "n_codes": n_codes,
        "n_hiddens": n_hiddens,
        "n_res_layers": n_res_layers,
        "downsample": [downsample] * n_dims,
        "codebook_beta": codebook_beta,
        "n_dims": n_dims,
        "input_shape": input_shape,
    }


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
        *[s // 8 for s in args["input_shape"]],
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
        *[s // d for s, d in zip(args["input_shape"], args["downsample"])],
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


def test_vqvae_backpropagation(args, batch_size):
    model = NDimVQVAE(**args)
    x = torch.randn(batch_size, 3, *args["input_shape"], requires_grad=True)

    recon_loss, x_recon, vq_output = model(x)

    # Compute a dummy loss
    loss = recon_loss + vq_output["commitment_loss"]

    # Perform backpropagation
    loss.backward()

    # Check if gradients are computed for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    # Check if input gradients are computed
    assert x.grad is not None, "No gradient for input"
