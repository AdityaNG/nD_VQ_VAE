import argparse
import math
import os
from datetime import datetime
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .attention_ndim import NDimMultiHeadAttention
from .utils import save_nd_tensor, shift_dim


class NDimVQVAE(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        n_codes: int,
        n_dims: int,
        downsample: List[int],
        n_hiddens: int,
        n_res_layers: int,
        codebook_beta: int,
        input_shape: List[int],
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.n_dims = n_dims
        self.downsample = downsample
        self.n_hiddens = n_hiddens
        self.n_res_layers = n_res_layers
        self.codebook_beta = codebook_beta
        self.input_shape = input_shape

        self.encoder = NDimEncoder(
            self.n_hiddens, self.n_res_layers, self.downsample, self.n_dims
        )
        self.decoder = NDimDecoder(
            self.n_hiddens, self.n_res_layers, self.downsample, self.n_dims
        )  # Use downsample as upsample

        self.pre_vq_conv = NDimSamePadConv(
            self.n_hiddens, self.embedding_dim, 1, self.n_dims
        )
        self.post_vq_conv = NDimSamePadConv(
            self.embedding_dim, self.n_hiddens, 1, self.n_dims
        )

        self.codebook = Codebook(
            self.n_codes, self.embedding_dim, self.codebook_beta
        )
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = tuple(self.input_shape)
        return tuple([s // d for s, d in zip(input_shape, self.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        else:
            return vq_output["encodings"]

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        recon_loss = F.mse_loss(x_recon, x) / 0.06

        return recon_loss, x_recon, vq_output

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        recon_loss, _, vq_output = self.forward(x)
        commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss

        self.log("train/recon_loss", recon_loss)
        self.log("train/perplexity", vq_output["perplexity"])
        self.log("train/commitment_loss", vq_output["commitment_loss"])

        if batch_idx % 100 == 0:
            self._log_reconstructions(x, batch_idx, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        recon_loss, _, vq_output = self.forward(x)
        self.log("val/recon_loss", recon_loss)
        self.log("val/perplexity", vq_output["perplexity"])
        self.log("val/commitment_loss", vq_output["commitment_loss"])

        if batch_idx == 0:
            self._log_reconstructions(x, batch_idx, "val")

    def _log_reconstructions(self, x, batch_idx, prefix):
        if not hasattr(self, "logger") or self.logger is None:
            return
        with torch.no_grad():
            _, samples, _ = self.forward(x)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.logger.log_dir,
                f"{prefix}_{current_time}_{str(batch_idx).zfill(8)}.npy",
            )
            output_file_gt = os.path.join(
                self.logger.log_dir,
                f"{prefix}_gt_{current_time}_{str(batch_idx).zfill(8)}.npy",
            )

            x_out = torch.clamp(x, -0.5, 0.5) + 0.5

            save_nd_tensor(samples, output_file)
            save_nd_tensor(x_out, output_file_gt)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--n_codes", type=int, default=2048)
        parser.add_argument("--n_hiddens", type=int, default=240)
        parser.add_argument("--n_res_layers", type=int, default=4)
        parser.add_argument("--downsample", nargs="+", default=[2, 2, 2])
        parser.add_argument("--codebook_beta", type=float, default=0.10)
        parser.add_argument("--n_dims", type=int, required=True)
        parser.add_argument(
            "--input_shape", nargs="+", type=int, required=True
        )
        return parser


class NDimAttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens, n_dims, shape):
        super().__init__()
        self.norm1 = NDimBatchNorm(n_hiddens, n_dims)
        self.relu1 = nn.ReLU()
        self.conv1 = NDimSamePadConv(
            n_hiddens, n_hiddens // 2, 3, n_dims, bias=False
        )
        self.norm2 = NDimBatchNorm(n_hiddens // 2, n_dims)
        self.relu2 = nn.ReLU()
        self.conv2 = NDimSamePadConv(
            n_hiddens // 2, n_hiddens, 1, n_dims, bias=False
        )
        self.norm3 = NDimBatchNorm(n_hiddens, n_dims)
        self.relu3 = nn.ReLU()
        self.attn = NDimMultiHeadAttention(
            shape=shape,
            dim_q=n_hiddens,
            dim_kv=n_hiddens,
            n_head=2,
            n_layer=1,
            causal=False,
            attn_type="full",
            attn_kwargs={"attn_dropout": 0.1},
        )

    def forward(self, x):
        h = self.norm1(x)
        h = self.relu1(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.norm3(h)
        h = self.relu3(h)
        h = self.attn(h, h, h)
        return x + h


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, beta):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.beta = beta
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        # d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(
            flat_inputs
        )
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = self.beta * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # # Add a safe perplexity calculation
        safe_avg_probs = torch.clamp(avg_probs, min=1e-7, max=1.0)
        safe_perplexity = torch.exp(
            -torch.sum(safe_avg_probs * torch.log(safe_avg_probs))
        )

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
            safe_perplexity=safe_perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


class NDimEncoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample, n_dims):
        super().__init__()
        self.n_dims = n_dims
        n_times_downsample = [int(math.log2(d)) for d in downsample]
        self.convs = nn.ModuleList()
        max_ds = max(n_times_downsample)
        in_channels = 3
        for i in range(max_ds):
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = NDimSamePadConv(
                in_channels, n_hiddens, 4, n_dims, stride=stride
            )
            self.convs.append(conv)
            in_channels = n_hiddens
            n_times_downsample = [max(0, d - 1) for d in n_times_downsample]
        self.conv_last = NDimSamePadConv(in_channels, n_hiddens, 3, n_dims)

        # Calculate the shape after downsampling
        self.downsampled_shape = [
            s // d for s, d in zip([32, 32, 32], downsample)
        ]  # Adjust this based on your input shape

        self.res_stack = nn.ModuleList(
            [
                NDimAttentionResidualBlock(
                    n_hiddens, n_dims, self.downsampled_shape
                )
                for _ in range(n_res_layers)
            ]
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        for res_block in self.res_stack:
            h = res_block(h)
        return h


class NDimDecoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample, n_dims):
        super().__init__()
        self.n_dims = n_dims

        # Calculate the shape after upsampling
        self.upsampled_shape = [
            s * u for s, u in zip([4, 4, 4], upsample)
        ]  # Adjust this based on your latent shape

        self.res_stack = nn.Sequential(
            *[
                NDimAttentionResidualBlock(
                    n_hiddens, n_dims, self.upsampled_shape
                )
                for _ in range(n_res_layers)
            ],
            NDimBatchNorm(n_hiddens, n_dims),
            nn.ReLU(),
        )

        n_times_upsample = [int(math.log2(d)) for d in upsample]
        max_us = max(n_times_upsample)
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = NDimSamePadConvTranspose(
                n_hiddens, out_channels, 4, n_dims, stride=us
            )
            self.convts.append(convt)
            n_times_upsample = [max(0, d - 1) for d in n_times_upsample]

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


class NDimSamePadConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_dims,
        stride=1,
        bias=True,
    ):
        super().__init__()
        self.n_dims = n_dims
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * n_dims
        if isinstance(stride, int):
            stride = (stride,) * n_dims

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = getattr(nn, f"Conv{n_dims}d")(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class NDimSamePadConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_dims,
        stride=1,
        bias=True,
    ):
        super().__init__()
        self.n_dims = n_dims
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * n_dims
        if isinstance(stride, int):
            stride = (stride,) * n_dims

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = getattr(nn, f"ConvTranspose{n_dims}d")(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


class NDimBatchNorm(nn.Module):
    def __init__(self, num_features, n_dims):
        super().__init__()
        self.bn = getattr(nn, f"BatchNorm{n_dims}d")(num_features)

    def forward(self, x):
        return self.bn(x)


# ... existing code ...
