import pytest
import torch
import numpy as np
import os
from nd_vq_vae.utils import (
    shift_dim,
    view_range,
    save_nd_tensor,
)


@pytest.fixture
def sample_tensor():
    return torch.randn(2, 3, 4, 5, 6)


def test_shift_dim(sample_tensor):
    shifted = shift_dim(sample_tensor, 1, -1)
    assert shifted.shape == (2, 4, 5, 6, 3)

    shifted_back = shift_dim(shifted, -1, 1)
    assert torch.all(shifted_back == sample_tensor)


def test_view_range(sample_tensor):
    # Get the original shape
    original_shape = sample_tensor.shape
    print(f"Original shape: {original_shape}")  # Add this line for debugging

    # Calculate the product of dimensions from i to j
    product_ij = torch.prod(torch.tensor(original_shape[2:4])).item()

    # Create a new shape that preserves the total number of elements
    new_shape = (2, product_ij // 2)
    print(f"New shape: {new_shape}")  # Add this line for debugging

    reshaped = view_range(sample_tensor, 2, 4, new_shape)
    print(f"Reshaped shape: {reshaped.shape}")  # Add this line for debugging
    assert reshaped.shape == (2, 3, 2, product_ij // 2, 6)

    # Reshape back to original
    original = view_range(reshaped, 2, 4, original_shape[2:4])
    assert torch.all(original == sample_tensor)


def test_save_nd_tensor(tmp_path):
    tensor = torch.randn(2, 3, 4, 5)
    fname = str(tmp_path / "test_tensor")
    save_nd_tensor(tensor, fname)

    loaded = np.load(fname + ".npy")
    assert np.allclose(tensor.numpy(), loaded)
