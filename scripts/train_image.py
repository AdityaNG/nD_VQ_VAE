"""
Create an image dataset of the following structure and you can run training!
$ tree data/image_dataset/
data/image_dataset/
├── test
│   ├── 0000001.png  # you can add more images to both folders
└── train
    └── 0000001.png
"""

import glob
import math
import os.path as osp
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint

from nd_vq_vae import NDimVQVAE

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/image_dataset/')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    args.embedding_dim = 64
    args.n_codes = 64
    args.n_hiddens = 64
    args.n_res_layers = 2
    args.codebook_beta = 0.10

    # Set n_dims to 2 for image data (H, W)
    # Note that channels are treated separate to the dimensions
    args.n_dims = 2
    # Set input_shape to match image dimensions
    args.input_shape = [3, args.resolution, args.resolution]
    # Set downsample to appropriately reduce each dimension
    args.downsample = [2, 2]  # Downsample spatial dimensions by 2

    data = ImageData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.val_dataloader()

    model = NDimVQVAE(
        embedding_dim=args.embedding_dim,
        n_codes=args.n_codes,
        n_dims=args.n_dims,
        downsample=args.downsample,
        n_hiddens=args.n_hiddens,
        n_res_layers=args.n_res_layers,
        codebook_beta=args.codebook_beta,
        input_shape=args.input_shape,
    )

    callbacks = []
    callbacks.append(ModelCheckpoint(
        monitor='val/recon_loss',
        mode='min',
        save_top_k=5,
        every_n_epochs=1,
    ))

    # Saving the last checkpoint
    callbacks.append(ModelCheckpoint(
        filename='last-{epoch:02d}-{step:06d}',
        every_n_train_steps=100,
        save_top_k=-1,  # Keep all checkpoints
        save_last=True,
    ))

    trainer = pl.Trainer(
        precision="bf16",  # Use bfloat16 precision
        callbacks=callbacks,
        max_steps=400000,
    )

    trainer.fit(model, data)

class ImageDataset(data.Dataset):
    """Generic dataset for image files stored in folders
    Returns CHW images in the range [-0.5, 0.5]"""

    exts = ["jpg", "jpeg", "png", "bmp"]

    def __init__(self, data_folder, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with images. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding images stored
            resolution: desired resolution of the images
        """
        super().__init__()
        self.train = train
        self.resolution = resolution

        folder = osp.join(data_folder, "train" if train else "test")
        self.files = sum(
            [
                glob.glob(osp.join(folder, "**", f"*.{ext}"), recursive=True)
                for ext in self.exts
            ],
            [],
        )

        # compute # of classes
        self.classes = list(set([get_parent_dir(f) for f in self.files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        resolution = self.resolution
        image_path = self.files[idx]
        image = Image.open(image_path).convert("RGB")

        class_name = get_parent_dir(image_path)
        label = self.class_to_label[class_name]
        return dict(data=preprocess_image(image, resolution), label=label)


def preprocess_image(image, resolution):
    # Convert PIL Image to tensor
    image = (
        torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    )  # CHW

    # scale shorter side to resolution
    c, h, w = image.shape
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    image = F.interpolate(
        image.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # center crop
    c, h, w = image.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    image = image[
        :, h_start : h_start + resolution, w_start : w_start + resolution
    ]

    image -= 0.5

    return image


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


class ImageData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        dataset = ImageDataset(
            self.args.data_path,
            train=train,
            resolution=self.args.resolution,
        )
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    main()
