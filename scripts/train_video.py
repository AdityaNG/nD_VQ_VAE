"""
Create a video dataset of the following structure and you can run training!
$ tree data/video_dataset/
data/video_dataset/
├── test
│   ├── Gu1D3BnIYZg.mkv  # you can add more videos to both folders
└── train
    └── ceEE_oYuzS4.mkv
"""

import glob
import math
import argparse
import pickle
import warnings
import os.path as osp

import pytorch_lightning as pl
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips
from pytorch_lightning.callbacks import ModelCheckpoint

from nd_vq_vae import NDimVQVAE

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/video_dataset/')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    args.embedding_dim = 64
    args.n_codes = 64
    args.n_hiddens = 64
    args.n_res_layers = 2
    args.codebook_beta = 0.10

    # Set n_dims to 3 for video data (C, T, H, W)
    # Note that channels are treated seperate to the dimensions
    args.n_dims = 3
    # Set input_shape to match video dimensions
    args.input_shape = [3, args.sequence_length, args.resolution, args.resolution]
    # Set downsample to appropriately reduce each dimension
    args.downsample = [2, 2, 2]  # Downsample time by 2, channels by 1, and spatial dimensions by 4

    data = VideoData(args)
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


class VideoDataset(data.Dataset):
    """Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5]"""

    exts = ["avi", "mp4", "webm", "mkv"]

    def __init__(
        self, data_folder, sequence_length, train=True, resolution=64
    ):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        folder = osp.join(data_folder, "train" if train else "test")
        files = sum(
            [
                glob.glob(osp.join(folder, "**", f"*.{ext}"), recursive=True)
                for ext in self.exts
            ],
            [],
        )

        # hacky way to compute # of classes
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings("ignore")
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, "wb"))
        else:
            metadata = pickle.load(open(cache_file, "rb"))
            clips = VideoClips(
                files, sequence_length, _precomputed_metadata=metadata
            )
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]
        return dict(data=preprocess(video, resolution), label=label)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255.0  # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(
        video, size=target_size, mode="bilinear", align_corners=False
    )

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[
        :, :, h_start : h_start + resolution, w_start : w_start + resolution
    ]
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    video -= 0.5

    return video


class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        assert osp.isdir(
            self.args.data_path
        ), f"Dataset is not a folder: {self.args.data_path}"
        dataset = VideoDataset(
            self.args.data_path,
            self.args.sequence_length,
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
