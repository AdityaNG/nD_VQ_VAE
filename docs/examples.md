# Usage

## How to use `NDimVQVAE`

Below is an example of encoding temporal video data. Video data is 3D since it spans height and width as well as time. Note that the channels are each treated separately and does not count as a dimension.
```py
from nd_vq_vae import NDimVQVAE

sequence_length = 3
channels = 3
res = (128, 256)
input_shape = (channels, sequence_length, res[0], res[1])

model = NDimVQVAE(
    embedding_dim=64,
    n_codes=64,
    n_dims=3,
    downsample=args.downsample,
    n_hiddens=64,
    n_res_layers=2,
    codebook_beta=0.10,
    input_shape=input_shape,
)

x = torch.randn(batch_size, *input_shape)
recon_loss, x_recon, vq_output = model(x)
```

## 3D: Train on Videos

Videos are 3 dimensional data with (Time, Height, Width).
You can construct a video dataset at `data/video_dataset/` as follows:
```bash
$ tree data/video_dataset/
data/video_dataset/
├── test
│   ├── Gu1D3BnIYZg.mkv  # you can add more videos to both folders
└── train
    └── ceEE_oYuzS4.mp4
```

## 2D: Train on Images

Videos are 2 dimensional data with (Height, Width).
You can construct a video dataset at `data/image_dataset/` as follows:
```bash
$ tree data/image_dataset/
data/image_dataset/
├── test
│   ├── 0000001.png  # you can add more images to both folders
└── train
    └── 0000001.png
```

Then you can use the video training script:

```bash
python scripts/train_image.py --data_path data/image_dataset/
```

## 1D

Coming soon!
