import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from nd_vq_vae import NDimVQVAE, VideoData

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

    # Set n_dims to 4 for video data (T, C, H, W)
    args.n_dims = 3
    # Set input_shape to match video dimensions
    args.input_shape = [args.sequence_length, 3, args.resolution, args.resolution]
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


if __name__ == '__main__':
    main()
