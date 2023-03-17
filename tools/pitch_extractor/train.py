from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from librosa.core import hz_to_mel, mel_frequencies
from loguru import logger
from matplotlib import pyplot as plt
from mmengine import Config
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from fish_diffusion.datasets.utils import build_loader_from_config
from fish_diffusion.modules.pitch_extractors.fish import FishPitchPredictor

torch.set_float32_matmul_precision("medium")


class FishPitchPredictorLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.predictor = FishPitchPredictor()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
        )

        scheduler = ExponentialLR(optim, 0.999)

        return [optim], [scheduler]

    @staticmethod
    def get_mask_from_lengths(lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return (~mask).float()

    def _step(self, batch, batch_idx, mode):
        mels, mel_lens, pitches, audios = (
            batch["mel"].float(),
            batch["mel_lens"].long(),
            batch["pitches"].float(),
            batch["audio"].float(),
        )

        mel_mask = self.get_mask_from_lengths(mel_lens, max_len=mels.shape[-1])

        predicted_pitches = self.predictor(mels, mel_mask)
        pitches = (F.relu(pitches) + 1e-6).log2()

        loss = F.l1_loss(predicted_pitches * mel_mask, pitches * mel_mask)

        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["mel"].shape[0],
        )

        if mode == "valid":
            # TensorBoard logger
            for idx, (mel, mel_len, pitch, pitch_hat) in enumerate(
                zip(
                    mels.cpu().numpy(),
                    mel_lens.cpu().numpy(),
                    pitches.cpu().numpy(),
                    predicted_pitches.cpu().numpy(),
                )
            ):
                image = self.viz(mel[:, :mel_len], pitch[:mel_len], pitch_hat[:mel_len])

                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(
                        f"sample-{idx}/mels",
                        image,
                        global_step=self.global_step,
                    )

                plt.close(image)

        return loss

    def viz(self, mel, pitch, pitch_hat):
        f_min = 0
        f_max = 8000
        n_mels = 128

        min_mel = hz_to_mel(f_min)
        max_mel = hz_to_mel(f_max)
        f_to_mel = lambda x: (hz_to_mel(x) - min_mel) / (max_mel - min_mel) * n_mels
        mel_freqs = mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)

        pitch = f_to_mel(2 ** pitch)
        pitch[pitch <= 1] = float("nan")

        pitch_hat = f_to_mel(2 ** pitch_hat)
        pitch_hat[pitch_hat <= 1] = float("nan")

        figure = plt.figure(figsize=(10, 5))
        plt.imshow(mel, aspect="auto", origin="lower")
        plt.plot(pitch, label="pitch", color="red")
        plt.plot(pitch_hat, label="pitch_hat", color="blue")
        plt.legend()
        plt.yticks(np.arange(0, 128, 10), np.round(mel_freqs[::10]).astype(int))
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time")

        return figure

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "valid")


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Use tensorboard logger, default is wandb.",
    )
    parser.add_argument("--resume-id", type=str, default=None, help="Wandb run id.")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity.")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name.")

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = FishPitchPredictorLightning(cfg)

    logger = (
        TensorBoardLogger("logs", name=cfg.model.type)
        if args.tensorboard
        else WandbLogger(
            project=cfg.model.type,
            save_dir="logs",
            log_model=True,
            name=args.name,
            entity=args.entity,
            resume="must" if args.resume_id else False,
            id=args.resume_id,
        )
    )

    trainer = pl.Trainer(
        logger=logger,
        **cfg.trainer,
    )

    train_loader, valid_loader = build_loader_from_config(cfg, trainer.num_devices)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)
