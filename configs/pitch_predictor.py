from functools import partial

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy

sampling_rate = 16000
hop_length = 256

trainer = dict(
    accelerator="gpu",
    devices=-1,
    max_epochs=-1,
    precision=16,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            save_top_k=3,
            save_last=True,
            monitor="valid_loss",
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="valid_loss", mode="min", min_delta=1e-3, patience=5),
    ],
    strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
)

model = dict(
    type="FishPitchPredictor",
)

dataset = dict(
    train=dict(
        type="FishPitchPredictorDataset",
        path="dataset/pitch-predictor/train",
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        augmentation=True,
        segment_size=16384,
    ),
    valid=dict(
        type="FishPitchPredictorDataset",
        path="dataset/pitch-predictor/valid",
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        augmentation=False,
        segment_size=-1
    ),
)

dataloader = dict(
    train=dict(
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    ),
)

preprocessing = dict(
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=True,
        f0_min=40.0,
        f0_max=2000.0,
    ),
)
