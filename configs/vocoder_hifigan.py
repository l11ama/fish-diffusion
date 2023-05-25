from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

_base_ = [
    "./_base_/schedulers/exponential.py",
]

sampling_rate = 44100
num_mels = 128
n_fft = 2048
hop_length = 512
win_length = 2048
grad_acc_steps = 5  # Simulate 2*10=20 batch size

trainer = dict(
    accelerator="gpu",
    devices=-1,
    max_epochs=-1,
    precision="16-mixed",
    val_check_interval=5000,
    check_val_every_n_epoch=None,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            save_on_train_epoch_end=False,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
    strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="nccl"),
)

model = dict(
    type="HiFiGAN",
    f_min=40.0,
    f_max=16000.0,
    generator=dict(
        hop_length=hop_length,
        upsample_rates=(8, 8, 2, 2, 2),
        upsample_kernel_sizes=(16, 16, 8, 2, 2),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        leaky_relu_slope=0.2,
        num_mels=num_mels,
        upsample_initial_channel=512,
    ),
    # Discriminators
    mpd=dict(periods=[2, 3, 5, 7, 11]),
    mrd=dict(
        resolutions=[
            [1024, 120, 600],
            [2048, 240, 1200],
            [512, 50, 240],
        ],
    ),
    multi_scale_mels=[
        (n_fft, hop_length, win_length),
        (2048, 270, 1080),
        (4096, 540, 2160),
    ],
)

dataset = dict(
    train=dict(
        type="NaiveVOCODERDataset",
        path="/mnt/nvme1/vocoder-dataset/train",
        segment_size=327680,
        pitch_shift=[-12, 12],
        loudness_shift=[0.1, 0.9],
        hop_length=hop_length,
        sampling_rate=sampling_rate,
    ),
    valid=dict(
        type="NaiveVOCODERDataset",
        path="/mnt/nvme1/vocoder-dataset/valid",
        segment_size=None,
        pitch_shift=None,
        loudness_shift=None,
        hop_length=hop_length,
        sampling_rate=sampling_rate,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    ),
)

preprocessing = dict(
    pitch_extractor=dict(
        # Not used in fact
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=2000.0,
        hop_length=hop_length,
    ),
)
