_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Import TV3S components
import sys
import os
# Add TV3S path - this will be handled by the training script
# tv3s_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'TV3S')
# if tv3s_path not in sys.path:
#     sys.path.insert(0, tv3s_path)

try:
    from mmseg.models.decode_heads.tv3s_head import TV3SHead_shift_city
    from mmseg.models.segmentors.encoder_decoder_clips import EncoderDecoder_clips
    from TV3S.utils.datasets.cityscapes import CityscapesDataset_clips
    # Import transforms to register them
    import TV3S.utils.datasets.transforms
    import TV3S.utils.datasets.dataset_pipelines
    print("TV3S components imported successfully")
except ImportError as e:
    print(f"Warning: TV3S components not available: {e}")
    print("They will be imported in the training script")

# Model configuration with TV3S temporal head
model = dict(
    type='EncoderDecoder_clips',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth')
    ),
    decode_head=dict(
        type='TV3SHead_shift_city',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Dataset configuration for Cityscapes video data
dataset_type = 'CityscapesDataset_clips'
data_root = 'dataset'  # Path to your dataset folder
img_dir = 'leftImg8bit_trainvaltest'  # Folder containing video frames
ann_dir = 'gtFine'  # Folder containing annotations
crop_size = (1024, 1024)

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Training pipeline for video clips
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='PackSegInputs')
]

# Test pipeline for video clips
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Data configuration
data = dict(
    samples_per_gpu=1,  # Smaller batch size for video clips
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline,
        dilation=[-9, -6, -3],  # Frame offsets for 9-frame temporal clip
        istraining=True,
        mamba_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        istraining=False,
        mamba_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        istraining=False,
        mamba_mode=False,
    )
)

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=160,
    val_interval=10,
)

# Evaluation configuration
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Optimizer configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# Logging and checkpoint configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=10)
)
