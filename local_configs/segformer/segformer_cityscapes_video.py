# =====================
# Runtime (from default_runtime.py)
# =====================
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')

# =====================
# Optimizer (from schedule_160k.py)
# =====================
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# =====================
# Norm + Preprocessor (from segformer_mit-b0.py)
# =====================
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# =====================
# Model configuration (custom override)
# =====================
model = dict(
    type='EncoderDecoder_clips',
    data_preprocessor=data_preprocessor,
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth')
    ),
    decode_head=dict(
        type='TV3SHead_shift_city',
        in_channels=[32, 64, 160, 256],
        feature_strides=[4, 8, 16, 32],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    val_cfg=dict(),
    test_cfg=dict(mode='whole')
)
val_cfg=dict()
test_cfg=dict()
# =====================
# Dataset configuration
# =====================
dataset_type = 'CityscapesDataset_clips'
data_root = 'dataset_preprocessed'
img_dir = 'leftImg8bit_trainvaltest'
ann_dir = 'gtFine'
crop_size = (128, 256)

# Normalization config used by Normalize_clips below
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    # Collect → PackSegInputs
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor')
    )
]

test_pipeline = [
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),  # same as train
    # dict(type='RandomFlip_clips', prob=0.5),  # optional, usually False for test
    # dict(type='PhotoMetricDistortion_clips'), # optional, usually disabled for test
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor')
    )
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline,
        dilation=[-9, -6, -3],
        istraining=True,
        # mamba_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        istraining=False,
        # mamba_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        istraining=False,
        # mamba_mode=False,
    )
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=data['train']
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=data['val']
)

test_dataloader = val_dataloader

# =====================
# Training configuration (override)
# =====================
train_cfg = dict(
    # _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=16000,
    val_interval=100,
)

# =====================
# Evaluation
# =====================
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    num_classes=19,
    ignore_index=255
)
test_evaluator = val_evaluator

# =====================
# LR Scheduler (override)
# =====================
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False)
]

# =====================
# Hooks (override)
# =====================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1600, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=500)
)
