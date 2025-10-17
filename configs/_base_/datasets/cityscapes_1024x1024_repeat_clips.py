# dataset settings (LOW-RES pretraining)
dataset_type = 'CityscapesDataset_clips'
data_root = 'dataset_preprocessed'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (128, 256)

# Cityscapes class names
used_labels = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

# --------------------------
# Training pipeline
# --------------------------
train_pipeline = [
    dict(type='LoadImageFromFile_clips'),        # clip-safe loader
    dict(type='LoadAnnotations'),                # fine, works on masks
    dict(type='Resize_clips', img_scale=(512, 256), keep_ratio=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# --------------------------
# Validation / Test pipeline
# --------------------------
test_pipeline = [
    dict(type='LoadImageFromFile_clips'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 256),
        flip=False,
        transforms=[
            dict(type='Resize_clips', keep_ratio=True),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

# --------------------------
# Dataset config
# --------------------------
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit_trainvaltest/train',
            ann_dir='gtFine/train',
            pipeline=train_pipeline,
            used_labels=used_labels,
            dilation=[-9, -6, -3],
            istraining=True,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline,
        used_labels=used_labels,
        dilation=[-9, -6, -3],
        istraining=False
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline,
        used_labels=used_labels,
        dilation=[-9, -6, -3],
        istraining=False
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
