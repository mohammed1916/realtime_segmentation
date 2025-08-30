_base_ = [
    'c:/Users/abd/d/ai/seg/segformer_pytorch/configs/_base_/models/segformer_mit-b0.py',
    'c:/Users/abd/d/ai/seg/segformer_pytorch/configs/_base_/datasets/cityscapes.py',
    'c:/Users/abd/d/ai/seg/segformer_pytorch/configs/_base_/default_runtime.py',
    'c:/Users/abd/d/ai/seg/segformer_pytorch/configs/_base_/schedules/schedule_160k.py'
]

# model settings for video segmentation
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        ),
    decode_head=dict(
        type='TV3SHead_shift_city',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dim=256,
            window_w=20,
            window_h=20,
            shift_size=10,
            real_shift=True,
            model_type=0,
            mamba2=False,
            val_mode=2,
            test_mode=False
        ),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
