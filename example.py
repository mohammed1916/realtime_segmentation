find_unused_parameters = True
model = dict(
    backbone=dict(style='pytorch', type='IMTRv21_5'),
    decode_head=dict(
        align_corners=False,
        channels=128,
        decoder_params=dict(
            embed_dim=256,
            mamba2=False,
            model_type=0,
            real_shift=True,
            shift_size=10,
            test_mode=False,
            val_mode=2,
            window_h=20,
            window_w=20),
        dropout_ratio=0.1,
        feature_strides=[
            4,
            8,
            16,
            32,
        ],
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=19,
        num_clips=4,
        type='TV3SHead_shift_city'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder_clips')
norm_cfg = dict(requires_grad=True, type='SyncBN')
