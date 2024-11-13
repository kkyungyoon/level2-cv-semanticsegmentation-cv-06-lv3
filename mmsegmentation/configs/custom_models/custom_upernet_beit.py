# configs/custom_models/custom_upernet_beit.py

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]
    ),
    neck=dict(
        embed_dim=1024, 
        rescales=[4, 2, 1, 0.5]
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=29,  # 클래스 수 수정
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',  # 손실 함수 변경
            use_sigmoid=True,          # 시그모이드 활성화 사용
            loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=29,  # 클래스 수 수정
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',  # 손실 함수 변경
            use_sigmoid=True,          # 시그모이드 활성화 사용
            loss_weight=0.4
        )
    ),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426))
)