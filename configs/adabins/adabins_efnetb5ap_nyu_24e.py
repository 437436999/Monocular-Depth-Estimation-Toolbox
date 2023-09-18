_base_ = [
    '../_base_/models/adabins.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=10,
        norm_cfg=norm_cfg),
    )

find_unused_parameters=True
# SyncBN=True

# optimizer
max_lr=0.000357
optimizer = dict(
    type='AdamW', 
    lr=max_lr, 
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys={
            'decode_head': dict(lr_mult=10), # x10 lr
        }))

# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
momentum_config = dict(
    policy='OneCycle'
)

# runtime
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# evaluation = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=20, interval=1)
evaluation = dict(by_epoch=True, interval=0.5, pre_eval=True)


# dataset settings
dataset_type = 'NYUDataset'
data_root = 'data/nyu/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(416, 544)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')),
        ])
]

# for visualization of pc
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.0), # set to zero
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', 
         keys=['img'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_train.txt',
        pipeline=train_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10))

