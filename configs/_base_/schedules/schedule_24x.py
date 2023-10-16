# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
max_lr = 1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))

# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    warmup_iters=1600 * 2,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=24, interval=1)
evaluation = dict(by_epoch=True, interval=2, pre_eval=True)