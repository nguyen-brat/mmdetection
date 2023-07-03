_base_ = '../htc/htc-without-semantic_r50_fpn_1x_coco.py'

# 1. Dataset settings
dataset_type = 'CocoDataset'
classes = ('human', 'ball')
data_root = "Data/vipriors-segmentation-data-2022"
image_size = (1624,1234)

load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=[[dict(type='AutoContrast')], [dict(type='Equalize')],
                                        [dict(type='Invert')], [dict(type='Rotate')],
                                        [dict(type='Posterize')], [dict(type='Solarize')],
                                        [dict(type='SolarizeAdd')], [dict(type='Color')],
                                        [dict(type='Contrast')], [dict(type='Brightness')],
                                        [dict(type='Sharpness')], [dict(type='ShearX')],
                                        [dict(type='ShearY')], [dict(type='TranslateX')],
                                        [dict(type='TranslateY')]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size=image_size)
  ]
  
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=10),
    dict(type='PackDetInputs')
  ] 

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=3,
    num_workers=1,
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
          type=dataset_type,
          data_root=data_root,
          ann_file='train.json',
          data_prefix=dict(img='train/'),
          metainfo=dict(classes=classes),
          filter_cfg=None,
          pipeline=load_pipeline),
        pipeline=train_pipeline 
      )
    )
  
val_dataloader = dict(
    batch_size=3,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(_delete_=True, img='val/'),
        metainfo=dict(classes=classes),
        filter_cfg=None,
        pipeline=test_pipeline)
    )

test_dataloader = dict(
    batch_size=3,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(_delete_=True, img='test/'),
        metainfo=dict(classes=classes),
        filter_cfg=None,
        pipeline=test_pipeline)
    )

# 2. Evaluator settings
val_evaluator = dict(
    ann_file = data_root + 'val.json',
    metric = ['segm']
    )

test_evaluator = dict(
    format_only=True,
    ann_file=data_root + 'test.json',
    metric = ['segm'],
    outfile_prefix='./work_dirs/htc_custom_dataset/submission'
    )

# 3. Model settings
model = dict(
    backbone = dict(
        init_cfg = None
        )
    )

_base_.model.roi_head.bbox_head[0].num_classes = 2
_base_.model.roi_head.bbox_head[1].num_classes = 2
_base_.model.roi_head.bbox_head[2].num_classes = 2

_base_.model.roi_head.mask_head[0].num_classes = 2
_base_.model.roi_head.mask_head[1].num_classes = 2
_base_.model.roi_head.mask_head[2].num_classes = 2

# 4. Trainning settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=10)

# 5. Optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
  )

# 6. Learning Rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[35, 50, 55],
        gamma=0.5)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
