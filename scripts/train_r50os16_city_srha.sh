#!/usr/bin/env bash
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=29116 train.py \
        --dataset cityscapes \
        --val_dataset bdd100k synthia gtav \
        --arch network.deepv3.DeepR50V3PlusD \
        --style_elimination \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.005 \
        --decoder_lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 120000 \
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --date 0101 \
        --exp train_r50os16_city_srha \
        --freeze_layer0 \
        --srm \
        --ga_layer layer1 layer2 layer3 layer4 \
        --ga_weight 0.4 0.6 0.8 1.0 \
        --ra_layer layer1 layer2 layer3 layer4 \
        --ra_weight 0.4 0.6 0.8 1.0 \
        --la_layer layer1 layer2 layer3 layer4 \
        --la_weight 0.4 0.6 0.8 1.0 \
        --pc_weight 10.0 \
        --alpha 1 \
        --ckpt ./logs/ \
        --tb_path ./logs/ 


