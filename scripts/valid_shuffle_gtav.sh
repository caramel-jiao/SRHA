#!/usr/bin/env bash
echo "Running inference on" ${1}
      #   --image_in \
     python -m torch.distributed.launch --nproc_per_node=1 --master_port=29165 valid.py \
        --val_dataset cityscapes bdd100k mapillary \
        --arch network.deepv3.DeepShuffleNetV3PlusD \
        --style_elimination \
        --date 0101 \
        --bs_mult_val 12 \
        --exp valid_shuffle_gtav \
        --snapshot ${1}