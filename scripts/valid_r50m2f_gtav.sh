#!/usr/bin/env bash
echo "Running inference on" ${1}
      #   --image_in \
     python -m torch.distributed.launch --nproc_per_node=1 --master_port=29844 valid.py \
        --val_dataset cityscapes bdd100k mapillary \
        --arch network.mask2former.R50M2F \
        --style_elimination \
        --date 0101 \
        --bs_mult_val 6 \
        --exp valid_r50m2f_gtav \
        --snapshot ${1}