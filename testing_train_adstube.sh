#!/usr/bin/bash

python train.py \
   --dataset_name llff \
   --root_dir /home/wf34/projects/nerf_pl/ads-tube --spheric_poses \
   --N_importance 64 --img_wh 180 320 --noise_std 0 \
   --num_epochs 24 --batch_size 2048 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --exp_name ads-tube-fw
