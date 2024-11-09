#!/usr/bin/bash

python train.py \
   --dataset_name llff \
   --root_dir /home/wf34/projects/nerf_pl/horsy-few --spheric_poses \
   --N_importance 64 --img_wh 480 270 --noise_std 0 \
   --num_epochs 24 --batch_size 4096 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name horsyfew_run
