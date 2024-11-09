#!/usr/bin/bash

python train.py \
   --dataset_name llff \
   --root_dir /home/wf34/projects/nerf_pl/nerf_llff_data/fern \
   --N_importance 64 --img_wh 4032 3024 --noise_std 0 \
   --num_epochs 24 --batch_size 64 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name fern_train_run
