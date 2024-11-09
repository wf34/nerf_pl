#!/usr/bin/bash

python eval.py \
   --root_dir /home/wf34/projects/nerf_pl/lego \
   --dataset_name blender --scene_name test_lego \
   --img_wh 800 800 --N_importance 64 \
   --ckpt_path /home/wf34/projects/nerf_pl/ckpts/testing_train_run/epoch=0.ckpt

