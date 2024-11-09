#!/usr/bin/env python3

from tap import Tap
from typing import Literal

import torch
from collections import defaultdict
import numpy as np
import mcubes

import trimesh
import trimesh.exchange

from models.rendering import *
from models.nerf import *

from datasets import dataset_dict

from utils import load_ckpt

STEP_ONE = 'first'
STEP_TWO = 'second'

class MeshExporter(Tap):
    step: Literal[STEP_ONE, STEP_TWO]
    N_emb_xyz:int = 10
    N_emb_dir:int = 4
    model_destination: str

def export_rough(args):
    pass
    # Change here #
    img_wh = (200, 200) # full resolution of the input images
    dataset_name = 'blender' # blender or llff (own data)
    scene_name = 'lego2-mesh' # whatever you want
    root_dir = '/home/wf34/projects/nerf_pl/lego2/' # the folder containing data
    ckpt_path = '/home/wf34/projects/nerf_pl/ckpts/testing_train_run2/epoch=15.ckpt' # the model path
    ###############
    
    kwargs = {'root_dir': root_dir,
              'img_wh': img_wh}
    if dataset_name == 'llff':
        kwargs['spheric_poses'] = True
        kwargs['split'] = 'test'
    else:
        kwargs['split'] = 'train'
        
    chunk = 1024*32
    dataset = dataset_dict[dataset_name](**kwargs)
    
    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    
    nerf_fine = NeRF(in_channels_xyz=6*args.N_emb_xyz+3, in_channels_dir=6*args.N_emb_dir+3)
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval();

    # until here is was just loading
    ### Tune these parameters until the whole object lies tightly in range with little noise ###
    N = 128 # controls the resolution, set this number small here because we're only finding
            # good ranges here, not yet for mesh reconstruction; we can set this number high
            # when it comes to final reconstruction.
    xmin, xmax = -1.2, 1.2 # left/right range
    ymin, ymax = -1.2, 1.2 # forward/backward range
    zmin, zmax = -1.2, 1.2 # up/down range

    ## Attention! the ranges MUST have the same length!
    sigma_threshold = 10. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
    ############################################################################################
    
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)
    
    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # (N, embed_xyz_channels)
            dir_embedded = embedding_dir(dir_[i:i+chunk]) # (N, embed_dir_channels)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
            out_chunks += [nerf_fine(xyzdir_embedded)]
        rgbsigma = torch.cat(out_chunks, 0)
        
    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0)
    sigma = sigma.reshape(N, N, N)
    
    # The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
    print('lengths: ', len(vertices), len(triangles))
    mesh = trimesh.Trimesh(vertices/N, triangles)

    with open(args.model_destination, 'wb') as the_file:
        the_file.write(trimesh.exchange.ply.export_ply(mesh, encoding='ascii'))


if __name__ == '__main__':
    args = MeshExporter().parse_args()
    if STEP_ONE == args.step:
        export_rough(args)
    elif STEP_TWO == args.step:
        export_complete(args)
