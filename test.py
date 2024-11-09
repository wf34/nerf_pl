#!/usr/bin/env python3
import torch
from tap import Tap

from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import time

from models.rendering import *
from models.nerf import *

import metrics

from datasets import dataset_dict
# from datasets.llff import *

torch.backends.cudnn.benchmark = True

N_samples = 64
N_importance = 64
use_disp = False
chunk = 1024*32*4

class TestArgs(Tap):
    N_emb_xyz:int = 10
    N_emb_dir:int = 4


@torch.no_grad()
def f(rays, models, embeddings, dataset):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def test(args):
    img_wh = (200, 200)
    
    # dataset = dataset_dict['llff'] \
    #           ('/home/ubuntu/data/nerf_example_data/my/silica4/', 'test_train', spheric_poses=True,
    #            img_wh=img_wh)
    
    dataset = dataset_dict['blender']('/home/wf34/projects/nerf_pl/lego2/', 'test', img_wh=img_wh)

    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    
    nerf_coarse = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3)
    nerf_fine = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3)
    
    ckpt_path = 'ckpts/testing_train_run2/epoch=15.ckpt'
    
    load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')
    
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()
    
    models = {'coarse': nerf_coarse, 'fine': nerf_fine}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    sample = dataset[0]
    rays = sample['rays'].cuda()
    
    t = time.time()
    results = f(rays, models, embeddings, dataset)
    torch.cuda.synchronize()
    
    print('one pass time:', time.time()-t)
    
    img_gt = sample['rgbs'].view(img_wh[1], img_wh[0], 3)
    img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
    alpha_pred = results['opacity_fine'].view(img_wh[1], img_wh[0]).cpu().numpy()
    depth_pred = results['depth_fine'].view(img_wh[1], img_wh[0])
    
    print('PSNR', metrics.psnr(img_gt, img_pred).item())
    
    plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplot(221)
    plt.title('GT')
    plt.imshow(img_gt)
    plt.subplot(222)
    plt.title('pred')
    plt.imshow(img_pred)
    plt.subplot(223)
    plt.title('depth')
    plt.imshow(visualize_depth(depth_pred).permute(1,2,0))
    plt.savefig('logs/testing_train_run2/test.png')


if '__main__' == __name__:
    args = TestArgs().parse_args()
    test(args)
