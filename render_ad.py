#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, source_path, name, iteration, views, gaussians, dmodel, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if args.include_feature:
            rendering = output["language_feature_image"]
            rendering = dmodel(rendering)
            # rendering = output["render"]
        else:
            # rendering = output["language_feature_image"]
            rendering = output["render"]
            
        if args.include_feature:
            # gt = view.original_image[0:3, :, :]
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)
        else:
            gt = view.original_image[0:3, :, :]
            # gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)

        language_feature_img = rendering.permute(1,2,0).cpu().numpy()[:,:,:3]
        # np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"), language_feature_img)
        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        # torchvision.utils.save_image((1/gt[3:6].max()) * rendering[0:3], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image((1/gt[3:6].max()) * gt[0:3], os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        #512
        torchvision.utils.save_image((1/gt[3*(idx):3*(idx+1)].max()) * rendering[3*(idx):3*(idx+1)], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image((1/gt[3*(idx):3*(idx+1)].max()) * gt[3*(idx):3*(idx+1)], os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # render_vis = torch.stack([torch.mean(rendering[0:8], 0),torch.mean(rendering[8:16], 0),torch.mean(rendering[16:], 0)])
        # gt_vis     = torch.stack([torch.mean(gt[0:8], 0), torch.mean(gt[8:16], 0), torch.mean(gt[16:], 0)])
        # torchvision.utils.save_image( (1/gt_vis.max()) * render_vis, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image( (1/gt_vis.max()) * gt_vis, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        # checkpoint = os.path.join("/home/zhongyao/dl/LangSplat/output/sofa_retrain_lan_3/", 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        dcheckpoint = os.path.join(args.model_path, '512decoder_chkpnt30000.pth')
        dmodel = torch.load(dcheckpoint)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             # render_set(dataset.model_path,dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(),gaussians, dmodel,pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, dmodel,pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)