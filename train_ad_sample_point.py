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

import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
from random import randint
from utils.loss_utils import l1_loss, ssim, get_sim, get_conloss, sigmoid_rampup
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from decoder.dmodel import decoder, init_weights_to_one
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # testing_iterations = [x * 1000 for x in range(61)]  # 31
    testing_iterations = [1,2,3,4,5,10,50] + [x * 100 for x in range(301)]  # 31
    saving_iterations  = [1000,5000,10000,15000,30000]  # 31
    checkpoint_iterations = [1000, 15000,30000]  # 31

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
        img_decoder = decoder(encoder_hidden_dims=24, decoder_hidden_dims=512).cuda()
        img_decoder.apply(init_weights_to_one)
        optimizer_decoder = torch.optim.Adam(img_decoder.parameters(), lr=0.0025)  # 0.00025
        scheduler = lr_scheduler.StepLR(optimizer_decoder, step_size=500, gamma=0.1)
        # scheduler = lr_scheduler.MultiStepLR(optimizer_decoder, milestones=[x * 1000 for x in range(31)], gamma=0.1)
        # optimizer_decoder = torch.optim.Adam(img_decoder.parameters(), lr=0.0025, weight_decay=1e-3) #0.00025
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
    # first_iter = 0
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        if opt.include_feature:
            img_decoder.train()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        image, language_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["language_feature_image"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        if opt.include_feature:
            gt_language_feature, language_feature_mask, seg_map = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level,retain_seg= True)
            con_loss = get_conloss(language_feature, seg_map)
            # Ll1_24 = l1_loss(language_feature * language_feature_mask, gt_language_feature[:24] * language_feature_mask)
            # language_feature = img_decoder(language_feature)
            language_feature = img_decoder(language_feature)
            Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)
            # loss = Ll1
            loss = Ll1 + 0.05 * con_loss *  sigmoid_rampup(iteration//30, 30)
            con_loss1= 0.05 * con_loss * sigmoid_rampup(iteration//30, 30)
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # print((gt_language_feature*language_feature_mask).max())
        # print((language_feature[:24] * language_feature_mask).max())
        loss.backward()
        if opt.include_feature:
            optimizer_decoder.step()
            scheduler.step()
            optimizer_decoder.zero_grad()
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, opt.include_feature,iteration, Ll1, loss, l1_loss, con_loss, con_loss1, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt),\
                            img_decoder, dataset.lf_path,dataset.feature_level)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature, opt.use_triplane), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if opt.include_feature:
                    torch.save(img_decoder, scene.model_path + "/512decoder_chkpnt" + str(iteration) + ".pth")
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, include_feature, iteration, Ll1, loss, l1_loss, conloss, con_loss1,elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, img_decoder, lf_path= None, feature_level = 0,):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/con_loss_o', conloss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/con_loss_1', con_loss1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if include_feature:
                        no_decode_img = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["language_feature_image"], 0.0, 1.0)[21:24]
                        image = torch.clamp(img_decoder(renderFunc(viewpoint, scene.gaussians, *renderArgs)["language_feature_image"]) * 10, 0.0, 1.0)[21:24]
                        # gt_image, mask = \
                        #     viewpoint.get_language_feature(language_feature_dir="/home/zhongyao/dl/LangSplat/data/preprocessed_dataset/sofa/language_features_backup", feature_level=3)
                        gt_language_feature, language_feature_mask = viewpoint.get_language_feature(
                            language_feature_dir=lf_path, feature_level=feature_level)
                        gt_image = torch.clamp(gt_language_feature.to("cuda")* 10, 0.0, 1.0)[21:24]
                    else:
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    #language feature
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["language_feature_image"], 0.0, 1.0)
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # gt_image, mask = \
                    #     viewpoint.get_language_feature(language_feature_dir="/home/zhongyao/dl/LangSplat/data/preprocessed_dataset/sofa/language_features_backup", feature_level=3)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_noad".format(viewpoint.image_name),
                                             no_decode_img[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.2")
    parser.add_argument('--port', type=int, default=55556)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1,7000,15000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1,7000,15000,30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1,7000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
