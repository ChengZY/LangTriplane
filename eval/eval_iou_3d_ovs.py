from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result


def eval_iou(sem_map,
                    image,
                    clip_model,
                    annotation_path,
                    image_name: Path = None,
                    thresh: float = 0.5,
                    colormap_options=None):
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    label_iou = []

    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 15 #30
            kernel = np.ones((scale, scale)) / (scale ** 2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel) #cv2.filter2D 用于检测图像中的特征，如边缘、纹理等
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)

            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            # mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8) #2D mask
            mask_gt = cv2.imread(os.path.join(annotation_path, clip_model.positives[k]+'.png')).astype(np.uint8) #read
            # mask_gt = cv2.resize(mask_gt, (mask_pred.shape[0],mask_pred.shape[1]))[:,:,0]
            mask_gt = cv2.resize(mask_gt, (mask_pred.shape[1], mask_pred.shape[0]))[:, :, 0]
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list

if __name__ == '__main__':
    parser = ArgumentParser(description="prompt any label")
    # parser.add_argument("--image_dir", type=str, default='../data/3d_ovs_8/lawn/images/', help="path to image")
    # parser.add_argument('--annotation_dir', type=str, default="../data/3d_ovs_8/lawn/segmentations/")
    # parser.add_argument('--feat_dir', type=str, default="../output/3d_ovs_8/lawn/retrain_lang_8_0/train/ours_None/renders_npy", help="path to predicted results")
    # parser.add_argument("--output_dir", type=str, default="../output/3d_ovs_8/lawn/eval", help="path to save the results")
    # parser.add_argument("--mask_thresh", type=float, default=0.4)
    # parser.add_argument("--scene", type=float, default=0.4)

    parser.add_argument("--image_dir", type=str, default='../data/3d_ovs_8/', help="path to image")
    parser.add_argument('--annotation_dir', type=str, default="../data/3d_ovs_8/")
    parser.add_argument('--feat_dir', type=str, default="../output/3d_ovs_8/", help="path to predicted results")
    parser.add_argument("--output_dir", type=str, default="../output/3d_ovs_8/", help="path to save the results")
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--scene", type=str, default='snacks')


    args = parser.parse_args()
    scene = args.scene
    image_path = os.path.join(args.image_dir,args.scene,'images')
    annotation_path= os.path.join(args.annotation_dir,args.scene,'segmentations')
    output_path = os.path.join(args.output_dir,args.scene,'eval_tp')  #args.output_dir
    print("begin to eval scene {}".format(args.scene))

    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    mask_thresh = 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''name of testing images [1,24,42,106 ...]'''
    # eval_index_list = [2,4,10,15,22]
    eval_index_list = [x for x in os.listdir(annotation_path) if os.path.isdir(os.path.join(annotation_path, x))]
    # feat_dir = ['../output/3d_ovs_8/{}/retrain_lang_8_0/train/ours_None/renders_npy'.format(args.scene), \
    #             '../output/3d_ovs_8/{}/retrain_lang_8_1/train/ours_None/renders_npy'.format(args.scene), \
    #             '../output/3d_ovs_8/{}/retrain_lang_8_2/train/ours_None/renders_npy'.format(args.scene), \
    #             '../output/3d_ovs_8/{}/retrain_lang_8_3/train/ours_None/renders_npy'.format(args.scene),
    #             ]
    feat_dir = ['../output/3d_ovs_8/{}/retrain_lang_8_tp_0/train/ours_None/renders_npy'.format(args.scene), \
                '../output/3d_ovs_8/{}/retrain_lang_8_tp_1/train/ours_None/renders_npy'.format(args.scene), \
                '../output/3d_ovs_8/{}/retrain_lang_8_tp_2/train/ours_None/renders_npy'.format(args.scene), \
                '../output/3d_ovs_8/{}/retrain_lang_8_tp_3/train/ours_None/renders_npy'.format(args.scene),
                ]
    # compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    # compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), 384, 512, 512), dtype=np.float32)       #sofa 384, 512
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), 378, 504, 512), dtype=np.float32)  # lawn/lawn 378, 504
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[int(idx)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    pd_chosen_iou_all = []
    acc_num = 0

    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / idx
        image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(os.path.join(image_path, str(idx).zfill(2)+'.jpg' ))[..., ::-1]  # original img
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        lvl, h, w, _ = sem_feat.shape
        restored_feat = sem_feat.view(lvl, h, w, -1)

        mask_path= os.path.join(annotation_path,str(idx).zfill(2))
        label_list = [x.split('.png')[0] for x in os.listdir(mask_path)]
        clip_model.set_positives(label_list)

        c_iou_list, c_lvl = eval_iou(restored_feat, rgb_img, clip_model, mask_path,image_name,
                                            thresh=mask_thresh, colormap_options=colormap_options)
        chosen_iou_all.extend(c_iou_list)
        pd_chosen_iou_all.append(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    pd_chosen_iou_all = pd.DataFrame(pd_chosen_iou_all, columns = label_list)
    # pd_chosen_iou_all.to_csv('../output/3d_ovs_8/lawn/eval/iou.csv', index=False)
    # pd_chosen_iou_all.to_csv('../output/3d_ovs_8/{}/eval/iou.csv'.format(args.scene), index=False)
    # print("mean IoU", np.array(chosen_iou_all).mean())
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    print('{} mean iou: {}'.format(args.scene,mean_iou_chosen))
    # print("done compressed")
