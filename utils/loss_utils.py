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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from .supconloss import SupConLoss
import numpy as np
import random

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
def conloss(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss
    """
    dist = torch.nn.functional.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss

def get_conloss0(language_feature, seg_map):
    language_feature[torch.isnan(language_feature)] = 0
    unique_values, counts = torch.unique(seg_map.flatten(), return_counts=True)
    # most_frequent_value = unique_values[torch.argmax(counts)]
    point_features_1 = []
    point_features_2 = []
    point_features_3 = []
    labels = []

    for value in unique_values:
        if value !=-1:
            mask = seg_map == value
            indices = torch.nonzero(mask, as_tuple=False)
            point_feature_1 = language_feature[:, indices[indices.shape[0] // 2, :][1], indices[indices.shape[0] // 2, :][2]]
            point_feature_1 = point_feature_1.reshape(1, 512)
            point_feature_2 = language_feature[:, indices[0, :][1], indices[0, :][2]]
            point_feature_2 = point_feature_2.reshape(1, 512)
            point_feature_3 = language_feature[:, indices[-1, :][1], indices[-1, :][2]]
            point_feature_3 = point_feature_3.reshape(1, 512)
            labels.append(value)
            point_features_1.append(point_feature_1)
            point_features_2.append(point_feature_2)
            point_features_3.append(point_feature_3)

    criterion = SupConLoss(temperature=0.07)

    point_features_1 = torch.stack(point_features_1, dim=0)
    point_features_2 = torch.stack(point_features_2, dim=0)
    point_features_3 = torch.stack(point_features_3, dim=0)
    point_features = torch.cat([point_features_1, point_features_2, point_features_3], dim=1)

    label1 = torch.tensor(labels)
    loss = criterion(point_features,label1)
    # print("con loss:", loss)
    return loss

def get_conloss0_random(language_feature, seg_map):
    language_feature[torch.isnan(language_feature)] = 0
    unique_values, counts = torch.unique(seg_map.flatten(), return_counts=True)
    # most_frequent_value = unique_values[torch.argmax(counts)]
    point_features_1 = []
    point_features_2 = []
    point_features_3 = []
    labels = []

    for value in unique_values:
        if value !=-1:
            mask = seg_map == value
            indices = torch.nonzero(mask, as_tuple=False)
            random_numbers = torch.randint(0, indices.shape[0], (3,))
            point_feature_1 = language_feature[:, indices[random_numbers[0], :][1], indices[random_numbers[0], :][2]]
            point_feature_1 = point_feature_1.reshape(1, 512)
            point_feature_2 = language_feature[:, indices[random_numbers[1], :][1], indices[random_numbers[1], :][2]]
            point_feature_2 = point_feature_2.reshape(1, 512)
            point_feature_3 = language_feature[:, indices[random_numbers[2], :][1], indices[random_numbers[2], :][2]]
            point_feature_3 = point_feature_3.reshape(1, 512)
            labels.append(value)
            point_features_1.append(point_feature_1)
            point_features_2.append(point_feature_2)
            point_features_3.append(point_feature_3)

    criterion = SupConLoss(temperature=0.7)

    point_features_1 = torch.stack(point_features_1, dim=0)
    point_features_2 = torch.stack(point_features_2, dim=0)
    point_features_3 = torch.stack(point_features_3, dim=0)
    point_features = torch.cat([point_features_1, point_features_2, point_features_3], dim=1)

    label1 = torch.tensor(labels)
    loss = criterion(point_features,label1)
    # print("con loss:", loss)
    return loss

def get_conloss1(language_feature, seg_map):
    language_feature[torch.isnan(language_feature)] = 0
    unique_values, counts = torch.unique(seg_map.flatten(), return_counts=True)
    # most_frequent_value = unique_values[torch.argmax(counts)]
    point_features_1 = []
    point_features_2 = []
    point_features_3 = []
    labels = []
    for value in unique_values:
        if value !=-1:
            mask = seg_map == value
            indices = torch.nonzero(mask, as_tuple=False)
            point_feature_1 = language_feature[:, indices[indices.shape[0] // 2, :][1], indices[indices.shape[0] // 2, :][2]]
            point_feature_1 = point_feature_1.reshape(1, 512)
            point_feature_2 = language_feature[:, indices[0, :][1], indices[0, :][2]]
            point_feature_2 = point_feature_2.reshape(1, 512)
            point_feature_3 = language_feature[:, indices[-1, :][1], indices[-1, :][2]]
            point_feature_3 = point_feature_3.reshape(1, 512)
            labels.append(value)
            point_features_1.append(point_feature_1)
            point_features_2.append(point_feature_2)
            point_features_3.append(point_feature_3)

    criterion = SupConLoss(temperature=0.07)

    point_features_1 = torch.stack(point_features_1, dim=0)
    point_features_2 = torch.stack(point_features_2, dim=0)
    point_features_3 = torch.stack(point_features_3, dim=0)
    point_features = torch.cat([point_features_1, point_features_2, point_features_3], dim=0)

    label1 = torch.tensor(labels)
    label1 = torch.cat([label1, label1, label1], dim=0)
    loss = criterion(point_features,label1)
    # print("con loss:", loss)
    return loss


def get_conloss(language_feature, seg_map):
    # language_feature[torch.isnan(language_feature)] = 0
    language_feature = language_feature.clone()
    language_feature[torch.isnan(language_feature)] = 0
    unique_values, counts = torch.unique(seg_map.flatten(), return_counts=True)
    # most_frequent_value = unique_values[torch.argmax(counts)]
    point_features_1 = []
    point_features_2 = []
    point_features_3 = []
    labels = []

    language_feature_flattened = language_feature.view(language_feature.shape[0], -1)
    seg_map_flatten = seg_map.view(1, -1)

    ind = []
    criterion = SupConLoss(temperature=0.07)
    con_loss = 0
    for value in unique_values:
        # if value != -1:
        mask = seg_map_flatten == value
        indices = torch.nonzero(mask, as_tuple=False)
        ind.append(indices)
        cls_feature = language_feature_flattened[:, indices[:, 1]]

        point_feature_1 = cls_feature.mean(dim=1).reshape(1, language_feature.shape[0])
        if indices.shape[0] > 1:
            point_feature_2 = cls_feature[:, :indices.shape[0] // 2].mean(dim=1).reshape(1, language_feature.shape[0])
            point_feature_3 = cls_feature[:, indices.shape[0] // 2:].mean(dim=1).reshape(1, language_feature.shape[0])
        else:
            point_feature_2 = point_feature_1
            point_feature_3 = point_feature_1
        labels.append(value)
        point_features_1.append(point_feature_1)
        point_features_2.append(point_feature_2)
        point_features_3.append(point_feature_3)
        # loss = criterion(torch.stack([point_feature_1, point_feature_2, point_feature_3], dim=0),torch.tensor([value]*3))
        # con_loss = con_loss + loss

    # con_loss = con_loss/len(unique_values)
    point_features_1 = torch.stack(point_features_1, dim=0)
    point_features_2 = torch.stack(point_features_2, dim=0)
    point_features_3 = torch.stack(point_features_3, dim=0)
    point_features = torch.cat([point_features_1, point_features_2, point_features_3], dim=1)

    label1 = torch.tensor(labels)
    loss = criterion(point_features,label1)
    # print("con loss:", loss)
    return loss

# def get_conloss(language_feature, seg_map):
#     language_feature[torch.isnan(language_feature)] = 0
#     unique_values, counts = torch.unique(seg_map.flatten(), return_counts=True)
#     # most_frequent_value = unique_values[torch.argmax(counts)]
#     point_features_1 = []
#     point_features_2 = []
#     point_features_3 = []
#     labels = []
#
#     language_feature_flattened = language_feature.view(512, -1)
#     seg_map_flatten = seg_map.view(1, -1)
#
#     for value in unique_values:
#         if value !=-1:
#             mask = seg_map_flatten == value
#             indices = torch.nonzero(mask, as_tuple=False)
#             a = indices[:, 1]
#             cls_feature = language_feature_flattened[:, indices[:, 1]]
#
#             point_feature_1 = cls_feature.mean(dim=1)
#             point_feature_2 = cls_feature[:,:cls_feature.shape[1]//2].mean(dim=1)
#             point_feature_3 = cls_feature[:,:cls_feature.shape[1]//2:].mean(dim=1)
#             labels.append(value)
#             point_features_1.append(point_feature_1)
#             point_features_2.append(point_feature_2)
#             point_features_3.append(point_feature_3)
#     criterion = SupConLoss(temperature=0.07)
#     point_features_1 = torch.stack(point_features_1, dim=0)
#     point_features_2 = torch.stack(point_features_2, dim=0)
#     point_features_3 = torch.stack(point_features_3, dim=0)
#     point_features = torch.cat([point_features_1, point_features_2, point_features_3], dim=0)
#
#     label1 = torch.tensor(labels)
#     label1 = torch.cat([label1, label1, label1], dim=0)
#     loss = criterion(point_features,label1)
#     # print("con loss:", loss)
#     return loss

def get_sim(language_feature, seg_map):
    language_feature[torch.isnan(language_feature)] = 0
    unique_values, counts = torch.unique(seg_map.flatten(), return_counts=True)
    most_frequent_value = unique_values[torch.argmax(counts)]
    mask = seg_map == most_frequent_value
    language_feature = language_feature * mask
    language_feature_mean = torch.mean(language_feature, dim=0, keepdim=True)
    language_feature_valid = language_feature[language_feature_mean != 0]
    point = language_feature_valid[language_feature_valid.shape[0]//2]


    # masked_pixels = language_feature[mask.repeat(512, 1, 1)]
    # indices = torch.nonzero(mask, as_tuple=False)
    # perm = torch.randperm(indices.size(0))
    # selected_indices = indices[perm[:2]]
    # point1 = language_feature[:, selected_indices[0][1], selected_indices[0][2]]
    # point2 = language_feature[:, selected_indices[1][1], selected_indices[1][2]]
    # sim = F.kl_div(point1, point2, reduction='mean')

    language_feature_mean = torch.mean(language_feature, dim=0, keepdim=True)
    if language_feature_mean.sum()==0:
        return 0

    min_value = torch.min(language_feature_mean[language_feature_mean != 0])
    max_value = torch.max(language_feature_mean[language_feature_mean != 0])
    if min_value == max_value:
        sim = 0
    else:
        min_index = torch.where(language_feature_mean == min_value)
        max_index = torch.where(language_feature_mean == max_value)
        point1 = language_feature[:, min_index[1][0], min_index[2][0]]
        point2 = language_feature[:, max_index[1][0], max_index[2][0]]
        # sim = torch.abs(F.kl_div(point1, point2, reduction='mean')) + 1e-6
        sim = torch.abs((point1 - point2)).mean() + 1e-6

    return sim
def get_kl0(network_output, gt):

    # gt = gt - torch.mean(gt, dim=-1, keepdim=True)
    # network_output = network_output - torch.mean(network_output, dim=-1, keepdim=True)
    # sim = torch.sqrt(F.mse_loss(gt, network_output)) #distill, to do KL
    sim = F.kl_div(gt, network_output, reduction='mean')

    return sim

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

