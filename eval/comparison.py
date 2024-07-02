import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

scene = 'bench'
lvl = 0
path_gt = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/gt".format(scene,lvl)
path1 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/renders".format(scene,lvl)
path2 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_{}/train/ours_None/renders".format(scene,lvl)
save_path = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/comparison/".format(scene)
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(path1)
file  = files[0]


for file in files:
    gts = []
    comp1 = []
    comp2 = []
    for lvl in range(4):
        path_gt = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/gt".format(scene,
                                                                                                                  lvl)
        path1 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/renders".format(scene,
                                                                                                                     lvl)
        path2 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_{}/train/ours_None/renders".format(
            scene, lvl)

        # file  = files[0]
        im_gt = cv2.imread(os.path.join(path_gt,file))
        gt = cv2.imread(os.path.join(path_gt,file))
        im1 = cv2.imread(os.path.join(path1,file))
        im2 = cv2.imread(os.path.join(path2,file))
        gts.append(im_gt)
        comp1.append(im1)
        comp2.append(im2)
    gts = np.concatenate(gts, axis=1)
    comp1 = np.concatenate(comp1, axis=1)
    comp2 = np.concatenate(comp2, axis=1)
    comp = np.concatenate([gts, comp1, comp2], axis=0)
    cv2.imwrite(os.path.join(save_path,file),comp)

# gts = np.concatenate(gts,axis = 1)
# comp1 = np.concatenate(comp1,axis = 1)
# comp2 = np.concatenate(comp2,axis = 1)
# comp = np.concatenate([gts, comp1,comp2],axis = 0)
plt.imshow(comp)
plt.show()