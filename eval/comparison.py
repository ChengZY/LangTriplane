import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

scene = 'bed'
lvl = 0
path_gt = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/gt".format(scene,lvl)
path1 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/renders".format(scene,lvl)
path2 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_{}/train/ours_None/renders".format(scene,lvl)

paths = []
save_path = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/comparison_2/".format(scene)
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(path1)
file  = files[0]


for file in files:
    cmp_lvl = []
    for lvl in range(4):
        paths = ["/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/gt".format(scene,lvl), \
                 "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/renders".format(scene,lvl),\
                 "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_{}/train/ours_None/renders".format(scene, lvl), \
                 "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_render_img2_{}/train/ours_None/renders".format(scene, lvl),\
                 # "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_render_img_only_con_{}/train/ours_None/renders".format(scene, lvl)
                 ]
        # file  = files[0]
        cmp_ims = []
        for path in paths:
            im = cv2.imread(os.path.join(path,file))
            cmp_ims.append(im)
        ims_cmp = np.concatenate(cmp_ims, axis=0)
        cmp_lvl.append(ims_cmp)
    cmp_lvl_save = np.concatenate(cmp_lvl, axis=1)
    cv2.imwrite(os.path.join(save_path,file),cmp_lvl_save)

# gts = np.concatenate(gts,axis = 1)
# comp1 = np.concatenate(comp1,axis = 1)
# comp2 = np.concatenate(comp2,axis = 1)
# comp = np.concatenate([gts, comp1,comp2],axis = 0)
# plt.imshow(comp)
# plt.show()