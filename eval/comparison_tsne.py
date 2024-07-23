import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

scene = 'bed'
lvl = 0
path_gt = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/gt_npy".format(scene,lvl)
path1 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/renders_npy".format(scene,lvl)
path2 = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_{}/train/ours_None/renders_npy".format(scene,lvl)
#retrain_lang_8_3
paths = []
save_path = "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/comparison_2/".format(scene)
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(path1)
file  = files[0]

# language_feature_name = os.path.join("./data/resize8/sofa/language_features_resize8/", self.image_name)

for file in files[:1]:
    seg_map = np.load(os.path.join("../data/3d_ovs_8/{}/language_features/".format(scene), str(int(file.split(".")[0])).zfill(2)) + '_s.npy')
    cmp_lvl = []
    for lvl in range(4):
        paths = ["/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/gt_npy".format(scene,lvl), \
                 "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_{}/train/ours_None/renders_npy".format(scene,lvl), \
                 "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_render_img_only_con_w0002_{}/train/ours_None/renders_npy".format(scene, lvl), \
                 "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_render_img_only_con_w0002_bs_{}/train/ours_None/renders_npy".format(
                     scene, lvl), \
                 # "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_{}/train/ours_None/renders_npy".format(scene, lvl), \
                 # "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_render_img2_{}/train/ours_None/renders_npy".format(scene, lvl),\
                 # "/home/zhongyao/dl/LangTriplane/output/3d_ovs_8/{}/retrain_lang_8_tp_render_img_only_con_{}/train/ours_None/renders".format(scene, lvl)
                 ]
        # file  = files[0]
        # feature_map = np.load(language_feature_name + '_f.npy')
        renders_npy = []
        for path in paths:
            render_npy = np.load(os.path.join(path,file))
            renders_npy.append(render_npy)
#         ims_cmp = np.concatenate(cmp_ims, axis=0)
        cmp_lvl.append(renders_npy)
#     cmp_lvl_save = np.concatenate(cmp_lvl, axis=1)
#     cv2.imwrite(os.path.join(save_path,file),cmp_lvl_save)
#
print ('process')
labels = seg_map[0]
labels = labels.reshape(-1)
pixels1 = cmp_lvl[0][1].reshape(-1, 512)
pixels2 = cmp_lvl[0][2].reshape(-1, 512)
pixels3 = cmp_lvl[0][3].reshape(-1, 512)
# tsne = TSNE(n_components=2, random_state=42, n_iter=250, learning_rate='auto')
# pixels_tsne = tsne.fit_transform(pixels)

pca = PCA(n_components=2)
pixels3_pca = pca.fit_transform(pixels3)
pixels2_pca = pca.fit_transform(pixels2)
pixels1_pca = pca.fit_transform(pixels1)

# Create subplots to compare the PCA results
fig, axs = plt.subplots(1, 3, figsize=(20, 8),sharex=True, sharey=True)

# Plot PCA result for the first feature map
scatter1 = axs[0].scatter(pixels1_pca[:, 0], pixels1_pca[:, 1], c=labels, cmap='viridis', s=1)
axs[0].set_title('PCA visualization of Feature Map 1')
axs[0].set_xlabel('PCA component 1')
axs[0].set_ylabel('PCA component 2')
plt.colorbar(scatter1, ax=axs[0], ticks=np.unique(labels))

# Plot PCA result for the second feature map
scatter2 = axs[1].scatter(pixels2_pca[:, 0], pixels2_pca[:, 1], c=labels, cmap='viridis', s=1)
axs[1].set_title('PCA visualization of Feature Map 2')
axs[1].set_xlabel('PCA component 1')
axs[1].set_ylabel('PCA component 2')
plt.colorbar(scatter2, ax=axs[1], ticks=np.unique(labels))

scatter3 = axs[2].scatter(pixels3_pca[:, 0], pixels3_pca[:, 1], c=labels, cmap='viridis', s=1)
axs[2].set_title('PCA visualization of Feature Map 3')
axs[2].set_xlabel('PCA component 1')
axs[2].set_ylabel('PCA component 2')
plt.colorbar(scatter3, ax=axs[1], ticks=np.unique(labels))

plt.savefig('tsne_feature_map.png')
plt.show()

print ('done')