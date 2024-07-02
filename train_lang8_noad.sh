CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane
CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane
CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane
CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane

#CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_noad --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane

