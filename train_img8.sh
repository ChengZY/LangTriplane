#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m output/3d_ovs_8/room/retrain_img_8 --feature_level 0 -r 1
CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8 --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth
CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8 --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth
CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8 --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth
CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8 --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth

#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/bed/ -m output/3d_ovs_8/bed/retrain_img_8/ --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/bench/ -m output/3d_ovs_8/bench/retrain_img_8/ --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/blue_sofa/ -m output/3d_ovs_8/blue_sofa/retrain_img_8/ --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/covered_desk/ -m output/3d_ovs_8/covered_desk/retrain_img_8/ --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/lawn/ -m output/3d_ovs_8/lawn/retrain_img_8/ --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/office_desk/ -m output/3d_ovs_8/office_desk/retrain_img_8/ --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/snacks/ -m output/3d_ovs_8/snacks/retrain_img_8/ --feature_level 0 -r 1
CUDA_VISIBLE_DEVICES=1 python train_ad_img.py -s ./data/3d_ovs_8/sofa/ -m output/3d_ovs_8/sofa/retrain_img_8/ --feature_level 0 -r 1