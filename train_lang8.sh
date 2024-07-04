#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m output/3d_ovs_8/room/retrain_img_8 --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp_trip_noencode --feature_level 0 -r 1 --include_feature --use_triplane --use_triplane --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth
CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.4
CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.4
CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.4
CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/room/ -m ./output/3d_ovs_8/room/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/room/retrain_img_8_0/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.4

#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bed/ -m ./output/3d_ovs_8/bed/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/bed/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bed/ -m ./output/3d_ovs_8/bed/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/bed/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bed/ -m ./output/3d_ovs_8/bed/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/bed/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bed/ -m ./output/3d_ovs_8/bed/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/bed/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane

#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bench/ -m ./output/3d_ovs_8/bench/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/bench/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bench/ -m ./output/3d_ovs_8/bench/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/bench/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bench/ -m ./output/3d_ovs_8/bench/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/bench/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/bench/ -m ./output/3d_ovs_8/bench/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/bench/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/sofa/ -m ./output/3d_ovs_8/sofa/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/sofa/ -m ./output/3d_ovs_8/sofa/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/sofa/ -m ./output/3d_ovs_8/sofa/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/sofa/ -m ./output/3d_ovs_8/sofa/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/blue_sofa/ -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/blue_sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/blue_sofa/ -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/blue_sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/blue_sofa/ -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/blue_sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/blue_sofa/ -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/blue_sofa/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/lawn/ -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/lawn/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/lawn/ -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/lawn/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/lawn/ -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/lawn/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/lawn/ -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/lawn/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/office_desk/ -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/office_desk/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/office_desk/ -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/office_desk/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/office_desk/ -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/office_desk/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/office_desk/ -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/office_desk/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/snacks/ -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/snacks/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/snacks/ -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp --feature_level 1 -r 1 --start_checkpoint ./output/3d_ovs_8/snacks/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/snacks/ -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp --feature_level 2 -r 1 --start_checkpoint ./output/3d_ovs_8/snacks/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/snacks/ -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp --feature_level 3 -r 1 --start_checkpoint ./output/3d_ovs_8/snacks/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane

#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2 --feature_level 0 -r 1 --start_checkpoint ./output/lerf_ovs_2/ramen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2 --feature_level 1 -r 1 --start_checkpoint ./output/lerf_ovs_2/ramen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2 --feature_level 2 -r 1 --start_checkpoint ./output/lerf_ovs_2/ramen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2 --feature_level 3 -r 1 --start_checkpoint ./output/lerf_ovs_2/ramen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2 --feature_level 0 -r 1 --start_checkpoint ./output/lerf_ovs_2/waldo_kitchen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2 --feature_level 1 -r 1 --start_checkpoint ./output/lerf_ovs_2/waldo_kitchen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2 --feature_level 2 -r 1 --start_checkpoint ./output/lerf_ovs_2/waldo_kitchen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2 --feature_level 3 -r 1 --start_checkpoint ./output/lerf_ovs_2/waldo_kitchen/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane

#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2 --feature_level 0 -r 1 --start_checkpoint ./output/lerf_ovs_2/figurines/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2 --feature_level 1 -r 1 --start_checkpoint ./output/lerf_ovs_2/figurines/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2 --feature_level 2 -r 1 --start_checkpoint ./output/lerf_ovs_2/figurines/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python train_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2 --feature_level 3 -r 1 --start_checkpoint ./output/lerf_ovs_2/figurines/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane

#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2 --feature_level 0 -r 1 --start_checkpoint ./output/lerf_ovs_2/teatime/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.2
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2 --feature_level 1 -r 1 --start_checkpoint ./output/lerf_ovs_2/teatime/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.2
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2 --feature_level 2 -r 1 --start_checkpoint ./output/lerf_ovs_2/teatime/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.2
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2 --feature_level 3 -r 1 --start_checkpoint ./output/lerf_ovs_2/teatime/retrain_img_2/_-1/chkpnt30000.pth --include_feature --use_triplane --ip 127.0.0.2