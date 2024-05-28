#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/snacks/ -m output/3d_ovs_8/snacks/retrain_img_8 --feature_level 0 -r 1
#CUDA_VISIBLE_DEVICES=0 python train_ad.py -s ./data/3d_ovs_8/snacks/ -m ./output/3d_ovs_8/snacks/retrain_lang_8 --feature_level 0 -r 1 --start_checkpoint ./output/3d_ovs_8/snacks/retrain_img_8_0/chkpnt30000.pth
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/bench -m ./output/3d_ovs_8/bench/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_tp_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene lawn
cd ..

CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_tp_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene office_desk
cd ..

CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_tp_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene snacks
cd ..

#cd eval
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene bed
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene bench
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene lawn
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene room
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene snacks
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene blue_sofa
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene office_desk
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene sofa # change size

#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_0 --feature_level 0 --include_feature

#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2_0 --feature_level 0 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2_1 --feature_level 1 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2_2 --feature_level 2 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/waldo_kitchen -m ./output/lerf_ovs_2/waldo_kitchen/retrain_lang_2_3 --feature_level 3 --include_feature

#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2_0 --feature_level 0 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2_1 --feature_level 1 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2_2 --feature_level 2 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/teatime -m ./output/lerf_ovs_2/teatime/retrain_lang_2_3 --feature_level 3 --include_feature
#
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2_0 --feature_level 0 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2_1 --feature_level 1 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2_2 --feature_level 2 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/ramen -m ./output/lerf_ovs_2/ramen/retrain_lang_2_3 --feature_level 3 --include_feature
#
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2_0 --feature_level 0 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2_1 --feature_level 1 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2_2 --feature_level 2 --include_feature
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/lerf_ovs_2/figurines -m ./output/lerf_ovs_2/figurines/retrain_lang_2_3 --feature_level 3 --include_feature

#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_12_0 --feature_level 0 --include_feature --use_triplane
#CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/snacks -m ./output/3d_ovs_8/snacks/retrain_lang_8_trip_noencode_0 --feature_level 0 --include_feature --use_triplane