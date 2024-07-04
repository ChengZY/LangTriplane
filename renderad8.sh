CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/bench -m ./output/3d_ovs_8/bench/retrain_lang_8_tp_render_img_only_con_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/bench -m ./output/3d_ovs_8/bench/retrain_lang_8_tp_render_img_only_con_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/bench -m ./output/3d_ovs_8/bench/retrain_lang_8_tp_render_img_only_con_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/bench -m ./output/3d_ovs_8/bench/retrain_lang_8_tp_render_img_only_con_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene bench
cd ..

CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/room -m ./output/3d_ovs_8/room/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/room -m ./output/3d_ovs_8/room/retrain_lang_8_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/room -m ./output/3d_ovs_8/room/retrain_lang_8_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/room -m ./output/3d_ovs_8/room/retrain_lang_8_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene room
cd ..

CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/bed -m ./output/3d_ovs_8/bed/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/bed -m ./output/3d_ovs_8/bed/retrain_lang_8_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/bed -m ./output/3d_ovs_8/bed/retrain_lang_8_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/bed -m ./output/3d_ovs_8/bed/retrain_lang_8_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene bed
cd ..

CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/lawn -m ./output/3d_ovs_8/lawn/retrain_lang_8_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene lawn
cd ..

CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/office_desk -m ./output/3d_ovs_8/office_desk/retrain_lang_8_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene office_desk
cd ..

CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/blue_sofa -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/blue_sofa -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/blue_sofa -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/blue_sofa -m ./output/3d_ovs_8/blue_sofa/retrain_lang_8_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene blue_sofa
cd ..

CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/sofa -m ./output/3d_ovs_8/sofa/retrain_lang_8_0 --feature_level 0 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/sofa -m ./output/3d_ovs_8/sofa/retrain_lang_8_1 --feature_level 1 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/sofa -m ./output/3d_ovs_8/sofa/retrain_lang_8_2 --feature_level 2 --include_feature
CUDA_VISIBLE_DEVICES=0 python render_ad.py -s ./data/3d_ovs_8/sofa -m ./output/3d_ovs_8/sofa/retrain_lang_8_3 --feature_level 3 --include_feature
cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene sofa
cd ..

cd eval
CUDA_VISIBLE_DEVICES=0 python eval_iou_3d_ovs.py --scene bench
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene lawn
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene room
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene blue_sofa
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs.py --scene office_desk
cd ..

cd eval
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene bed
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene bench
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene lawn
#CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene room
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene sofa
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene blue_sofa
CUDA_VISIBLE_DEVICES=1 python eval_iou_3d_ovs_vote.py --scene office_desk
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