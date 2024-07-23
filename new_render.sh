DIR="/home/zhongyao/dl/LangTriplane/data/3d_ovs_8/"

#for FILE in "$DIR"/*
#do
#    for level in 0 1 2 3
#    do
#      SUBDIR_NAME=$(basename "$FILE")
#      echo "Processing $SUBDIR_NAME"
#      CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/${SUBDIR_NAME}\
#       -m ./output/3d_ovs_8/${SUBDIR_NAME}/retrain_lang_8_${level} --feature_level ${level} \
#       --include_feature \
#       --use_triplane
#    done
#done

for FILE in "$DIR"/*
do
    for level in 0 1 2 3
    do
      SUBDIR_NAME=$(basename "$FILE")
      echo "Processing $SUBDIR_NAME"
      CUDA_VISIBLE_DEVICES=1 python render_ad.py -s ./data/3d_ovs_8/${SUBDIR_NAME}\
       -m ./output/3d_ovs_8/${SUBDIR_NAME}/retrain_lang_8_tp_render_img_only_con_w0002_bs_${level} --feature_level ${level} \
       --include_feature \
       --use_triplane
    done
done