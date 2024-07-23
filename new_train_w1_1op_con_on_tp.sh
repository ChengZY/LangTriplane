DIR="/home/zhongyao/dl/LangTriplane/data/3d_ovs_8/"
for FILE in "$DIR"/*
do
#    echo "Processing $FILE"
    for level in 0 1 2 3
    do
      SUBDIR_NAME=$(basename "$FILE")
      echo "Processing $SUBDIR_NAME"
      CUDA_VISIBLE_DEVICES=0 python train_ad_con24_1op.py -s ./data/3d_ovs_8/${SUBDIR_NAME}/ \
      -m ./output/3d_ovs_8/${SUBDIR_NAME}/retrain_lang_8_tp_trans_con24_on_tp_w0002 \
      --feature_level ${level} -r 1 \
      --start_checkpoint ./output/3d_ovs_8/${SUBDIR_NAME}/retrain_img_8/_0/chkpnt30000.pth --include_feature --use_triplane \
      --ip 127.0.0.9
      # For example, you might wantsh to copy each file to another directory
      # cp "$FILE" /path/to/destination/directory/ #      -m ./output/3d_ovs_8/${SUBDIR_NAME}/retrain_lang_8_tp_trans_con512_w0002 \
    done
done