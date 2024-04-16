import os
import argparse
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=False, default="/home/zhongyao/dl/LangTriplane/data/preprocessed_dataset/sofa/")
    parser.add_argument('--scale', type=int, default=8)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    scale = args.scale
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        orig_w, orig_h = image.shape[1], image.shape[0]
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)
        cv2.imwrite(os.path.join("/home/zhongyao/dl/LangTriplane/data/resize8/sofa/images",data_path),image)

print('resize done')