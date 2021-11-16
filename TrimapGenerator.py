import os
import cv2
import glob
import argparse
import numpy as np
from itertools import chain

def arg_parse():
    '''
      各種パラメータの読み込み
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--trimap_dir', default="./trimaps", type=str)
    parser.add_argument('--mask_dir', default="./masks", type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()
    os.makedirs(args.trimap_dir, exist_ok=True)

    ext_list = ["jpg", "png"]
    mask_list = sorted(list(chain.from_iterable([glob.glob(os.path.join(args.mask_dir, "*." + ext)) for ext in ext_list])))

    for idx, image_path in enumerate(mask_list):
      # 画像読み込み
      mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
      height, width = mask_image.shape[:2]

      # 二値化 & 輪郭検出
      _, mask_image = cv2.threshold(mask_image, 127, 255, 0)
      labels, contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

      # 輪郭を膨張してtrimap生成
      trimap_image = np.zeros((width, height))
      trimap_image = cv2.drawContours(trimap_image, contours, -1, color=(255, 255, 255), thickness=2)

      trimap_image = cv2.dilate(trimap_image, (5, 5), iterations=10)
      trimap_image = trimap_image / 255 * 128

      # 保存
      image_name = os.path.basename(image_path)
      output_path = os.path.join(args.trimap_dir, image_name)
      cv2.imwrite(output_path, trimap_image)




