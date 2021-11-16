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

      # マスク画像の膨張収縮を用いてTrimap生成
      kernel = np.ones((3, 3), np.uint8)
      eroded_mask = cv2.erode(mask_image, kernel, iterations=10)
      dilated_mask = cv2.dilate(mask_image, kernel, iterations=10)
      trimap_image = np.full((height, width), 128)
      trimap_image[eroded_mask >= 254] = 255
      trimap_image[dilated_mask <= 1] = 0
      trimap_image = trimap_image.astype(np.uint8)

      # 保存
      image_name = os.path.basename(image_path)
      output_path = os.path.join(args.trimap_dir, image_name)
      cv2.imwrite(output_path, trimap_image)




