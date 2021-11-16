
import os
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from itertools import chain
from collections import OrderedDict

import torch
import torch.nn as nn
from IndexNetMatting.scripts.hlmobilenetv2 import hlmobilenetv2

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

def arg_parse():
    '''
      各種パラメータの読み込み
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default="./data/images", type=str)
    parser.add_argument('--trimap_dir', default="./data/trimaps", type=str)
    parser.add_argument('--out_dir', default="./data/results", type=str)
    args = parser.parse_args()

    return args

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x

def inference(image_path, trimap_path):
    with torch.no_grad():
        image, trimap = read_image(image_path), read_image(trimap_path)
        trimap = np.expand_dims(trimap, axis=2)
        image = np.concatenate((image, trimap), axis=2)
        
        h, w = image.shape[:2]

        image = image.astype('float32')
        image = (IMG_SCALE * image - IMG_MEAN) / IMG_STD
        image = image.astype('float32')

        image = image_alignment(image, 32)
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
        inputs = inputs.to(device)
        
        # inference
        outputs = net(inputs)

        outputs = outputs.squeeze().cpu().numpy()
        alpha = cv2.resize(outputs, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = trimap.squeeze()
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha = (1 - mask) * trimap + mask * alpha

        return alpha.astype(np.uint8)

if __name__ == "__main__":

  args = arg_parse()
  os.makedirs(args.out_dir, exist_ok=True)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # モデル読み込み
  net = hlmobilenetv2(
          pretrained=False,
          freeze_bn=True, 
          output_stride=32,
          apply_aspp=True,
          conv_operator='std_conv',
          decoder='indexnet',
          decoder_kernel_size=5,
          indexnet='depthwise',
          index_mode='m2o',
          use_nonlinear=True,
          use_context=True
      )
    
  try:
      checkpoint_path = "./IndexNetMatting/pretrained/indexnet_matting.pth.tar"
      checkpoint = torch.load(checkpoint_path, map_location=device)
      pretrained_dict = OrderedDict()
      for key, value in checkpoint['state_dict'].items():
          if 'module' in key:
              key = key[7:]
          pretrained_dict[key] = value
  except:
      raise Exception('Please download the pretrained model!')
  net.load_state_dict(pretrained_dict)
  net.to(device)
  if torch.cuda.is_available():
      net = nn.DataParallel(net)

  net.eval()

  ext_list = ["jpg", "png", "jpeg"]
  image_list = sorted(list(chain.from_iterable([glob.glob(os.path.join(args.image_dir, "*." + ext)) for ext in ext_list])))
  trimap_list = sorted(list(chain.from_iterable([glob.glob(os.path.join(args.trimap_dir, "*." + ext)) for ext in ext_list])))


  for image, trimap in zip(image_list, trimap_list):
    alpha_mask = inference(image, trimap)

    image_name = os.path.splitext(os.path.basename(image))[0]
    image_name += ".png"
    output_path = os.path.join(args.out_dir, image_name)
    cv2.imwrite(output_path, alpha_mask)









