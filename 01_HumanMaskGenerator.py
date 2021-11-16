import os
import cv2
import glob
import argparse
from itertools import chain
from SOLO.mmdet.apis import init_detector, inference_detector, show_result_ins


def arg_parse():
    '''
      各種パラメータの読み込み
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="./SOLO/checkpoints/DECOUPLED_SOLO_R50_3x.pth", type=str)
    parser.add_argument('--config', default="./SOLO/configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py", type=str)
    parser.add_argument('--out_dir', default="./data/masks", type=str)
    parser.add_argument('--input_dir', default="./data/images", type=str)
    parser.add_argument('--score_threshold', default=0.25, type=float)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()
    os.makedirs(args.out_dir, exist_ok=True)

    config_file = args.config
    checkpoint_file = args.checkpoint

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    results = []
    ext_list = ["jpg", "png", "jpeg"]
    image_list = sorted(list(chain.from_iterable([glob.glob(os.path.join(args.input_dir, "*." + ext)) for ext in ext_list])))
    for idx, image_path in enumerate(image_list):
        result = inference_detector(model, image_path)
        human_mask = show_result_ins(image_path, result, model.CLASSES, score_thr=args.score_threshold, out_human_mask=True)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_name += ".png"
        output_path = os.path.join(args.out_dir, image_name)
        cv2.imwrite(output_path, human_mask)
