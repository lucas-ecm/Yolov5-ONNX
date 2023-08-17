import argparse
import os
import time
import numpy as np
import cv2
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx
from vidsz.opencv import Reader, Writer
from cvu.utils.google_utils import gdrive_download
from timeit import default_timer as timer

CLASSES = [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper',
]

def detect_image(model, image_path, output_image):
    # read image
    start = timer()
    image = cv2.imread(image_path)
    end = timer()
    preprocess_time = end - start

    # inference
    start = timer()
    preds = model(image)
    print(preds)
    end = timer()
    inference_time = end - start
    
    # draw image
    start = timer()
    preds.draw(image)

    # write image
    cv2.imwrite(output_image, image)
    end = timer()
    postprocess_time = end - start

    return preprocess_time, inference_time, postprocess_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='yolov5s',
                        help='onnx weights path')

    parser.add_argument('--input_dir',
                        type=str,
                        default='people.mp4',
                        help='complete path to input files dir')

    parser.add_argument('--output_dir',
                        type=str,
                        default='people_out.mp4',
                        help='complete path to output files dir')

    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu')

    opt = parser.parse_args()

    # inputs_list = []
    # outputs_list = []
    count = 0
    preprocess_time = 0
    inference_time = 0
    postprocess_time = 0

    model = Yolov5Onnx(classes=CLASSES,
           backend="onnx",
           weight=opt.weights,
           device=opt.device)
    
    for filename in os.listdir(opt.input_dir):
        print(f'Detecting image {count}: {filename}')
        count += 1
        # images_list.append(filename)
        stripped_name = filename.split('/')[-1]
        input = os.path.join(opt.input_dir, filename)
        output = os.path.join(opt.output_dir, filename)

        curr_pre_time, curr_inf_time, curr_post_time = detect_image(
            model, input, output
        )

        preprocess_time += curr_pre_time
        inference_time += curr_inf_time
        postprocess_time += curr_post_time

    print(f'Preprocess_time: {preprocess_time}')
    print(f'Inference_time: {inference_time}')
    print(f'Postprocess_time: {postprocess_time}')



        
        
        

    

