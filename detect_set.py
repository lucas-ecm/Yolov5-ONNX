import argparse
import os
import time
import numpy as np
import cv2
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx
from vidsz.opencv import Reader, Writer
from cvu.utils.google_utils import gdrive_download
from timeit import default_timer as timer
import psutil
import pandas as pd

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

    return preprocess_time, inference_time, postprocess_time, preds

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
    parser.add_argument('--backend', type=str, default='onnx', help='onnx or tflite')
    parser.add_argument('--max_iters', type=int, default=None, help='max_iterations')

    opt = parser.parse_args()

    # inputs_list = []
    # outputs_list = []
    count = 0
    preprocess_time = 0
    inference_time = 0
    postprocess_time = 0

    cpu_pctgs = []
    ram_pctgs = []
    ram_totals = []
    
    all_preds = []

    model = Yolov5Onnx(classes=CLASSES,
           backend=opt.backend,
           weight=opt.weights,
           device=opt.device)
    
    for filename in os.listdir(opt.input_dir):
        print(f'Detecting image {count}: {filename}')
        count += 1
        # images_list.append(filename)
        stripped_name = filename.split('/')[-1]
        input = os.path.join(opt.input_dir, filename)
        output = os.path.join(opt.output_dir, filename)

        curr_pre_time, curr_inf_time, curr_post_time, preds = detect_image(
            model, input, output
        )
        all_preds.append(preds)

        cpu_pctg = psutil.cpu_percent()
        print('The CPU usage is: ', cpu_pctg)
        cpu_pctgs.append(cpu_pctg)

        # Getting % usage of virtual_memory ( 3rd field)
        ram_pctg = psutil.virtual_memory()[2]
        print('RAM memory % used:', ram_pctg)
        ram_pctgs.append(ram_pctg)
        # Getting usage of virtual_memory in GB ( 4th field)
        ram_total = psutil.virtual_memory()[3]/1000000000
        print('RAM Used (GB):',ram_total )
        ram_totals.append(ram_total)
        

        preprocess_time += curr_pre_time
        inference_time += curr_inf_time
        postprocess_time += curr_post_time     

        if opt.max_iters and count == opt.max_iters:
            break

    print(f'Preprocess_time: {preprocess_time}')
    print(f'Inference_time: {inference_time}')
    print(f'Postprocess_time: {postprocess_time}')

    print(f'Avg cpu load: {np.mean(cpu_pctgs[1:])}')
    # Getting loadover15 minutes
    load1, load5, load15 = psutil.getloadavg()
     
    cpu_usage = (load1/os.cpu_count()) * 100
     
    print("Avg cpu usage (getloadavg) is : ", cpu_usage)
    print(f'Avg ram load: {np.mean(ram_pctgs[1:])}')
    print(f'Avg ram load [GB]: {np.mean(ram_totals[1:])}')

    all_preds = pd.DataFrame(all_preds)
    all_preds.to_csv(os.path.join(opt.output_dir,'csv_results.csv'), index = False)





        
        
        

    

