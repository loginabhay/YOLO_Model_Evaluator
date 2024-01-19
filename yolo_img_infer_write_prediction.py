"""This Script can be used as generating txt yolo
format from using opencv 
"""


import os
import argparse
import numpy as np
import cv2
from module.loader import YOLOcv
from progressbar import ProgressBar


base_folder = ''
output_folder = ''

def parse_args():
    """Parse input arguments."""
    desc = ('Run the Opencv detection on test images '
            'save the result in txt file in yolo format')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-i', '--image_folder', type=str, required=True,
        help='Path to the input image folder')
    parser.add_argument(
        '-o', '--output_folder', type=str, default='./image_predictions',
        help='Path to the output folder')
    parser.add_argument(
        '-c', '--cfg', type=str, required=True,
        help='YOLO Model CFG file')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='YOLO Model weight file')
    # parser.add_argument(
    #     '-nm', '--names_file', type=str, required=True,
    #     help='Path to the classes names file')
    parser.add_argument(
        '-conf', '--confidence_score', type=float, default=0.3,
        help='Change for the confidence value')
    parser.add_argument(
        '-nms', '--nms_score', type=float, default=0.5,
        help='Change for the nms value')
    parser.add_argument(
        '-n', '--network_shape', type=int, default=416,
        help='Mention the network shape')
    parser.add_argument(
        '-ca', '--create_annotation', type=bool, default=False,
        help='Create annotation instead of prediction')
    args = parser.parse_args()
    return args

# below funtion converts opencv rectangle box to yolo format
def bnd_box_to_yolo_line(box,img_size):
    (x_min, y_min) = (box[0], box[1])
    (w, h) = (box[2], box[3])
    x_max = x_min+w
    y_max = y_min+h
    
    x_center = float((x_min + x_max)) / 2 / img_size[1]
    y_center = float((y_min + y_max)) / 2 / img_size[0]

    w = float((x_max - x_min)) / img_size[1]
    h = float((y_max - y_min)) / img_size[0]

    return x_center, y_center, w, h

# Below function writes the text file in the output folder
def write_to_txt(boxes,clas,score,img_size,txt_name):
    txt_path =  output_folder + '/' + txt_name
    txt_file = open(txt_path, 'w+')
    for (classid, box, score) in zip(clas, boxes, score):
        classid = int(classid)
        bb = bnd_box_to_yolo_line(box, img_size)
        if not args.create_annotation:
            txt_file.write(str(classid) + " " + " ".join([str(a) for a in bb]) + " " + str(score) + '\n')
        else:
            txt_file.write(str(classid) + " " + " ".join([str(a) for a in bb]) + '\n')
    txt_file.close()
    
def loop_and_detect(yolo_model, conf_th, nms):
    """Continuously capture images from folder and do object detection.

    # Arguments
      folder: input folder path.
      yolo_model: the YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
    """
    total_image = len(os.listdir(base_folder))
    pbar = ProgressBar(start=0, maxval=total_image)
    for index_value,files in enumerate(os.listdir(base_folder)):
        file = os.path.join(base_folder, files)
        name = os.path.basename(file)
        frame = cv2.imread(file)
        img_size = frame.shape[:2]
        classes,scores,boxes = yolo_model.detect(frame,conf_th,nms)
        txt_name = name.rstrip('.jpeg') + '.txt'
        write_to_txt(boxes,classes,scores,img_size,txt_name)

        # progress = 100*index_value/total_image
        pbar.update(index_value)
    
    pbar.finish()


def main():
    
    os.makedirs(output_folder, exist_ok=True)
    yolo_model = YOLOcv(args.cfg,args.model,args.network_shape)
    
    loop_and_detect(yolo_model, conf_th=args.confidence_score, nms=args.nms_score)

    print('\n>>>>>>>>>>>>[INFO] Done')


if __name__ == '__main__':
    args = parse_args()
    base_folder = args.image_folder
    output_folder = args.output_folder
    main()
