import numpy as np
import cv2
import os
from datetime import datetime
from progressbar import ProgressBar
from module import metric_module
import pandas as pd
import argparse


DIRNAME_PREDICTION = ''
DIRNAME_GROUNDTRUTH = ''
THRESH_CONFIDENCE      = 0.3
THRESH_IOU_CONFUSION   = 0.5


def parse_args():
    """Parse input arguments."""
    desc = ('Compare the ground truth and model prediction '
            'output confusion matrix and other important evaluation matrix')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-ip', '--image_prediction', type=str, required=True,
        help='Path to the input prediction folder')
    parser.add_argument(
        '-ig', '--image_groundtruth', type=str, required=True,
        help='Path to the ground truth folder'
    )
    parser.add_argument(
        '-o', '--output_folder', type=str, default='./generated_metrices',
        help='Path to the output folder')
    parser.add_argument(
        '-s', '--save_output', type=bool, default=True,
        help='Save the generated metrices in the csv format')
    parser.add_argument(
        '-conf', '--confidence_score', type=float, default=0.3,
        help='Change for the confidence value')
    parser.add_argument(
        '-nms', '--nms_score', type=float, default=0.5,
        help='Change for the nms value')
    parser.add_argument(
        '-nm', '--names_file', type=str, required=True,
        help='Path to the classes names file')
    args = parser.parse_args()
    
    return args 

def calculate_metric(metric):

    total_image = len(os.listdir(DIRNAME_PREDICTION))
    pbar = ProgressBar(start=0, maxval=total_image)
    for index,files in enumerate(os.listdir(DIRNAME_PREDICTION)):
        file = os.path.join(DIRNAME_PREDICTION, files)
        filename = os.path.basename(file)
        textname_prediction = file
        textname_groundtruth = DIRNAME_GROUNDTRUTH + filename

        with open(textname_groundtruth) as f:
            info_groundtruth = f.read().splitlines()
        bboxes_groundtruth = []
        labels_groundtruth = []
        for bbox in info_groundtruth:
            bbox = bbox.split()
            label = int(bbox[0])
            #label = 0
            bboxes_groundtruth.append([float(c) for c in bbox[1:5]])
            labels_groundtruth.append(label)

        with open(textname_prediction) as f:
            info_prediction = f.read().splitlines()
        bboxes_prediction = []
        labels_prediction = []
        scores_prediction = []
        for bbox in info_prediction:
            bbox = bbox.split()
            label      = int(bbox[0])
            #label      = 0
            confidence = float(bbox[5])
            if confidence>=THRESH_CONFIDENCE:
                bboxes_prediction.append([float(c) for c in bbox[1:5]])
                labels_prediction.append(label)
                scores_prediction.append(confidence)

        metric.update(bboxes_prediction=bboxes_prediction,
                    labels_prediction=labels_prediction,
                    scores_prediction=scores_prediction,
                    bboxes_groundtruth=bboxes_groundtruth,
                    labels_groundtruth=labels_groundtruth)
        # progress = 100*index/total_image
        pbar.update(index)
    pbar.finish()
    return metric

#metric.get_mAP(type_mAP="VOC07",
#               conclude=True)
#print
# metric.get_mAP(type_mAP="VOC12",
#                conclude=True)
# print
# metric.get_mAP(type_mAP="COCO",
#                conclude=True)

# metric.get_mAP(type_mAP="USER_DEFINED",
#                conclude=True)
# print
# metric.get_confusion(thresh_confidence=THRESH_CONFIDENCE,
#                      thresh_IOU=THRESH_IOU_CONFUSION,
#                      conclude=True)

def get_output_metric(metric, save_files, NAMES_CLASS, out_folder):
    
    if save_files:
        results, matrix_confusion, total, pr_value = metric.get_confusion(thresh_confidence=THRESH_CONFIDENCE,
                            thresh_IOU=THRESH_IOU_CONFUSION,
                            conclude=False)

        with open(out_folder+'/complete_results_matrix_'+ datetime.now().strftime('%d%m%Y_%H%M%S%f')[:-3] + '.txt', 'w+') as res:
            res.writelines(results)
        res.close()

        NAMES_CLASS.append('Missed Detections')
        total.append(0)
        # print(matrix_confusion.shape,len(total))
        matrix_confusion = np.column_stack((matrix_confusion,total))
        matrix_confusion_added_labels = np.hstack((np.atleast_2d(NAMES_CLASS).T,
                                                     matrix_confusion))
        NAMES_CLASS.insert(0,'S.NO')
        NAMES_CLASS.append('Total')
        df = pd.DataFrame(matrix_confusion_added_labels, columns=NAMES_CLASS)
        # print(df)
        df.to_csv(out_folder+'/out_cm_yolo_'+ datetime.now().strftime('%d%m%Y_%H%M%S%f')[:-3] + '.csv')

        df2 = pd.DataFrame(pr_value, columns = ['Name','Precision','Recall','Avg IOU'])
        df2.to_csv(out_folder+'/out_pr_'+ datetime.now().strftime('%d%m%Y_%H%M%S%f')[:-3] + '.csv')

    else:
        metric.get_confusion(thresh_confidence=THRESH_CONFIDENCE,
                     thresh_IOU=THRESH_IOU_CONFUSION,
                     conclude=True)

def main():

    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok = True)
    with open(args.names_file) as f:
        NAMES_CLASS = f.read().splitlines()
    NUMBER_CLASSES = len(NAMES_CLASS)

    metric = metric_module.ObjectDetectionMetric(names_class=NAMES_CLASS,
                                             check_class_first=False)

    op_metric = calculate_metric(metric)

    get_output_metric(op_metric, args.save_output, NAMES_CLASS, out_folder)


if __name__ == '__main__':
    args = parse_args()
    DIRNAME_PREDICTION = args.image_prediction
    DIRNAME_GROUNDTRUTH = args.image_groundtruth
    THRESH_CONFIDENCE = args.confidence_score
    THRESH_IOU_CONFUSION = args.nms_score
    main()
