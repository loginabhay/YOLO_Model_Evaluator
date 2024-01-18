import os
import cv2
import configparser


# class to load the yolo v4 model with opencv
class YOLOcv(object):

    def __init__(self, cfg, weights,shape):
        self.cfg = cfg
        # config = configparser.ConfigParser(strict=False)
        # config.read(self.cfg)
        # self.width = config[net][width]
        # self.height = config[net][height]    size=(self.width,self.height)
        self.weights = weights
        self.shape = shape
        self.net = cv2.dnn.readNet(self.weights, self.cfg)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.shape,self.shape), scale=1/255, swapRB=True)

    def detect(self,frame,conf,nms):
        classes, scores, boxes = self.model.detect(frame, conf, nms)
        return classes,scores,boxes