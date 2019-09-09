import cv2, mtcnn
import numpy as np
detect_net = mtcnn.MTCNN()

imgpath = '../Data/Face_Normalization/'
imglist = '../Data/Face_Normalization/crop_list.txt'
savepath = '../Data/Face_Normalization/'
boxes = detect_net.crop_image(imgpath, imglist, savepath)
