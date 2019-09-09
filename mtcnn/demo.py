import cv2, mtcnn, os
import numpy as np

detect_net = mtcnn.MTCNN()

#imglist = ['test.png']
imgpath = '../Data/Multi-PIE/Session 1/Multiview/'
imglist = np.loadtxt('./imglist.txt', dtype='str', delimiter=',')
#img_list = np.loadtxt('../Data/experiment_Multi-PIE/experiment_list.txt', dtype='string', delimiter=',')
images = [os.path.join(imgpath, img) for img in imglist]
faces = detect_net.detect(images)
print(faces)
