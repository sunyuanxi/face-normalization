import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from config import cfg
from resnet50 import Resnet50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id

face_model = Resnet50()
face_model.build()

images = np.loadtxt('../Data/c_Multi-PIE/front_list/front_fea_list.txt', dtype='str')
image_list = [os.path.join('../Data/c_Multi-PIE/front_list/', img) for img in images]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config, graph = tf.get_default_graph()) as sess:
    sess.run(tf.global_variables_initializer())
    features = {}
    for i in image_list:
        img_raw = tf.read_file(i)
        img_tensor = tf.image.decode_jpeg(img_raw, channels = 3)
        img_value = tf.image.resize_images(img_tensor, [224, 224])
        img_value = tf.reshape(img_value, [1, 224, 224, 3])
        image_feature = face_model.forward(img_value, 'image_enc')
        features[i.split('_')[3]] = tf.reshape(image_feature[-1], [2048]).eval()

    print(features.keys()[:10])

    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f, protocol = 2)
