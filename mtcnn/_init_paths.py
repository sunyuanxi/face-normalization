import sys, os
import os.path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

caffe_path = '/home/syx_internship/CNN_tools/caffe/'

# Add caffe to PYTHONPATH
caffe_path = osp.join(caffe_path, 'python')
add_path(caffe_path)

os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints

