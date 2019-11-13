from extract_cnn_vgg16_keras import VGGNet
from keras.backend import clear_session
import tensorflow as tf
import numpy as np
import h5py
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

def get_image_search(im_file):
    h5f = h5py.File('featureCNN.h5','r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    model = VGGNet()
    q_vector = model.extract_feat(im_file)
    #print("清除训练模型！")
    clear_session()
    tf.reset_default_graph()

    scores = np.dot(q_vector, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    maxres = 5
    im_list = [str(imgNames[index].decode()) for i,index in enumerate(rank_ID[0:maxres])]
    
    im_score = [str(rank_score[i]) for i in range(maxres)]
    result_dict = dict(zip(im_list, im_score))
    return result_dict
