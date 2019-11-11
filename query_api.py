from extract_cnn_vgg16_keras import VGGNet
import numpy as np
import h5py
import cv2

def get_database_index():
    h5f = h5py.File('featureCNN.h5','r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    return feats, imgNames
 
def get_image_feature(im_file,feats,imgNames):

    model = VGGNet()
    queryVec = model.extract_feat(im_file)
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    maxres = 3
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " %maxres, imlist)

    print(rank_score[0:3])
    return rank_score

def get_image_search(im_file):
    h5f = h5py.File('featureCNN.h5','r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    model = VGGNet()

    q_vector = model.extract_feat(im_file)

    scores = np.dot(q_vector, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    maxres = 5
    im_list = [str(imgNames[index].decode()) for i,index in enumerate(rank_ID[0:maxres])]
    
    im_score = [str(rank_score[i]) for i in range(maxres)]
    result_dict = dict(zip(im_list, im_score))

    return result_dict
