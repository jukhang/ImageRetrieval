from extract_cnn_vgg16_keras import VGGNet
import numpy as np
import h5py
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = True,
	help = "Path for output retrieved images")
args = vars(ap.parse_args())

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
#print(feats)
imgNames = h5f['dataset_2'][:]
#print(imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")
    
# 载入VGG模型
model = VGGNet()

# 获取图像的特征参数
queryDir = args["query"]
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

#print(rank_ID[0:10])
#print("-----------")
#print(rank_score)

maxres = 3
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " %maxres, imlist)

print(rank_score[0:3])
