# Unipower Image Retrieval Engine Based on Keras

```python
In [1]: import keras
Using Theano backend.
```
### 使用

- 步骤一

`python index.py -database <path-to-dataset> -index <name-for-output-index>`

- 步骤二

`python query_online.py -query <path-to-query-image> -index <path-to-index-flie> -result <path-to-images-for-retrieval>`

```sh
├── database 图像数据集
├── extract_cnn_vgg16_keras.py 使用预训练vgg16模型提取图像特征
|── index.py 对图像集提取特征，建立索引
├── query_online.py 库内搜索
└── README.md
```

#### 示例

```sh
# 对database文件夹内图片进行特征提取，建立索引文件featureCNN.h5
python index.py -database database -index featureCNN.h5

# 使用database文件夹内001_accordion_image_0001.jpg作为测试图片，在database内以featureCNN.h5进行近似图片查找，并显示最近似的3张图片
python query_online.py -query database/001_accordion_image_0001.jpg -index featureCNN.h5 -result database
```

#### Flask Web Restful 接口

`python web_restful.py`

#### 相关资料

##### 图像特征提取算法

常用的图像特征提取算法有 `SIFT`, `SURF`, `CNN` `HOG`特征，`LBP`特征，`Haar`特征

##### 特征举证相似度的计算方法

计算两个特征矩阵之间的余弦相似度。
高维数据的快速最近邻算法FLANN，构造kd Tree。优先搜索k-means树算法，层次聚类树，
设计高可用的哈希算法，把高维矩阵映射到哈希值上，LSH(Locality Sensitive Hashing)局部敏感哈希算法

https://blog.csdn.net/jinxueliu31/article/details/37768995
https://blog.csdn.net/wonner_/article/details/80985727
https://blog.csdn.net/guoziqing506/article/details/53019049
https://waltyou.github.io/Faiss-In-Project/
