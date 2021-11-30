# -*- coding: utf-8 -*-
import tensorflow as tf
from scipy import misc
import facenet
import numpy as np

#FaceNet网络类
class FaceNetModel():
    #类的构造函数
    def __init__(self, model_file):
        #实例化一个数据流图做为运行环境的默认数据流图
        tf.Graph().as_default()

        #根据指定的FaceNet网络模型文件构建数据流图
        facenet.load_model('20180402-114759.pb')
    
        #设定网络模型的输入张量至相应的占位符
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        #设定网络模型的输出张量至前向传输操作
        self.embeddings_op = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        #创建一个会话流程启动数据流图
        self.sess = tf.Session()
        
    #通过完成前向传输运算获取描述特征
    def get_decriptor(self, image):
        #将输入图像变换到指定尺寸
        image = misc.imresize(image, (160, 160), interp='bilinear')
        #对输入图像做白化处理
        image = facenet.prewhiten(image)
        #增加输入数据的维度
        images = np.stack([image])

        #为会话流程中的占位符赋值(设定images为输入图像/设定训练标记为否)
        feed_dict = { self.images_placeholder: images, self.phase_train_placeholder:False }
        #运行会话流程计算embeddings_op的值
        emb = self.sess.run(self.embeddings_op, feed_dict=feed_dict)
        #返回从图像中获取的描述特征向量
        return emb[0,:]
        
    #类的析构函数
    def __del__(self):
        #关闭会话流程
        self.sess.close()

#创建FaceNet网络
def get_FaceNetModel(model_file):
    return FaceNetModel(model_file)
