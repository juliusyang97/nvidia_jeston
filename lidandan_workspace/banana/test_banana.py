﻿# -*- coding:utf-8 -*-
import caffe
import numpy as np
import cv2
import time

#网络结构文件
deploy_file = 'deploy.prototxt'
#预训练模型文件
model_file = 'snapshot_iter_40.caffemodel'
#网络输入层的名称
in_name = 'data'
#网络输出层的名称
out_name = 'score'

#载入分类网络
fcn_net = caffe.Net(deploy_file, model_file, caffe.TEST)

#打开摄像头
cap=cv2.VideoCapture(0)
#设定摄像头采集分辨率为320x240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

#如果摄像头成功打开
while(cap.isOpened()):
    #读取摄像头采集的当前帧图片数据
    success, frame=cap.read()
    #如果成功读取摄像头
    if (success):
        #开始计时
        time0 = time.time()

        #转换图像矩阵维度及数据类型与分类网络输入层相符
        image_data = np.array(frame, dtype = np.float32)
        image_data = image_data.transpose([2,0,1])

        #设定分类网络的输入数据
        fcn_net.blobs[in_name].data[...] = image_data
        #分类网络完成前向推理
        fcn_net.forward()

        #取出分类网络score层的数据（每个像素属于各个类的概率）
        prediction = fcn_net.blobs[out_name].data[0]
        #逐个像素找出概率最大的分类做为分类结果
        result = prediction.argmax(axis=0)

        #输出本次推理耗时(单位是秒)
        print("Inference Cost" , time.time()-time0)

        #根据像素级的分类结果(图像分割结果)生成红色掩膜
        mask = np.zeros(frame.shape, dtype='uint8')
        mask[:,:,2] = result.astype(np.uint8) * 255

        #将图像分割掩膜和原图叠加并显示
        show = cv2.add(frame,mask)
        cv2.imshow('test_banana', show)

        #如果用户按下取消键则退出循环
        if ( cv2.waitKey(1) & 0xFF == 27 ): 
            break
    #如果读取摄像头失败
    else:
        break

#释放摄像头
cap.release()
