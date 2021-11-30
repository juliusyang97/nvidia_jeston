# -*- coding:utf-8 -*-
import cv2
import caffe

#网络结构文件
deploy_file = 'model/deploy.prototxt'
#预训练模型文件
model_file = 'model/snapshot_iter_4690.caffemodel'
#平均图片文件
mean_file = 'model/mean.binaryproto'
#网络输入层的名称
in_name = 'data'
#网络输出层的名称
out_name = 'softmax'

#载入平均图片
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_file, 'rb').read())
#转换平均图片数据类型为数列
mnist_mean = caffe.io.blobproto_to_array(mean_blob)
#载入分类网络
minist_net = caffe.Net(deploy_file, model_file, caffe.TEST)

#依序遍历十张图片
for i in range(10):
    #图片文件
    image_file = 'mytest/' + str(i) + '.png'
    #以灰度模式读取图片文件
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    #变换图像矩阵尺寸与分类网络输入层相符
    image_resize = cv2.resize(image,(28,28))
    #转换图像矩阵维度及数据类型与分类网络输入层相符
    image_data = image_resize.reshape((1, 1, 28, 28)).astype(float)

    #设定分类网络的输入数据为图像数据减去平均图片
    minist_net.blobs[in_name].data[...] = image_data - mnist_mean
    #分类网络完成前向推理
    minist_net.forward()
    #取出分类网络prob层的数据（各个分类的概率列表）
    prediction = minist_net.blobs[out_name].data[0]
    #找出概率最大的分类做为分类结果
    pre_class = prediction.argmax()
    #打印出分类结果及其概率
    print pre_class, prediction[pre_class]

    #将图像从灰度模式变换为彩色模式
    image_rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #将分类结果以文本形式写在图像左上角
    cv2.putText(image_rgb, str(pre_class), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255,0,0), 2, cv2.LINE_AA)
    #显示写有分类结果文本的图像
    cv2.imshow("handwriting", image_rgb)
    #等待用户敲击键盘
    cv2.waitKey(0)
