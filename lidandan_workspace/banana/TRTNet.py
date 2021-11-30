# -*- coding:utf-8 -*-
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os

#TRT网络类
class TRTNet():
    #类的初始化函数
    def __init__(self, engine_file, deploy_file, model_file, out_name):
        #设定LOG信息等级(等级越高记录的信息越少)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        #如果已经存在序列化的加速引擎文件
        if os.path.exists(engine_file):
            print('Deserializing trt engine from ' + engine_file + '...')
            #读取序列化的加速引擎文件解序为加速引擎
            with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                
        #如果不存在序列化的加速引擎文件
        else:
            #TRT加速引擎构建器
            builder = trt.Builder(TRT_LOGGER)
            #设定构建器最大可用内存(越大越好)
            builder.max_workspace_size = 1 << 30 #1G

            print('Parsing caffe model into trt engine...')
            #加载并解析Caffe网络结构文件及预训练模型文件
            network = builder.create_network()
            parser = trt.CaffeParser()
            model_tensors = parser.parse(deploy=deploy_file, 
                                         model=model_file, 
                                         network=network, dtype=trt.float32)
            #设定Caffe网络的输出层名称
            network.mark_output(model_tensors.find('score'))
            #从Caffe网络结构文件及预训练模型中构建加速引擎
            engine = builder.build_cuda_engine(network)

            print('Serializing trt engine into ' + engine_file + '...')
            #将构建好的引擎序列化之后存储到指定文件中
            with open(engine_file, 'wb') as f:
                f.write(engine.serialize())
        print('Done!')
        #网络输出层的尺寸
        self.out_shape = engine.get_binding_shape(1)

        #确定维度并为输入和输出创建分页锁定内存缓冲区(即不会切换到磁盘)
        self.h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), 
                                             dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), 
                                              dtype=np.float32)

        #为输入和输出分配设备存储空间(显存)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        #指定使用CUDA流进行输入输出的复制并进行推理
        self.stream = cuda.Stream()
        #创建执行推理的上下文
        self.context = engine.create_execution_context()

    #推理函数
    def Inference(self):
        #将输入数据传送给GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        #执行推理过程
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], 
                                   stream_handle=self.stream.handle)
        #从GPU取出输出数据
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        #同步流操作
        self.stream.synchronize()

#创建加速引擎网络
def get_TRTNet(engine_file, deploy_file='', model_file='', out_name=''):
    return TRTNet(engine_file, deploy_file, model_file, out_name)
