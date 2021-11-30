#-*-coding:utf-8-*-
import numpy as np
import tensorflow as tf

#只有8个单词的小字典
small_dict=['EOS','a','my','sleeps','on','dog','cat','the','bed','floor'] 
#只有2个句子的小语库
X=np.array([[2,6,3,4,2,8,0],[1,5,3,4,7,9,0]],dtype=np.int32) 

#迭代次数
epochs=300
#绘制loss曲线的张量
plot_loss=[]
#隐层单元数
num_hidden=24
#一个样本的序列长度（batch的列数）
num_steps=X.shape[1]
#词典长度
dict_length=len(small_dict)
#一个batch中样本的数量（batch的行数）
batch_size=2
#重置图表
tf.reset_default_graph()

#随机初始化权重张量和偏置张量
#tf.truncated_normal从截断的正态分布中输出随机值
variables_dict = {
    "weights1": tf.Variable(tf.truncated_normal([num_hidden,dict_length], stddev=1.0,dtype=tf.float32),name="weights1"),
    "biases1": tf.Variable(tf.truncated_normal([dict_length],stddev=1.0,dtype=tf.float32), name="biases1")}

#生成X的独热编码
X_one_hot=tf.nn.embedding_lookup(np.identity(dict_length), X) #[batch,num_steps,dictionary_length][2,6,7]
y=np.zeros((batch_size,num_steps),dtype=np.int32)
#y是X左移一列，空列以0补齐
y[:,:-1]=X[:,1:]
#生成y的独热编码
y_one_hot=tf.unstack(tf.nn.embedding_lookup(np.identity(dict_length), y),num_steps,1) #[batch,num_steps,dictionary_length][2,6,7]
#变换y的独热编码的维度
y_target_reshape=tf.reshape(y_one_hot,[batch_size*num_steps,dict_length])

#创建LSTM神经元
#cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
num_layers=4
dropout=0.5
layer_cell=[]
for _ in range(num_layers):
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                       input_keep_prob=dropout,
                                       output_keep_prob=dropout)
    layer_cell.append(lstm_cell)
cell = tf.contrib.rnn.MultiRNNCell(layer_cell, state_is_tuple=True)

#使用指定的RNN神经元创建单层循环神经网络
outputs, last_states = tf.contrib.rnn.static_rnn(
    cell=cell,
    dtype=tf.float32,
    inputs=tf.unstack(tf.to_float(X_one_hot),num_steps,1))

#变换output的维度以适配张量乘法
output_reshape=tf.reshape(outputs, [batch_size*num_steps,num_hidden]) #[12==batch_size*num_steps,num_hidden==12]
#p=w*output+b
pred=tf.matmul(output_reshape, variables_dict["weights1"]) +variables_dict["biases1"]
#定义损失函数为交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_target_reshape))
#定义Adam为优化器
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

#初始化变量操作
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())    

plot_loss=[]

with tf.Session() as sess:
    #初始化变量
    sess.run(init_op)
    #初始化协调器
    coord = tf.train.Coordinator()
    #初始化线程
    threads = tf.train.start_queue_runners(coord=coord)       
    #逐个迭代循环
    for i in range(epochs):   
        #反向传播更新权重
        loss,_,y_target,y_pred,output =  sess.run([cost,optimizer,y_target_reshape,pred,outputs])
        #缓存loss曲线数据
        plot_loss.append([loss])
        if i% 25 ==0:
            print("iteration: ",i," loss: ",loss)
                
    coord.request_stop()
    coord.join(threads)
    sess.close()  

#测试训练集第0个句子
print("Input Sentence")
print([small_dict[ind] for ind in X[0,:]])
#对于训练集句子中每个词预测的下一个词
print("Predicted Words")
print([small_dict[ind] for ind in np.argmax(y_pred[0::2],1)])
