import tensorflow as tf
import math
import time
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
images = tf.reshape(x, [-1,28,28,1])

keep_prob = tf.placeholder(tf.float32)


# 在函数inference_op()内定义前向传播的过程
def inference_op(images,keep_prob):
    parameters = []

    # 在命名空间conv1下实现第一个卷积层
    with tf.name_scope("conv1"):
        ## 初始化核参数
        kernel = tf.Variable(tf.truncated_normal([2, 2, 1, 96], dtype=tf.float32,stddev=1e-1), name="weights")
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding="SAME")
        ## 初始化偏置
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),trainable=True, name="biases")
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases))

        conv1 = tf.nn.dropout(conv1, keep_prob)
        # 打印第一个卷积层的网络结构
        print(conv1.op.name, ' ', conv1.get_shape().as_list())

        parameters += [kernel, biases]

    # 添加一个LRN层和最大池化层
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn1")
    pool1 = tf.nn.max_pool2d(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="VALID", name="pool1")

    # 打印池化层网络结构
    print(pool1.op.name, ' ', pool1.get_shape().as_list())

    # 在命名空间conv2下实现第二个卷积层
    with tf.name_scope("conv2"):
        kernel = tf.Variable(tf.truncated_normal([2, 2, 96, 256], dtype=tf.float32,stddev=1e-1), name="weights")
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name="biases")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        conv12 = tf.nn.dropout(conv2, keep_prob)
        parameters += [kernel, biases]

        # 打印第二个卷积层的网络结构
        print(conv2.op.name, ' ', conv2.get_shape().as_list())

    # 添加一个LRN层和最大池化层
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn2")
    pool2 = tf.nn.max_pool2d(lrn2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                           padding="VALID", name="pool2")
    # 打印池化层的网络结构
    print(pool2.op.name, ' ', pool2.get_shape().as_list())

    # 在命名空间conv3下实现第三个卷积层
    with tf.name_scope("conv3"):
        kernel = tf.Variable(tf.truncated_normal([2,2, 256, 384],
                                                 dtype=tf.float32, stddev=1e-1),
                             name="weights")
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name="biases")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        conv3 = tf.nn.dropout(conv3, keep_prob)
        parameters += [kernel, biases]

        # 打印第三个卷积层的网络结构
        print(conv3.op.name, ' ', conv3.get_shape().as_list())

    # 在命名空间conv4下实现第四个卷积层
    with tf.name_scope("conv4"):
        kernel = tf.Variable(tf.truncated_normal([2, 2, 384, 384],
                                                 dtype=tf.float32, stddev=1e-1),
                             name="weights")
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name="biases")
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        conv4 = tf.nn.dropout(conv4, keep_prob)
        parameters += [kernel, biases]

        # 打印第四个卷积层的网络结构
        print(conv4.op.name, ' ', conv4.get_shape().as_list())

    # 在命名空间conv5下实现第五个卷积层
    with tf.name_scope("conv5"):
        kernel = tf.Variable(tf.truncated_normal([2, 2, 384, 256],
                                                 dtype=tf.float32, stddev=1e-1),
                             name="weights")
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name="biases")

        conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        conv5 = tf.nn.dropout(conv5, keep_prob)
        parameters += [kernel, biases]

        # 打印第五个卷积层的网络结构
        print(conv5.op.name, ' ', conv5.get_shape().as_list())

    # 添加一个最大池化层

    pool5 = tf.nn.max_pool2d(conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                           padding="VALID", name="pool5")
    # 打印最大池化层的网络结构
    print(pool5.op.name, ' ', pool5.get_shape().as_list())

    #pool5输出4*4*256
    # 将pool5输出的矩阵汇总为向量的形式，为的是方便作为全连层的输入
    pool_shape = pool5.get_shape().as_list()#返回[btch_size,4,4,256]
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool5,[-1,nodes])
    print(reshaped.op.name, ' ', reshaped.get_shape().as_list())
    #reshaped=tf.layers.flatten(pool5)

    # 创建第一个全连接层
    with tf.name_scope("fc_1"):
        fc1_weights = tf.Variable(tf.truncated_normal([nodes, 4096], dtype=tf.float32,stddev=1e-1), name="weights")
        fc1_bias = tf.Variable(tf.constant(0.0, shape=[4096],dtype=tf.float32), trainable=True, name="biases")
        fc_1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_bias)
        fc_1 = tf.nn.dropout(fc_1, keep_prob)
        parameters += [fc1_weights, fc1_bias]

        # 打印第一个全连接层的网络结构信息
        print(fc_1.op.name, ' ', fc_1.get_shape().as_list())

    # 创建第二个全连接层
    with tf.name_scope("fc_2"):
        fc2_weights = tf.Variable(tf.truncated_normal([4096, 10], dtype=tf.float32,
                                                      stddev=1e-1), name="weights")
        fc2_bias = tf.Variable(tf.constant(0.0, shape=[10],
                                           dtype=tf.float32), trainable=True, name="biases")
        fc_2 = tf.add(tf.matmul(fc_1, fc2_weights),fc2_bias)
        fc_2 = tf.nn.dropout(fc_2, keep_prob)
        parameters += [fc2_weights, fc2_bias]

        # 打印第二个全连接层的网络结构信息
        print(fc_2.op.name, ' ', fc_2.get_shape().as_list())

    y_conv = tf.nn.softmax(fc_2)

    # 返回全连接层处理的结果
    return y_conv, parameters

y_conv, parameters = inference_op(images,keep_prob)

print (parameters)
# 计算代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化全局变量
tf.global_variables_initializer().run()


for i in range(20000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob:0.75})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        #cross_entropy=sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})
        #print ('step %d, loss %f'%(i, cross_entropy))
    #loss=sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #print (loss)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1}))


