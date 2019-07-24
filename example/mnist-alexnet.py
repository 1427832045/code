import tensorflow as tf
import math
import time
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10])
images = tf.reshape(x, [-1,28,28,1])

tf.summary.image('input_images', images, 10)

keep_prob = tf.placeholder(tf.float32,name='keep_prob')


# 在函数inference_op()内定义前向传播的过程
def inference_op(images,keep_prob):
    parameters = []

    # 在命名空间conv1下实现第一个卷积层
    with tf.name_scope("conv1"):
        ## 初始化核参数
        kernel = tf.Variable(tf.truncated_normal([2, 2, 1, 96], dtype=tf.float32,stddev=1e-1), name="weights")
        #tf.contrib.layers.l2_regularizer(0.001)(kernel)
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding="SAME")
        ## 初始化偏置
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),trainable=True, name="biases")
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases))

        #conv1 = tf.nn.dropout(conv1, keep_prob)
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
        #tf.contrib.layers.l2_regularizer(0.001)(kernel)
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name="biases")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        #conv12 = tf.nn.dropout(conv2, keep_prob)
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
        #tf.contrib.layers.l2_regularizer(0.001)(kernel)
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name="biases")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        #conv3 = tf.nn.dropout(conv3, keep_prob)
        parameters += [kernel, biases]

        # 打印第三个卷积层的网络结构
        print(conv3.op.name, ' ', conv3.get_shape().as_list())

    # 在命名空间conv4下实现第四个卷积层
    with tf.name_scope("conv4"):
        kernel = tf.Variable(tf.truncated_normal([2, 2, 384, 384],
                                                 dtype=tf.float32, stddev=1e-1),
                             name="weights")
        #tf.contrib.layers.l2_regularizer(0.001)(kernel)
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")

        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name="biases")
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        #conv4 = tf.nn.dropout(conv4, keep_prob)
        parameters += [kernel, biases]

        # 打印第四个卷积层的网络结构
        print(conv4.op.name, ' ', conv4.get_shape().as_list())

    # 在命名空间conv5下实现第五个卷积层
    with tf.name_scope("conv5"):
        kernel = tf.Variable(tf.truncated_normal([2, 2, 384, 256],
                                                 dtype=tf.float32, stddev=1e-1),
                             name="weights")
        #tf.contrib.layers.l2_regularizer(0.001)(kernel)
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name="biases")

        conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        #conv5 = tf.nn.dropout(conv5, keep_prob)
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

    # with tf.name_scope("fc_3"):
    #     fc3_weights = tf.Variable(tf.truncated_normal([2048, 10], dtype=tf.float32,
    #                                                   stddev=1e-1), name="weights")
    #     fc3_bias = tf.Variable(tf.constant(0.0, shape=[10],
    #                                        dtype=tf.float32), trainable=True, name="biases")
    #     fc_3 = tf.add(tf.matmul(fc_2, fc3_weights),fc3_bias)
    #     fc_3 = tf.nn.dropout(fc_3, keep_prob)
    #     parameters += [fc3_weights, fc3_bias]

    #     # 打印第二个全连接层的网络结构信息
    #     print(fc_3.op.name, ' ', fc_3.get_shape().as_list())

    y_conv = tf.nn.softmax(fc_2)

    # 返回全连接层处理的结果
    return y_conv, parameters

#global_step = tf.Variable(0)
#learning_rate = tf.train.exponential_decay(0.8, global_step, 100, 0.1, staircase=True)

y_conv, parameters = inference_op(images,keep_prob)
tf.add_to_collection('pred', y_conv)
#print (parameters)
# 计算代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv+1e-10), reduction_indices=[1]))
# 使用AdamOptimizer进行优化
#tf.add_to_collection('losses', cross_entropy)
#loss = tf.add_n(tf.get_collection('losses'))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.AdagradOptimizer(1e-3).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# tf.summary.scalar("validation_accuracy", accuracy)

# 写到指定的磁盘路径中

from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'tensorboard/train/' + TIMESTAMP
writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)


# test_log_dir = 'tensorboard/test/' + TIMESTAMP
# writer_test = tf.summary.FileWriter(test_log_dir)

# val_log_dir = 'tensorboard/val/' + TIMESTAMP
# writer_val = tf.summary.FileWriter(val_log_dir)

merged = tf.summary.merge_all()
# merge_summary1 = tf.summary.merge(train_accuracy)
# merge_summary2 = tf.summary.merge(validation_accuracy)

# 初始化全局变量
tf.global_variables_initializer().run()


##训练过程
saver=tf.train.Saver(max_to_keep=1)
max_acc=0

for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob:0.9})
        #print('step %d, training accuracy %g' % (i, train_accuracy))
        loss_value = sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})
        print ('After step %d, training accuracy %g' % (i, train_accuracy),'training loss is %f'%(loss_value))
        summary, _ = sess.run([merged, train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})
        writer_train.add_summary(summary, i)

        if train_accuracy>max_acc:
            max_acc=train_accuracy
            saver.save(sess,"Model_save/mnist-alexnet-best-model.ckpt",global_step=i+1)


        #测试过程
batch = mnist.test.next_batch(100)
test_accuracy=accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob:1})
# print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1}))
# _, test_loss_value = sess.run([train_step, cross_entropy],feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1})
test_loss_value = sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1],keep_prob:1})
print ('test accuracy %g' % (test_accuracy),',test loss is %f'%(test_loss_value))
        # summary2, _ = sess.run([merge_summary2, train_step],feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1})
        # writer_test.add_summary(summary2)


        #验证过程
x_val, y_val = mnist.validation.next_batch(100)
validate_feed = {x: x_val, y_: y_val,keep_prob: 1.0}
validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
val_loss_value = sess.run(cross_entropy,feed_dict=validate_feed)
print ('validation accuracy is %g' % (validate_accuracy),',val loss is %f'%(val_loss_value))
        # summary, validate_accuracy = sess.run([merged, validate_accuracy],feed_dict=validate_feed)
        # writer_val.add_summary(summary,i)
        #cross_entropy=sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})
        #print ('step %d, loss %f'%(i, cross_entropy))
    #loss=sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #print (loss)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#saver.save(sess, "Model_save/mnist-alexnet-model3.ckpt")

##训练过程
# After step 0, training accuracy 0.08 training loss is 17.736095
# After step 100, training accuracy 0.05 training loss is 20.160532
# After step 200, training accuracy 0.04 training loss is 21.457884
# After step 300, training accuracy 0.08 training loss is 20.909039
# After step 400, training accuracy 0.1 training loss is 21.183775
# W0724 15:46:59.093833 47833230480256 deprecation.py:323] From /share/home/flt/flt_data/python3/lib/python3.6/site-packages/tensorflow/python/training/saver.py:960: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use standard file APIs to delete files with this prefix.
# After step 500, training accuracy 0.13 training loss is 20.493006
# After step 600, training accuracy 0.08 training loss is 20.652809
# After step 700, training accuracy 0.1 training loss is 20.492990
# After step 800, training accuracy 0.12 training loss is 20.373852
# After step 900, training accuracy 0.1 training loss is 19.947857
# After step 1000, training accuracy 0.17 training loss is 18.889069
# After step 1100, training accuracy 0.14 training loss is 19.987909
# After step 1200, training accuracy 0.16 training loss is 19.230265
# After step 1300, training accuracy 0.1 training loss is 20.491203
# After step 1400, training accuracy 0.12 training loss is 20.109844
# After step 1500, training accuracy 0.13 training loss is 19.982796
# After step 1600, training accuracy 0.17 training loss is 19.620068
# After step 1700, training accuracy 0.09 training loss is 20.978941
# After step 1800, training accuracy 0.07 training loss is 21.204811
# After step 1900, training accuracy 0.1 training loss is 20.262749
# After step 2000, training accuracy 0.11 training loss is 19.871660
# After step 2100, training accuracy 0.11 training loss is 20.159481
# After step 2200, training accuracy 0.13 training loss is 19.216852
# After step 2300, training accuracy 0.12 training loss is 20.656723
# After step 2400, training accuracy 0.12 training loss is 18.392149
# After step 2500, training accuracy 0.16 training loss is 18.509066
# After step 2600, training accuracy 0.27 training loss is 16.681864
# After step 2700, training accuracy 0.27 training loss is 16.378202
# After step 2800, training accuracy 0.24 training loss is 17.855186
# After step 2900, training accuracy 0.23 training loss is 16.645983
# After step 3000, training accuracy 0.17 training loss is 16.842300
# After step 3100, training accuracy 0.3 training loss is 16.428062
# After step 3200, training accuracy 0.15 training loss is 19.172789
# After step 3300, training accuracy 0.16 training loss is 18.649330
# After step 3400, training accuracy 0.14 training loss is 19.091053
# After step 3500, training accuracy 0.21 training loss is 16.927349
# After step 3600, training accuracy 0.18 training loss is 18.649342
# After step 3700, training accuracy 0.16 training loss is 19.807447
# After step 3800, training accuracy 0.12 training loss is 19.333038
# After step 3900, training accuracy 0.16 training loss is 18.991861
# After step 4000, training accuracy 0.11 training loss is 18.943796
# After step 4100, training accuracy 0.18 training loss is 17.834969
# After step 4200, training accuracy 0.29 training loss is 16.461004
# After step 4300, training accuracy 0.23 training loss is 17.454000
# After step 4400, training accuracy 0.18 training loss is 18.088390
# After step 4500, training accuracy 0.2 training loss is 18.190601
# After step 4600, training accuracy 0.16 training loss is 19.187857
# After step 4700, training accuracy 0.23 training loss is 17.621605
# After step 4800, training accuracy 0.18 training loss is 18.224733
# After step 4900, training accuracy 0.31 training loss is 15.775989
# After step 5000, training accuracy 0.25 training loss is 16.053364
# After step 5100, training accuracy 0.21 training loss is 17.985643
# After step 5200, training accuracy 0.29 training loss is 16.470892
# After step 5300, training accuracy 0.21 training loss is 17.825048
# After step 5400, training accuracy 0.18 training loss is 18.806517
# After step 5500, training accuracy 0.23 training loss is 17.178345
# After step 5600, training accuracy 0.21 training loss is 17.176056
# After step 5700, training accuracy 0.26 training loss is 16.757114
# After step 5800, training accuracy 0.2 training loss is 17.197386
# After step 5900, training accuracy 0.27 training loss is 16.473433
# After step 6000, training accuracy 0.32 training loss is 15.717622
# After step 6100, training accuracy 0.22 training loss is 17.415268
# After step 6200, training accuracy 0.19 training loss is 17.988022
# After step 6300, training accuracy 0.2 training loss is 18.769417
# After step 6400, training accuracy 0.15 training loss is 18.802408
# After step 6500, training accuracy 0.27 training loss is 16.778711
# After step 6600, training accuracy 0.26 training loss is 17.559298
# After step 6700, training accuracy 0.34 training loss is 15.595689
# After step 6800, training accuracy 0.27 training loss is 15.847251
# After step 6900, training accuracy 0.31 training loss is 15.704062
# After step 7000, training accuracy 0.23 training loss is 17.603964
# After step 7100, training accuracy 0.24 training loss is 17.868847
# After step 7200, training accuracy 0.12 training loss is 19.131874
# After step 7300, training accuracy 0.36 training loss is 13.246341
# After step 7400, training accuracy 0.26 training loss is 16.349833
# After step 7500, training accuracy 0.39 training loss is 13.818157
# After step 7600, training accuracy 0.25 training loss is 16.388950
# After step 7700, training accuracy 0.23 training loss is 15.516763
# After step 7800, training accuracy 0.19 training loss is 18.292051
# After step 7900, training accuracy 0.25 training loss is 16.014223
# After step 8000, training accuracy 0.33 training loss is 15.942495
# After step 8100, training accuracy 0.24 training loss is 15.601926
# After step 8200, training accuracy 0.28 training loss is 13.742344
# After step 8300, training accuracy 0.14 training loss is 17.764210
# After step 8400, training accuracy 0.21 training loss is 18.056938
# After step 8500, training accuracy 0.21 training loss is 17.323259
# After step 8600, training accuracy 0.35 training loss is 13.350826
# After step 8700, training accuracy 0.24 training loss is 17.780815
# After step 8800, training accuracy 0.37 training loss is 14.645062
# After step 8900, training accuracy 0.25 training loss is 15.438651
# After step 9000, training accuracy 0.3 training loss is 14.258685
# After step 9100, training accuracy 0.17 training loss is 17.235603
# After step 9200, training accuracy 0.24 training loss is 16.544779
# After step 9300, training accuracy 0.35 training loss is 15.424758
# After step 9400, training accuracy 0.26 training loss is 16.697565
# After step 9500, training accuracy 0.25 training loss is 15.939940
# After step 9600, training accuracy 0.33 training loss is 15.316041
# After step 9700, training accuracy 0.29 training loss is 15.969722
# After step 9800, training accuracy 0.38 training loss is 13.062436
# After step 9900, training accuracy 0.25 training loss is 16.055300
# After step 10000, training accuracy 0.3 training loss is 13.375650
# After step 10100, training accuracy 0.18 training loss is 16.345476
# After step 10200, training accuracy 0.24 training loss is 15.270555
# After step 10300, training accuracy 0.34 training loss is 13.417302
# After step 10400, training accuracy 0.23 training loss is 15.635087
# After step 10500, training accuracy 0.26 training loss is 16.391380
# After step 10600, training accuracy 0.3 training loss is 15.170972
# After step 10700, training accuracy 0.23 training loss is 13.958658
# After step 10800, training accuracy 0.21 training loss is 16.209816
# After step 10900, training accuracy 0.39 training loss is 12.522318
# After step 11000, training accuracy 0.34 training loss is 13.060935
# After step 11100, training accuracy 0.33 training loss is 14.003781
# After step 11200, training accuracy 0.24 training loss is 14.424706
# After step 11300, training accuracy 0.39 training loss is 11.724072
# After step 11400, training accuracy 0.3 training loss is 14.601559
# After step 11500, training accuracy 0.37 training loss is 11.090340
# After step 11600, training accuracy 0.38 training loss is 12.118400
# After step 11700, training accuracy 0.43 training loss is 10.365143
# After step 11800, training accuracy 0.35 training loss is 11.489926
# After step 11900, training accuracy 0.27 training loss is 9.908133
# After step 12000, training accuracy 0.34 training loss is 11.938272
# After step 12100, training accuracy 0.28 training loss is 12.435469
# After step 12200, training accuracy 0.22 training loss is 12.589588
# After step 12300, training accuracy 0.33 training loss is 11.512057
# After step 12400, training accuracy 0.32 training loss is 9.782860
# After step 12500, training accuracy 0.33 training loss is 12.303653
# After step 12600, training accuracy 0.39 training loss is 10.893354
# After step 12700, training accuracy 0.35 training loss is 12.500114
# After step 12800, training accuracy 0.35 training loss is 12.350081
# After step 12900, training accuracy 0.3 training loss is 10.912044
# After step 13000, training accuracy 0.37 training loss is 9.973323
# After step 13100, training accuracy 0.41 training loss is 10.522859
# After step 13200, training accuracy 0.42 training loss is 9.206843
# After step 13300, training accuracy 0.35 training loss is 9.332168
# After step 13400, training accuracy 0.3 training loss is 9.885763
# After step 13500, training accuracy 0.48 training loss is 7.753685
# After step 13600, training accuracy 0.46 training loss is 8.497909
# After step 13700, training accuracy 0.47 training loss is 7.246444
# After step 13800, training accuracy 0.55 training loss is 7.199606
# After step 13900, training accuracy 0.39 training loss is 9.681015
# After step 14000, training accuracy 0.49 training loss is 7.191450
# After step 14100, training accuracy 0.46 training loss is 8.248914
# After step 14200, training accuracy 0.45 training loss is 8.787733
# After step 14300, training accuracy 0.38 training loss is 9.410194
# After step 14400, training accuracy 0.54 training loss is 7.268569
# After step 14500, training accuracy 0.51 training loss is 7.735691
# After step 14600, training accuracy 0.5 training loss is 7.911113
# After step 14700, training accuracy 0.45 training loss is 7.724850
# After step 14800, training accuracy 0.49 training loss is 6.838945
# After step 14900, training accuracy 0.52 training loss is 6.807449
# After step 15000, training accuracy 0.47 training loss is 7.442217
# After step 15100, training accuracy 0.47 training loss is 7.019479
# After step 15200, training accuracy 0.44 training loss is 7.163089
# After step 15300, training accuracy 0.38 training loss is 7.915998
# After step 15400, training accuracy 0.52 training loss is 6.883517
# After step 15500, training accuracy 0.6 training loss is 5.207537
# After step 15600, training accuracy 0.51 training loss is 7.629001
# After step 15700, training accuracy 0.45 training loss is 7.804983
# After step 15800, training accuracy 0.54 training loss is 5.126070
# After step 15900, training accuracy 0.49 training loss is 6.683472
# After step 16000, training accuracy 0.54 training loss is 6.535808
# After step 16100, training accuracy 0.59 training loss is 5.801793
# After step 16200, training accuracy 0.55 training loss is 5.793479
# After step 16300, training accuracy 0.57 training loss is 4.898628
# After step 16400, training accuracy 0.56 training loss is 5.964867
# After step 16500, training accuracy 0.45 training loss is 4.915380
# After step 16600, training accuracy 0.5 training loss is 6.577002
# After step 16700, training accuracy 0.45 training loss is 8.666220
# After step 16800, training accuracy 0.51 training loss is 7.133719
# After step 16900, training accuracy 0.66 training loss is 3.828265
# After step 17000, training accuracy 0.59 training loss is 5.518208
# After step 17100, training accuracy 0.54 training loss is 5.595361
# After step 17200, training accuracy 0.5 training loss is 6.036848
# After step 17300, training accuracy 0.59 training loss is 3.998649
# After step 17400, training accuracy 0.56 training loss is 5.053585
# After step 17500, training accuracy 0.52 training loss is 3.855441
# After step 17600, training accuracy 0.47 training loss is 5.733597
# After step 17700, training accuracy 0.48 training loss is 7.179093
# After step 17800, training accuracy 0.5 training loss is 7.064246
# After step 17900, training accuracy 0.52 training loss is 6.785619
# After step 18000, training accuracy 0.63 training loss is 4.637827
# After step 18100, training accuracy 0.57 training loss is 4.794792
# After step 18200, training accuracy 0.48 training loss is 6.016166
# After step 18300, training accuracy 0.53 training loss is 5.485583
# After step 18400, training accuracy 0.53 training loss is 5.451005
# After step 18500, training accuracy 0.6 training loss is 3.275996
# After step 18600, training accuracy 0.56 training loss is 5.071087
# After step 18700, training accuracy 0.59 training loss is 4.206409
# After step 18800, training accuracy 0.63 training loss is 3.726809
# After step 18900, training accuracy 0.53 training loss is 3.776127
# After step 19000, training accuracy 0.54 training loss is 3.587554
# After step 19100, training accuracy 0.52 training loss is 4.419050
# After step 19200, training accuracy 0.65 training loss is 2.744479
# After step 19300, training accuracy 0.69 training loss is 2.146461
# After step 19400, training accuracy 0.65 training loss is 3.951783
# After step 19500, training accuracy 0.5 training loss is 3.116653
# After step 19600, training accuracy 0.42 training loss is 4.622154
# After step 19700, training accuracy 0.58 training loss is 4.686757
# After step 19800, training accuracy 0.56 training loss is 4.964306
# After step 19900, training accuracy 0.55 training loss is 5.224113
# After step 20000, training accuracy 0.66 training loss is 3.095155
# After step 20100, training accuracy 0.56 training loss is 3.577125
# After step 20200, training accuracy 0.68 training loss is 2.733826
# After step 20300, training accuracy 0.54 training loss is 2.963385
# After step 20400, training accuracy 0.52 training loss is 3.700491
# After step 20500, training accuracy 0.58 training loss is 2.930607
# After step 20600, training accuracy 0.59 training loss is 2.900882
# After step 20700, training accuracy 0.67 training loss is 2.240169
# After step 20800, training accuracy 0.62 training loss is 2.372167
# After step 20900, training accuracy 0.63 training loss is 2.561781
# After step 21000, training accuracy 0.58 training loss is 2.974546
# After step 21100, training accuracy 0.64 training loss is 2.222404
# After step 21200, training accuracy 0.71 training loss is 1.757085
# After step 21300, training accuracy 0.6 training loss is 3.239111
# After step 21400, training accuracy 0.63 training loss is 1.705203
# After step 21500, training accuracy 0.65 training loss is 2.383682
# After step 21600, training accuracy 0.68 training loss is 1.572957
# After step 21700, training accuracy 0.68 training loss is 1.792883
# After step 21800, training accuracy 0.59 training loss is 2.598166
# After step 21900, training accuracy 0.59 training loss is 2.384406
# After step 22000, training accuracy 0.59 training loss is 3.003453
# After step 22100, training accuracy 0.63 training loss is 1.937226
# After step 22200, training accuracy 0.71 training loss is 1.936393
# After step 22300, training accuracy 0.63 training loss is 2.108016
# After step 22400, training accuracy 0.64 training loss is 2.262949
# After step 22500, training accuracy 0.58 training loss is 2.231676
# After step 22600, training accuracy 0.56 training loss is 1.956404
# After step 22700, training accuracy 0.58 training loss is 2.575101
# After step 22800, training accuracy 0.7 training loss is 1.765916
# After step 22900, training accuracy 0.6 training loss is 1.981088
# After step 23000, training accuracy 0.59 training loss is 2.014808
# After step 23100, training accuracy 0.67 training loss is 2.279468
# After step 23200, training accuracy 0.67 training loss is 1.919723
# After step 23300, training accuracy 0.61 training loss is 2.155522
# After step 23400, training accuracy 0.66 training loss is 1.969660
# After step 23500, training accuracy 0.65 training loss is 2.209089
# After step 23600, training accuracy 0.66 training loss is 1.663203
# After step 23700, training accuracy 0.72 training loss is 1.589479
# After step 23800, training accuracy 0.63 training loss is 1.500160
# After step 23900, training accuracy 0.59 training loss is 2.184794
# After step 24000, training accuracy 0.63 training loss is 2.486907
# After step 24100, training accuracy 0.64 training loss is 2.471125
# After step 24200, training accuracy 0.71 training loss is 1.935014
# After step 24300, training accuracy 0.63 training loss is 1.777805
# After step 24400, training accuracy 0.64 training loss is 1.918194
# After step 24500, training accuracy 0.78 training loss is 1.463883
# After step 24600, training accuracy 0.7 training loss is 1.430693
# After step 24700, training accuracy 0.67 training loss is 1.813032
# After step 24800, training accuracy 0.62 training loss is 1.981501
# After step 24900, training accuracy 0.62 training loss is 1.959454
# After step 25000, training accuracy 0.72 training loss is 1.336421
# After step 25100, training accuracy 0.69 training loss is 2.063260
# After step 25200, training accuracy 0.63 training loss is 1.338724
# After step 25300, training accuracy 0.7 training loss is 1.418508
# After step 25400, training accuracy 0.68 training loss is 1.370598
# After step 25500, training accuracy 0.72 training loss is 1.357399
# After step 25600, training accuracy 0.69 training loss is 1.649856
# After step 25700, training accuracy 0.65 training loss is 1.779465
# After step 25800, training accuracy 0.54 training loss is 1.823158
# After step 25900, training accuracy 0.63 training loss is 1.544150
# After step 26000, training accuracy 0.64 training loss is 1.615449
# After step 26100, training accuracy 0.67 training loss is 2.025056
# After step 26200, training accuracy 0.7 training loss is 1.285776
# After step 26300, training accuracy 0.67 training loss is 1.700079
# After step 26400, training accuracy 0.67 training loss is 1.678087
# After step 26500, training accuracy 0.7 training loss is 1.434242
# After step 26600, training accuracy 0.7 training loss is 1.777616
# After step 26700, training accuracy 0.73 training loss is 1.504302
# After step 26800, training accuracy 0.66 training loss is 1.606474
# After step 26900, training accuracy 0.75 training loss is 1.597459
# After step 27000, training accuracy 0.7 training loss is 1.292852
# After step 27100, training accuracy 0.71 training loss is 1.343691
# After step 27200, training accuracy 0.7 training loss is 1.260018
# After step 27300, training accuracy 0.64 training loss is 1.912849
# After step 27400, training accuracy 0.68 training loss is 1.838005
# After step 27500, training accuracy 0.8 training loss is 1.160767
# After step 27600, training accuracy 0.66 training loss is 1.903787
# After step 27700, training accuracy 0.65 training loss is 1.731614
# After step 27800, training accuracy 0.65 training loss is 1.630073
# After step 27900, training accuracy 0.72 training loss is 1.607709
# After step 28000, training accuracy 0.72 training loss is 1.307437
# After step 28100, training accuracy 0.62 training loss is 2.085824
# After step 28200, training accuracy 0.68 training loss is 1.143930
# After step 28300, training accuracy 0.76 training loss is 1.188209
# After step 28400, training accuracy 0.73 training loss is 0.995840
# After step 28500, training accuracy 0.71 training loss is 1.861706
# After step 28600, training accuracy 0.68 training loss is 1.835247
# After step 28700, training accuracy 0.73 training loss is 0.996306
# After step 28800, training accuracy 0.7 training loss is 1.311675
# After step 28900, training accuracy 0.69 training loss is 1.490569
# After step 29000, training accuracy 0.64 training loss is 1.660916
# After step 29100, training accuracy 0.75 training loss is 1.146862
# After step 29200, training accuracy 0.73 training loss is 1.052012
# After step 29300, training accuracy 0.63 training loss is 1.614924
# After step 29400, training accuracy 0.72 training loss is 1.230900
# After step 29500, training accuracy 0.62 training loss is 1.552768
# After step 29600, training accuracy 0.7 training loss is 1.395806
# After step 29700, training accuracy 0.76 training loss is 1.286476
# After step 29800, training accuracy 0.65 training loss is 1.366936
# After step 29900, training accuracy 0.7 training loss is 1.755470
# 2019-07-24 15:49:11.611268: W tensorflow/core/framework/allocator.cc:107] Allocation of 752640000 exceeds 10% of system memory.
# 2019-07-24 15:49:11.772706: W tensorflow/core/framework/allocator.cc:107] Allocation of 752640000 exceeds 10% of system memory.
# 2019-07-24 15:49:15.583549: W tensorflow/core/framework/allocator.cc:107] Allocation of 752640000 exceeds 10% of system memory.
# 2019-07-24 15:49:15.744916: W tensorflow/core/framework/allocator.cc:107] Allocation of 752640000 exceeds 10% of system memory.
# test accuracy 0.8221 test loss is 0.704569
# validation accuracy is 0.91 val loss is 0.383206





