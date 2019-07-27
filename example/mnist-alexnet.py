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
trainaccuracy=tf.summary.scalar('trainaccuracy', accuracy)
testaccuracy=tf.summary.scalar('testaccuracy', accuracy)
valaccuracy=tf.summary.scalar('valaccuracy', accuracy)

# tf.summary.scalar("validation_accuracy", accuracy)

# 写到指定的磁盘路径中

from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'tensorboard/train/' + TIMESTAMP
writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)


test_log_dir = 'tensorboard/test/' + TIMESTAMP
writer_test = tf.summary.FileWriter(test_log_dir)


val_log_dir = 'tensorboard/val/' + TIMESTAMP
writer_val = tf.summary.FileWriter(val_log_dir)

#merged = tf.summary.merge_all()
merge_summary1 = tf.summary.merge([trainaccuracy])
merge_summary2 = tf.summary.merge([testaccuracy])
merge_summary3 = tf.summary.merge([valaccuracy])

# 初始化全局变量
tf.global_variables_initializer().run()


##训练过程
saver=tf.train.Saver(max_to_keep=1)
max_acc=0

for i in range(30000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob:0.9})
        #print('step %d, training accuracy %g' % (i, train_accuracy))
        loss_value = sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})
        print ('After step %d, training accuracy %g' % (i, train_accuracy),'training loss is %f'%(loss_value))
        summary,_= sess.run([merge_summary1,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})
        writer_train.add_summary(summary, i)

        if train_accuracy>max_acc:
            max_acc=train_accuracy
            saver.save(sess,"Model_save/mnist-alexnet-best-model.ckpt",global_step=i+1)


        #测试过程
        batch1 = mnist.test.next_batch(100)
        test_accuracy=accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob:1})
        test_loss_value = sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1],keep_prob:1})
        print ('test accuracy %g' % (test_accuracy),',test loss is %f'%(test_loss_value))
        summary2,_= sess.run([merge_summary2,train_step],feed_dict={x: batch1[0], y_: batch1[1],keep_prob:1})
        writer_test.add_summary(summary2,i)


        #验证过程
        x_val, y_val = mnist.validation.next_batch(100)
        validate_feed = {x: x_val, y_: y_val,keep_prob: 1.0}
        validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
        val_loss_value = sess.run(cross_entropy,feed_dict=validate_feed)
        print ('validation accuracy is %g' % (validate_accuracy),',val loss is %f'%(val_loss_value))

        summary3,_= sess.run([merge_summary3,train_step],feed_dict={x: batch[0], y_: batch[1],keep_prob:1})
        writer_val.add_summary(summary3,i)


        # summary, validate_accuracy = sess.run([merged, validate_accuracy],feed_dict=validate_feed)
        # writer_val.add_summary(summary,i)
        #cross_entropy=sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})
        #print ('step %d, loss %f'%(i, cross_entropy))
    #loss=sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #print (loss)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#saver.save(sess, "Model_save/mnist-alexnet-model3.ckpt")

##一部分训练过程

# After step 2500, training accuracy 0.11 training loss is 20.631170
# test accuracy 0.11 ,test loss is 20.493008
# validation accuracy is 0.08 ,val loss is 21.183784
# After step 2600, training accuracy 0.09 training loss is 20.629143
# test accuracy 0.09 ,test loss is 20.953524
# validation accuracy is 0.12 ,val loss is 20.262749
# After step 2700, training accuracy 0.11 training loss is 19.818325
# test accuracy 0.12 ,test loss is 20.262749
# validation accuracy is 0.08 ,val loss is 21.183784
# After step 2800, training accuracy 0.09 training loss is 20.684128
# test accuracy 0.1 ,test loss is 20.723267
# validation accuracy is 0.08 ,val loss is 21.151003
# After step 2900, training accuracy 0.04 training loss is 22.140034
# test accuracy 0.03 ,test loss is 22.303749
# validation accuracy is 0.1 ,val loss is 20.382765
# After step 3000, training accuracy 0.15 training loss is 19.770458
# test accuracy 0.14 ,test loss is 19.711866
# validation accuracy is 0.13 ,val loss is 20.032490
# After step 3100, training accuracy 0.2 training loss is 17.729908
# test accuracy 0.13 ,test loss is 19.116764
# validation accuracy is 0.22 ,val loss is 17.959959
# After step 3200, training accuracy 0.23 training loss is 18.309219
# test accuracy 0.2 ,test loss is 17.868164
# validation accuracy is 0.17 ,val loss is 18.612488
# After step 3300, training accuracy 0.27 training loss is 15.946198
# test accuracy 0.12 ,test loss is 19.816610
# validation accuracy is 0.19 ,val loss is 18.468884
# After step 3400, training accuracy 0.12 training loss is 20.492884
# test accuracy 0.13 ,test loss is 20.032486
# validation accuracy is 0.12 ,val loss is 20.262629
# After step 3500, training accuracy 0.06 training loss is 20.953222
# test accuracy 0.07 ,test loss is 21.414028
# validation accuracy is 0.11 ,val loss is 20.492826
# After step 3600, training accuracy 0.12 training loss is 19.781393
# test accuracy 0.14 ,test loss is 19.802231
# validation accuracy is 0.13 ,val loss is 20.032490
# After step 3700, training accuracy 0.08 training loss is 20.721716
# test accuracy 0.12 ,test loss is 20.058067
# validation accuracy is 0.14 ,val loss is 19.751139
# After step 3800, training accuracy 0.18 training loss is 18.792675
# test accuracy 0.17 ,test loss is 19.013931
# validation accuracy is 0.16 ,val loss is 18.953564
# After step 3900, training accuracy 0.23 training loss is 17.941029
# test accuracy 0.23 ,test loss is 17.549736
# validation accuracy is 0.19 ,val loss is 18.541000
# After step 4000, training accuracy 0.15 training loss is 19.198257
# test accuracy 0.21 ,test loss is 17.848743
# validation accuracy is 0.25 ,val loss is 17.081099
# After step 4100, training accuracy 0.2 training loss is 18.068865
# test accuracy 0.24 ,test loss is 17.499668
# validation accuracy is 0.2 ,val loss is 18.329306
# After step 4200, training accuracy 0.22 training loss is 17.528803
# test accuracy 0.23 ,test loss is 17.729904
# validation accuracy is 0.17 ,val loss is 18.690769
# After step 4300, training accuracy 0.26 training loss is 17.169495
# test accuracy 0.28 ,test loss is 16.578611
# validation accuracy is 0.16 ,val loss is 19.120321
# After step 4400, training accuracy 0.22 training loss is 17.954948
# test accuracy 0.24 ,test loss is 17.424583
# validation accuracy is 0.19 ,val loss is 18.650938
# After step 4500, training accuracy 0.18 training loss is 18.649414
# test accuracy 0.18 ,test loss is 18.760633
# validation accuracy is 0.26 ,val loss is 17.039139
# After step 4600, training accuracy 0.21 training loss is 18.757771
# test accuracy 0.22 ,test loss is 17.766859
# validation accuracy is 0.17 ,val loss is 18.920042
# After step 4700, training accuracy 0.19 training loss is 19.509150
# test accuracy 0.19 ,test loss is 18.650938
# validation accuracy is 0.18 ,val loss is 18.669226
# After step 4800, training accuracy 0.18 training loss is 18.356012
# test accuracy 0.19 ,test loss is 18.653391
# validation accuracy is 0.25 ,val loss is 16.699663
# After step 4900, training accuracy 0.13 training loss is 20.571770
# test accuracy 0.11 ,test loss is 20.300690
# validation accuracy is 0.29 ,val loss is 16.348669
# After step 5000, training accuracy 0.17 training loss is 18.880760
# test accuracy 0.19 ,test loss is 18.650938
# validation accuracy is 0.17 ,val loss is 18.894974
# After step 5100, training accuracy 0.21 training loss is 17.375145
# test accuracy 0.24 ,test loss is 17.111492
# validation accuracy is 0.21 ,val loss is 17.924740
# After step 5200, training accuracy 0.17 training loss is 18.547428
# test accuracy 0.25 ,test loss is 15.152561
# validation accuracy is 0.18 ,val loss is 18.659002
# After step 5300, training accuracy 0.21 training loss is 18.828051
# test accuracy 0.18 ,test loss is 18.739983
# validation accuracy is 0.17 ,val loss is 19.000265
# After step 5400, training accuracy 0.12 training loss is 19.860836
# test accuracy 0.13 ,test loss is 19.947582
# validation accuracy is 0.18 ,val loss is 18.871197
# After step 5500, training accuracy 0.13 training loss is 20.252193
# test accuracy 0.15 ,test loss is 19.571972
# validation accuracy is 0.2 ,val loss is 18.273636
# After step 5600, training accuracy 0.11 training loss is 20.953524
# test accuracy 0.1 ,test loss is 20.427017
# validation accuracy is 0.22 ,val loss is 17.647087
# After step 5700, training accuracy 0.23 training loss is 17.489569
# test accuracy 0.24 ,test loss is 17.306005
# validation accuracy is 0.22 ,val loss is 17.962868
# After step 5800, training accuracy 0.16 training loss is 18.413980
# test accuracy 0.2 ,test loss is 18.347948
# validation accuracy is 0.21 ,val loss is 18.015318
# After step 5900, training accuracy 0.17 training loss is 18.648367
# test accuracy 0.19 ,test loss is 18.499411
# validation accuracy is 0.28 ,val loss is 16.583485
# After step 6000, training accuracy 0.2 training loss is 17.319765
# test accuracy 0.26 ,test loss is 16.909134
# validation accuracy is 0.23 ,val loss is 17.729904
# After step 6100, training accuracy 0.22 training loss is 18.417673
# test accuracy 0.23 ,test loss is 17.722895
# validation accuracy is 0.2 ,val loss is 18.407461
# After step 6200, training accuracy 0.25 training loss is 17.339298
# test accuracy 0.2 ,test loss is 18.384127
# validation accuracy is 0.31 ,val loss is 15.449014
# After step 6300, training accuracy 0.18 training loss is 18.740555
# test accuracy 0.23 ,test loss is 17.507488
# validation accuracy is 0.21 ,val loss is 17.972466
# After step 6400, training accuracy 0.18 training loss is 18.559790
# test accuracy 0.24 ,test loss is 16.937969
# validation accuracy is 0.28 ,val loss is 16.141809
# After step 6500, training accuracy 0.34 training loss is 15.146499
# test accuracy 0.33 ,test loss is 15.081622
# validation accuracy is 0.25 ,val loss is 16.952972
# After step 6600, training accuracy 0.21 training loss is 17.727646
# test accuracy 0.28 ,test loss is 16.359690
# validation accuracy is 0.35 ,val loss is 14.931065
# After step 6700, training accuracy 0.31 training loss is 16.087154
# test accuracy 0.27 ,test loss is 15.495953
# validation accuracy is 0.29 ,val loss is 15.996944
# After step 6800, training accuracy 0.24 training loss is 15.973668
# test accuracy 0.32 ,test loss is 15.462735
# validation accuracy is 0.16 ,val loss is 18.680513
# After step 6900, training accuracy 0.29 training loss is 16.209351
# test accuracy 0.3 ,test loss is 15.260778
# validation accuracy is 0.23 ,val loss is 17.063108
# After step 7000, training accuracy 0.25 training loss is 16.932703
# test accuracy 0.2 ,test loss is 17.626585
# validation accuracy is 0.34 ,val loss is 13.911949
# After step 7100, training accuracy 0.23 training loss is 16.236792
# test accuracy 0.26 ,test loss is 16.195637
# validation accuracy is 0.16 ,val loss is 19.125141
# After step 7200, training accuracy 0.15 training loss is 19.468977
# test accuracy 0.15 ,test loss is 19.310743
# validation accuracy is 0.22 ,val loss is 17.687349
# After step 7300, training accuracy 0.19 training loss is 18.227768
# test accuracy 0.21 ,test loss is 17.985085
# validation accuracy is 0.22 ,val loss is 17.960163
# After step 7400, training accuracy 0.16 training loss is 19.454473
# test accuracy 0.15 ,test loss is 19.286400
# validation accuracy is 0.26 ,val loss is 16.503786
# After step 7500, training accuracy 0.26 training loss is 16.033987
# test accuracy 0.29 ,test loss is 15.096319
# validation accuracy is 0.25 ,val loss is 16.588812
# After step 7600, training accuracy 0.23 training loss is 17.684486
# test accuracy 0.25 ,test loss is 17.105782
# validation accuracy is 0.28 ,val loss is 16.222084






