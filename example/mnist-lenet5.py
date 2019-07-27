import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/share/home/flt/flt_data/code/MNIST_data/", one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 1000

#定义网络模型，模型结构：卷积1+池化1+卷积2+池化2+卷积3+池化3+全连接1+全连接2+全连接3
def hidden_layer(input_tensor,regularizer,avg_class,resuse):
    #创建第一个卷积层，得到特征图大小为32@28x28
    #通过tf.variable_scope()指定作用域进行区分，如with tf.variable_scope("conv1")这行代码指定了第一个卷积层作用域为conv1，在这个作用域下有两个变量weights和biases。

    with tf.variable_scope("C1-conv",reuse=resuse):
        # reuse为True的时候表示用tf.get_variable 得到的变量可以在别的地方重复使用
        # tf.get_variable获取一个已经存在的变量或者创建一个新的变量，可以参数共享
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    #创建第一个池化层，池化后的结果为32@14x14
    with tf.name_scope("S2-max_pool",):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 创建第二个卷积层，得到特征图大小为64@14x14。注意，第一个池化层之后得到了32个
    # 特征图，所以这里设输入的深度为32，我们在这一层选择的卷积核数量为64，所以输出
    # 的深度是64，也就是说有64个特征图
    with tf.variable_scope("C3-conv",reuse=resuse):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #创建第二个池化层，池化后结果为64@7x7
    with tf.name_scope("S4-max_pool",):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        #get_shape()函数可以得到这一层维度信息，由于每一层网络的输入输出都是一个batch的矩阵，
        #所以通过get_shape()函数得到的维度信息会包含这个batch中数据的个数信息
        #shape[1]是长度方向，shape[2]是宽度方向，shape[3]是深度方向
        #shape[0]是一个batch中数据的个数，reshape()函数原型reshape(tensor,shape,name)
        shape = pool2.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]    #nodes=3136
        reshaped = tf.reshape(pool2, [shape[0], nodes])

    #创建第一个全连层
    with tf.variable_scope("layer5-full1",reuse=resuse):
        Full_connection1_weights = tf.get_variable("weight", [nodes, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
        tf.add_to_collection("losses", regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable("bias", [512],
                                                     initializer=tf.constant_initializer(0.1))

        ##使用avg_class.average函数来计算得出变量的滑动平均值。
        if avg_class ==None:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, Full_connection1_weights) + \
                                                                   Full_connection1_biases)
        else:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(Full_connection1_weights))
                                                   + avg_class.average(Full_connection1_biases))

    #创建第二个全连层
    with tf.variable_scope("layer6-full2",reuse=resuse):
        Full_connection2_weights = tf.get_variable("weight", [512, 10],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
        tf.add_to_collection("losses", regularizer(Full_connection2_weights))
        Full_connection2_biases = tf.get_variable("bias", [10],
                                                   initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            result = tf.matmul(Full_1, Full_connection2_weights) + Full_connection2_biases
        else:
            result = tf.matmul(Full_1, avg_class.average(Full_connection2_weights)) + \
                                                  avg_class.average(Full_connection2_biases)
    return result


#训练可以吧下面这部分写成一个函数
x = tf.placeholder(tf.float32, [batch_size ,784],name="x-input")
images = tf.reshape(x, [batch_size,28,28,1])
y_ = tf.placeholder(tf.float32, [batch_size, 10], name="y-input")

regularizer = tf.contrib.layers.l2_regularizer(0.0001)

y = hidden_layer(images,regularizer,avg_class=None,resuse=False)
#一般会将代表训练轮数的变量指定为不可训练的参数
training_step = tf.Variable(0, trainable=False)
#给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
#在所有代表神经网络参数的变量上使用滑动平均
variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

average_y = hidden_layer(images,regularizer,variable_averages,resuse=True)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
#总损失等于交叉熵损失与正则化损失的和
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
#设置指数衰减的学习率，training_step为训练次数，learning_rate为初始学习率，mnist.train.num_examples /batch_size衰减次数，learning_rate_decay为衰减率
learning_rate = tf.train.exponential_decay(learning_rate,
                                 training_step, mnist.train.num_examples /batch_size ,
                                 learning_rate_decay, staircase=True)


##输出不使用滑动平均的精度和loss
crorent_predicition_noaverage = tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
accuracy_noaverage = tf.reduce_mean(tf.cast(crorent_predicition_noaverage,tf.float32))


train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)

#指定执行顺序，在执行完 train_step, variables_averages_op 操作之后，才能执行train_op 操作
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

crorent_predicition = tf.equal(tf.arg_max(average_y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(crorent_predicition,tf.float32))

saver=tf.train.Saver(max_to_keep=1)
max_acc=0

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_steps):
        if i %100==0:
            x_val, y_val = mnist.validation.next_batch(batch_size)
            validate_feed = {x: x_val, y_: y_val}

            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%" % (i, validate_accuracy * 100))

            x_train, y_train = mnist.train.next_batch(batch_size)
            train_feed = {x: x_train, y_: y_train}

            train_accuracy = sess.run(accuracy_noaverage, feed_dict=train_feed)
            print("After %d trainging step(s) ,train accuracy"
                  "not using average model is %g%%" % (i, train_accuracy * 100))

            loss_value = sess.run(loss, feed_dict=train_feed)
            print("After %d trainging step(s) ,train loss_value"
                  "not using average model is %f" % (i, loss_value))


        x_train, y_train = mnist.train.next_batch(batch_size)

        sess.run(train_op, feed_dict={x: x_train, y_: y_train})

        if validate_accuracy>max_acc:
            max_acc=validate_accuracy
            saver.save(sess,"model_save/mnist-lenet5-model.ckpt",global_step=i+1)

            #训练过程如下
After 200 trainging step(s) ,validation accuracyusing average model is 92%
After 200 trainging step(s) ,train accuracynot using average model is 92%
After 200 trainging step(s) ,train loss_valuenot using average model is 1.525790
After 300 trainging step(s) ,validation accuracyusing average model is 96%
After 300 trainging step(s) ,train accuracynot using average model is 94%
After 300 trainging step(s) ,train loss_valuenot using average model is 1.474380
After 400 trainging step(s) ,validation accuracyusing average model is 90%
After 400 trainging step(s) ,train accuracynot using average model is 92%
After 400 trainging step(s) ,train loss_valuenot using average model is 1.467097
After 500 trainging step(s) ,validation accuracyusing average model is 95%
After 500 trainging step(s) ,train accuracynot using average model is 93%
After 500 trainging step(s) ,train loss_valuenot using average model is 1.474987
After 600 trainging step(s) ,validation accuracyusing average model is 96%
After 600 trainging step(s) ,train accuracynot using average model is 94%
After 600 trainging step(s) ,train loss_valuenot using average model is 1.476646
After 700 trainging step(s) ,validation accuracyusing average model is 98%
After 700 trainging step(s) ,train accuracynot using average model is 97%
After 700 trainging step(s) ,train loss_valuenot using average model is 1.374275
After 800 trainging step(s) ,validation accuracyusing average model is 96%
After 800 trainging step(s) ,train accuracynot using average model is 97%
After 800 trainging step(s) ,train loss_valuenot using average model is 1.369796
After 900 trainging step(s) ,validation accuracyusing average model is 97%
After 900 trainging step(s) ,train accuracynot using average model is 96%
After 900 trainging step(s) ,train loss_valuenot using average model is 1.395066


        


