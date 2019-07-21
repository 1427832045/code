import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


with tf.Session() as sess:
	saver = tf.train.import_meta_graph('Model_save/mnist-alexnet-model3.ckpt.meta')# 加载图结构
	saver.restore(sess,tf.train.latest_checkpoint('Model_save/'))#加载变量

	y=tf.get_collection("pred")[0]
	graph = tf.get_default_graph()

	input_x = graph.get_operation_by_name('x-input').outputs[0]
	keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]


	tf.global_variables_initializer().run()
	
	print("mnist.test.images数据的预测值是:",sess.run(y, feed_dict={input_x: mnist.test.images,keep_prob:0.9}))
	y_ = tf.placeholder(tf.float32, [None, 10])
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print('test accuracy %g' % accuracy.eval(feed_dict={input_x: mnist.test.images, y_: mnist.test.labels,keep_prob:1}))
