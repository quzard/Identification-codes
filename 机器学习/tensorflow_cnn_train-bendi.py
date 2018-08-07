#coding:utf-8

from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

def convert2gray(img):
	if len(img.shape) > 2:
		gray = np.mean(img, -1)
		return gray
	else:
		return img

CHAR_SET_LEN = 63

def text2vec(text):
	text_len = len(text)

	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
	def char2pos(c):
		if c =='_':
			k = 62
			return k
		k = ord(c)-48
		if k > 9:
			k = ord(c) - 55
			if k > 35:
				k = ord(c) - 61
				if k > 61:
					raise ValueError('No Map')
		return k
	for i, c in enumerate(text):
		idx = i * CHAR_SET_LEN + char2pos(c)
		vector[idx] = 1
	return vector
# 向量转回文本
def vec2text(vec):
	char_pos = vec.nonzero()[0]
	text=[]
	for i, c in enumerate(char_pos):
		char_at_pos = i #c/63
		char_idx = c % CHAR_SET_LEN
		if char_idx < 10:
			char_code = char_idx + ord('0')
		elif char_idx < 36:
			char_code = char_idx - 10 + ord('A')
		elif char_idx < 62:
			char_code = char_idx-  36 + ord('a')
		elif char_idx == 62:
			char_code = ord('_')
		else:
			raise ValueError('error')
		text.append(chr(char_code))
	return "".join(text)

# 生成一个训练batch

def get_next_batch(batch_size=128):
	global now
	global TEXT
	global file_name
	global num
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

	# 有时生成图像大小不是(60, 160, 3)
	def wrap_gen_captcha_text_and_image():
		global now
		global TEXT
		global file_name
		global num
		global step
		global acc
		while True:
			text=TEXT[now]
			image=Image.open(str(file_name[now])+'.jpg')
			image = np.array(image) #将图片装换为数组
			if(now==0):
				print(step, acc)
			now+=1
			if(now==num):
				now==0
			if image.shape == (60, 160, 3):#此部分应该与开头部分图片宽高吻合
				return text, image

	for i in range(batch_size):
		if(now<num):
			text, image = wrap_gen_captcha_text_and_image()
			image = convert2gray(image)

			# 将图片数组一维化 同时将文本也对应在两个二维组的同一行
			batch_x[i,:] = image.flatten() / 255
			batch_y[i,:] = text2vec(text)
		else:
			break
	# 返回该训练批次
	return batch_x, batch_y

####################################################################
# 申请占位符 按照图片
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	# 将占位符 转换为 按照图片给的新样式
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])


	w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32])) # 从正太分布输出随机值
	b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)

	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)

	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	return out

# 训练
def train_crack_captcha_cnn():
	global num
	global TEXT
	global now
	global file_name
	global step
	global acc
	files = os.listdir('.')#获得当前 硬盘目录中的所有文件  
	for i in files:#逐个文件遍历  
		if( os.path.isfile(i)):# 判断当前是一个文件夹'''   
			if(os.path.splitext(i)[1]=='.jpg'and os.path.splitext(i)[0]!='verifycode'):# 当前不是文件夹 获得当前的文件的扩展名  
				num=num+1;
				TEXT.append(os.path.splitext(i)[0][:4])
				file_name.append(os.path.splitext(i)[0])

	output = crack_captcha_cnn()
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
	# optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
	max_idx_p = tf.argmax(predict, 2)
	max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="accuracy")
	if not os.path.exists('tmp/'):
		os.mkdir('tmp/');
	saver = tf.train.Saver()
	module_file =  tf.train.latest_checkpoint('tmp/')
	W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
	b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if module_file is not None:
			saver.restore(sess, module_file)
			
		step = 0
		acc=0
		while True :
			batch_x, batch_y = get_next_batch(64)
			if(now==num):
				now=0
			if(now<num):
				_, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
				print(step, loss_,acc)

				# 每100 step计算一次准确率
				if step % 100 == 0:
					batch_x_test, batch_y_test = get_next_batch(100)
					acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
					#print(step, acc)
					# 如果准确率大于80%,保存模型,完成训练
					if acc > 0.98:
						#output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['SAME/predict'])
						saver.save(sess, 'tmp/model.ckpt', global_step=step)
						tf.train.write_graph(sess.graph_def, '', 'tmp/graph.pb')
						break
				step += 1
global now
global TEXT
global file_name
global num
now=0
num=0
TEXT=[]
file_name=[]
train_crack_captcha_cnn()
