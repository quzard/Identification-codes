#coding:utf-8

import http.cookiejar
from urllib import request
import urllib.request
import numpy as np
import tensorflow as tf
import collections
from PIL import Image
import io 
from captcha.image import ImageCaptcha  
import matplotlib.pyplot as plt
import random,time,os

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

# 把彩色图像转为灰度图像
def convert2gray(img):
	if len(img.shape) > 2:
		gray = np.mean(img, -1)
		return gray
	else:
		return img

CHAR_SET_LEN = 63
# 文本转向量
def text2vec(text):
	text_len = len(text)
	if text_len > MAX_CAPTCHA:
		raise ValueError('验证码最长4个字符')

	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN) #生成一个默认为0的向量
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
		elif char_idx <36:
			char_code = char_idx - 10 + ord('A')
		elif char_idx < 62:
			char_code = char_idx-  36 + ord('a')
		elif char_idx == 62:
			char_code = ord('_')
		else:
			raise ValueError('error')
		text.append(chr(char_code))
	return "".join(text)


####################################################################

# 申请三个占位符
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])


	#3 个 转换层
	w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
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

	# 最后连接层
	w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	# 输出层
	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	return out

def crack_captcha(captcha_image):
	
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('tmp/'))

		predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
		text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

		text = text_list[0].tolist()
		vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
		i = 0
		for n in text:
				vector[i*CHAR_SET_LEN + n] = 1
				i += 1
		return vec2text(vector)

def getUrlResponse(url,head) :
    global cookie
    url = str(url)
    req = request.Request(url)
    for eachhead in head.keys():
        req.add_header(eachhead,head[eachhead])
        
    resp = request.urlopen(req)  
    return resp
 

    

def getCheckCode(url,headerdic):
    global cookie
    global image
    checkCode=0
    cookie = http.cookiejar.LWPCookieJar()
    opener =urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie), urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    respHtml = getUrlResponse(url,headerdic).read()

    image = Image.open(io.BytesIO(respHtml))
    return checkCode

if __name__ == '__main__':
	global cookie
	global image
	headerdic={'Host': 'xk.urp.seu.edu.cn',
				'Connection': 'keep-alive',
				'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36',
				'Origin': 'http://xk.urp.seu.edu.cn'
				}
    
    #打开登陆页面
	img_url="http://xk.urp.seu.edu.cn/studentService/getCheckCode"
	header=headerdic
	output = crack_captcha_cnn()

	saver = tf.train.Saver()
	while(True):
		text=getCheckCode(img_url,header)
		image = image.resize((160, 60), Image.ANTIALIAS)
		
		im = np.array(image) #将图片装换为数组		

		mage = convert2gray(im) #生成一张新图
		mage = mage.flatten() / 255 # 将图片一维化
		predict_text = crack_captcha(mage) #导入模型识别
		print(  "预测: {}".format(predict_text))
		f = plt.figure()
		ax = f.add_subplot(111)
		ax.text(0.1, 0.9,predict_text, ha='center', va='center', transform=ax.transAxes)
		plt.imshow(image)
		plt.show()

