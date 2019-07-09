# coding: utf-8
__author__ = "2019/7/5 22:31"
__time__ = "Quzard"
import tensorflow as tf
from PIL import Image
import numpy as np


class code_predict:
    def __init__(self):
        self.CHAR_SET_LEN = 63
        self.saver = tf.train.Saver()
        self.MAX_CAPTCHA = 4
        self.output = self.crack_captcha_cnn()

    # 把彩色图像转为灰度图像
    def convert2gray(self, img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            return gray
        else:
            return img

    # 向量转回文本
    def vec2text(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_at_pos = i  # c/63
            char_idx = c % self.CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    # 定义CNN
    def crack_captcha_cnn(self, w_alpha=0.01, b_alpha=0.1):
        # 申请三个占位符
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        keep_prob = tf.placeholder(tf.float32)  # dropout
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        # 3 个 转换层
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, keep_prob)

        # 最后连接层
        w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)

        # 输出层
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
        b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out

    def crack_captcha(self, captcha_image):

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('tmp/'))

            predict = tf.argmax(tf.reshape(self.output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

            text = text_list[0].tolist()
            vector = np.zeros(self.MAX_CAPTCHA * self.CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * self.CHAR_SET_LEN + n] = 1
                i += 1
            return self.vec2text(vector)

    def predict_text(self, image):
        self.image = image.resize((160, 60), Image.ANTIALIAS)
        im = np.array(self.image)  # 将图片装换为数组
        mage = self.convert2gray(im)  # 生成一张新图
        mage = mage.flatten() / 255  # 将图片一维化
        self.text = self.crack_captcha(mage)  # 导入模型识别
        return self.text
