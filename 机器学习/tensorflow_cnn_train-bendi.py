# coding: utf-8
__author__ = "2019/7/5 22:31"
__time__ = "Quzard"

import random
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
CHAR_SET_LEN = 63


def convert2gray(img):  # 把彩色图像转为灰度图像
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def char2pos(c):
    if c == '_':
        k = 62
        return k
    k = ord(c) - 48
    if k > 9:
        k = ord(c) - 55
        if k > 35:
            k = ord(c) - 61
            if k > 61:
                raise ValueError('No Map')
    return k


def text2vec(text):  # 文本转向量
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 有时生成图像大小不是(60, 160, 3)
def wrap_gen_captcha_text_and_image(TEXT_, file_name_, n):
    while True:
        text = TEXT_[n]
        image = Image.open(str(file_name_[n]) + '.jpg')
        image = np.array(image)  # 将图片装换为数组
        if image.shape == (60, 160, 3):  # 此部分应该与开头部分图片宽高吻合
            return text, image


def get_next_batch_train(batch_size=128):  # 生成一个训练batch
    global now_train
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        if now_train < num_train:
            text, image = wrap_gen_captcha_text_and_image(y_train, X_train, now_train)
            now_train += 1
            if now_train == num_train:
                now_train = 0
            image = convert2gray(image)

            # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = text2vec(text)
        else:
            break
    # 返回该训练批次
    return batch_x, batch_y


def get_next_batch_test(batch_size=128):  # 生成一个训练batch
    global now_test
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        if now_test < num_test:
            text, image = wrap_gen_captcha_text_and_image(y_test, X_test, now_test)
            now_test += 1
            if now_test == num_test:
                now_test = 0
            image = convert2gray(image)

            # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = text2vec(text)
        else:
            break
    # 返回该训练批次
    return batch_x, batch_y


# 按照图片大小申请占位符
X = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='input')
Y = tf.compat.v1.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN], name='labels_placeholder')
# 防止过拟合 训练时启用 测试时不启用
keep_prob = tf.compat.v1.placeholder(tf.float32)


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):  # 定义CNN

    # 将占位符 转换为 按照图片给的新样式
    x_image = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 第一层
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = tf.Variable(w_alpha * tf.random.normal([3, 3, 1, 32]))  # 从正太分布输出随机值
    b_conv1 = tf.Variable(b_alpha * tf.random.normal([32]))
    # rulu激活函数
    h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1))
    # 池化
    h_pool1 = tf.nn.max_pool2d(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout防止过拟合
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # 第二层
    w_conv2 = tf.Variable(w_alpha * tf.random.normal([3, 3, 32, 64]))
    b_conv2 = tf.Variable(b_alpha * tf.random.normal([64]))
    h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_drop1, w_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2))
    h_pool2 = tf.nn.max_pool2d(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # 第三层
    w_conv3 = tf.Variable(w_alpha * tf.random.normal([3, 3, 64, 64]))
    b_conv3 = tf.Variable(b_alpha * tf.random.normal([64]))
    h_conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_drop2, w_conv3, strides=[1, 1, 1, 1], padding='SAME'), b_conv3))
    h_pool3 = tf.nn.max_pool2d(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    # 全连接层
    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])

    w_fc = tf.Variable(w_alpha * tf.random.normal([image_height * image_width * 64, 1024]))
    b_fc = tf.Variable(b_alpha * tf.random.normal([1024]))
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height * image_width * 64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop3_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    with tf.name_scope('output'):
        w_out = tf.Variable(w_alpha * tf.random.normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
        tf.compat.v1.summary.histogram('output/weight', w_out)
        b_out = tf.Variable(b_alpha * tf.random.normal([MAX_CAPTCHA * CHAR_SET_LEN]))
        tf.compat.v1.summary.histogram('output/biases', b_out)
        out = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return out


# 最小化loss

def optimize_graph(y, y_conv):
    '''
    优化计算图
    :param y:
    :param y_conv:
    :return:
    '''

    # 交叉熵计算loss

    # sigmod_cross适用于每个类别相互独立但不互斥，如图中可以有字母和数字

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))

    # 最小化loss优化

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return optimizer, loss


# 偏差计算
def accuracy_graph(y, y_conv, width=CHAR_SET_LEN, height=MAX_CAPTCHA):
    '''
    :param y:
    :param y_conv:
    :param width:
    :param height:
    :return:
    '''

    # 预测值

    predict = tf.reshape(y_conv, [-1, height, width])

    max_predict_idx = tf.argmax(predict, 2)

    # 标签

    label = tf.reshape(y, [-1, height, width])

    max_label_idx = tf.argmax(label, 2)

    correct_p = tf.equal(max_predict_idx, max_label_idx)

    accuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32), name="accuracy")

    return accuracy


def train_crack_captcha_cnn():  # 训练

    global num
    global TEXT
    global now
    global file_name
    global step
    global acc
    global X_train, X_test, y_train, y_test
    global num_train
    global num_test
    global now_train
    global now_test

    # cnn模型
    y_conv = crack_captcha_cnn()

    # 最优化
    optimizer, loss = optimize_graph(Y, y_conv)

    # 偏差
    accuracy = accuracy_graph(Y, y_conv)

    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')

    # 启动会话.开始训练
    saver = tf.train.Saver()
    module_file = tf.train.latest_checkpoint('tmp/')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if module_file is not None:
            saver.restore(sess, module_file)

        step = 0
        acc = 0
        while True:
            # 每批次64个样本
            batch_x, batch_y = get_next_batch_train(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch_test(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于80%,保存模型,完成训练
                if step % 1000 == 0:
                    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['accuracy'])
                    saver.save(sess, 'tmp/model.ckpt', global_step=step)
                    tf.io.write_graph(sess.graph_def, '', 'tmp/graph.pb')
                    with tf.gfile.FastGFile('tmp/CTNModel.pb', mode='wb') as f:
                        f.write(output_graph_def.SerializeToString())
                        tf.train.write_graph(output_graph_def, '.', 'tmp/graph_placeholder.pb', as_text=False)
                        tf.train.write_graph(output_graph_def, '.', 'tmp/graph_placeholder_txt.pb', as_text=True)
                    # if acc > 0.98:
                    #     break
                    # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
                    # for tensor_name in tensor_name_list:
                    #     print(tensor_name, '\n')
                if step % 10000 == 0 and step>0:
                    break
                # if acc > 0.98:
                #     X_train, X_test, y_train, y_test = train_test_split(file_name, TEXT, test_size=0.3, random_state=random.randint(1, 300))
                #     num_train = len(X_train)
                #     num_test = len(X_test)
                #     now_train = 0
                #     now_test = 0
            step += 1
        sess.close()

now = 0
num = 0
TEXT = []
file_name = []
step = 0
acc = 0

files = os.listdir('../验证码样本2/.')  # 获得当前 硬盘目录中的所有文件
for i in files:  # 逐个文件遍历
    if os.path.splitext(i)[1] == '.jpg' and os.path.splitext(i)[0] != 'verifycode':
        num = num + 1
        TEXT.append(os.path.splitext(i)[0][:4])

        file_name.append('../验证码样本2/' + os.path.splitext(i)[0])
# X_train, X_test, y_train, y_test = train_test_split(file_name, TEXT, test_size=0.3, random_state=random.randint(1, 300))

# X_train = file_name
# y_train = TEXT
# X_test = file_name
# y_test = TEXT
X_train, X_test, y_train, y_test = train_test_split(file_name, TEXT, test_size=0.3, random_state=random.randint(1, 300))


num_train = len(X_train)
num_test = len(X_test)
now_train = 0
now_test = 0

with tf.device('/cpu:0'):
    train_crack_captcha_cnn()

