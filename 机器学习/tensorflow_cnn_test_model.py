#coding:utf-8
import os
import http.cookiejar
from urllib import request
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image
import io 
import matplotlib.pyplot as plt
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
CHAR_SET_LEN = 63
# 按照图片大小申请占位符
X = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.compat.v1.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
# 防止过拟合 训练时启用 测试时不启用
keep_prob = tf.compat.v1.placeholder(tf.float32)

# 把彩色图像转为灰度图像
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


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
    text = None
    image = None
    for i in range(batch_size):
        if now_train < num_train:
            text, image = wrap_gen_captcha_text_and_image(y_train, X_train, now_train)
            now_train += 1
            # if now_train == num_train:
            #     now_train = 0
        else:
            break
    # 返回该训练批次
    return image, text


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



if __name__ == '__main__':
    global cookie
    global image
    global now
    global num
    global TEXT
    global file_name
    global step
    num_train = 0
    now_train = 0
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
    X_train = file_name
    y_train = TEXT
    num_train = len(X_train)
    now_train = 0
    now_test = 0

    headerdic={'Host': 'xk.urp.seu.edu.cn',
                'Connection': 'keep-alive',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36',
                'Origin': 'http://xk.urp.seu.edu.cn'
                }
    
    # 打开登陆页面
    img_url="http://xk.urp.seu.edu.cn/studentService/getCheckCode"
    header=headerdic
    output = crack_captcha_cnn()
    filename = 'test.txt'


    saver = tf.train.Saver()
    while(now_train < num_train):
        mage, text = get_next_batch_train(1)
        mage = convert2gray(mage)  # 生成一张新图
        mage = mage.flatten() / 255  # 将图片一维化
        predict_text = crack_captcha(mage)  # 导入模型识别
        # print("预测: {},{}".format(predict_text, text))
        print("剩下:{}".format(num_train - now_train))
        if text != predict_text:
            with open('file.txt', 'a+') as fp:
                fp.write(X_train[now_train-1]+'\n')

        # text = getCheckCode(img_url,header)
        # image = image.resize((160, 60), Image.ANTIALIAS)
        #
        # im = np.array(image) # 将图片装换为数组
        #
        # mage = convert2gray(im)  # 生成一张新图
        # mage = mage.flatten() / 255  # 将图片一维化
        # predict_text = crack_captcha(mage)  # 导入模型识别
        # print("预测: {}".format(predict_text))
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9,predict_text, ha='center', va='center', transform=ax.transAxes)
        # plt.imshow(image)
        # plt.show()

