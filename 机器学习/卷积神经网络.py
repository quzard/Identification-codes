from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
from tensorflow.python.keras.api import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# CNN需要读取的数据必须是这样的
# total_data x width x height x channels


x_train = x_train[:,:,:,np.newaxis] / 255.0 # 从60,000 x 28 x 28变为60,000 x 28 x 28 x 1
x_test = x_test[:,:,:,np.newaxis] / 255.0  # 从10,000 x 28 x 28到10,000 x 28 x 28 x 1
y_train = to_categorical(y_train)  # 60000*10 10个类别
y_test = to_categorical(y_test)  # 10000*10  10个类别

# 连续模型
model4 = Sequential()
# conv2d会将您的28x28x1图像更改为28x28x64。想象一下这是64个隐藏层单元格。
# 尽可能使用Relu。在每个隐藏层上
model4.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28, 1)))
# MaxPooling将减小宽度和高度，因此您无需计算所有单元格。它将大小减小到14x14x64。
model4.add(MaxPooling2D(pool_size=2))
# 最后压扁只是缩小MaxPooling的输出。进入隐藏的12544细胞层。 12544 = 14*14*64
model4.add(Flatten())
# 输出层上的Sigmoid有两个类别  Sigmoid-激励函数
model4.add(Dense(10, activation='softmax'))
# categorical_crossentropy多类别情况  binary_crossentropy 两个类别情况
# 使用adam或rmsprop作为优化器
# 需要accuracy或categorical_accuracy作为检查网络性能的指标。
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
# 10％的训练数据作为验证数据，因此validation_split设置为0.1
# Epoch是我们将要做的训练循环次数。
model4.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model4.evaluate(x_test, y_test)
print(test_acc)





# print(model4.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 12544)             0
# _________________________________________________________________
# dense (Dense)                (None, 10)                125450
# =================================================================
# Total params: 125,770
# Trainable params: 125,770
# Non-trainable params: 0
# _________________________________________________________________
# None

