from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # 从60,000 x 28 x 28变为60,000 x 784
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0  # 从10,000 x 28 x 28到10,000 x 784。
y_train = to_categorical(y_train)  # 60000*10 10个类别
y_test = to_categorical(y_test)  # 10000*10  10个类别

# 连续模型
model = Sequential()
# 添加10个单元格的隐藏层，来自784个输入单元格。
# 尽可能使用Relu。在每个隐藏层上
model.add(Dense(10, input_dim=784, activation='relu'))
# 将输出层添加到网络中,10个类别  softmax-激励函数
# 输出层上的Sigmoid有两个类别  Sigmoid-激励函数
model.add(Dense(10, activation='softmax'))#在输出图层上使用Softmax，可以预测两个以上的类别
# categorical_crossentropy多类别情况  binary_crossentropy 两个类别情况
# 使用adam或rmsprop作为优化器
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 10％的训练数据作为验证数据，因此validation_split设置为0.1
# Epoch是我们将要做的训练循环次数。
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model.evaluate(x_test, y_test)
print(test_acc)

