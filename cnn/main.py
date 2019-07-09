# -*- coding: utf-8 -*- 
# @Time : 2019/5/31 下午4:28 
# @Author : Quzard 

# import tensorflow as tf
from tensorflow.python.keras import layers
import tensorflow.python.keras as keras


def main():
    model = keras.Sequential()
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # 


if __name__ == "__main__":
    main()
