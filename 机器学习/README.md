
使用freeze_graph.py 将tf.train.write_graph生成的pb模型文件和saver.save保存的权重数据合并成一个pb文件
代码案例：
 py freeze_graph.py --input_graph "graph.pb" --input_checkpoint  "model.ckpt-0" --output_graph  "222.
py" --output_node_names  accuracy

使用tensorflowjs将tf_frozen_model模型转化为TensorFlow.js可用的模型
参考网站：https://github.com/tensorflow/tfjs-converter
代码案例：
tensorflowjs_converter --input_format tf_frozen_model --output_node_names='accuracy' C:\Users\11
918\Desktop\222.pb C:\Users\11918\Desktop
