# TensorFlow关闭调试日志
```
//早期tensorflow版本可用
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed

// tensorflow 1.15版本测试可用
import logging
tf.get_logger().setLevel(logging.ERROR)
```

# TensorFlow Lite guide
tflite主要由两个部分构成
- tflite interpreter
- tflite converter

## tflite interpreter
在移动设备/嵌入式设备上运行优化后的模型

## tflite converter
将tensorflow模型转换成特定的格式供interpreter使用
将fp32转换到int8也分2种情况
- post-training
- during training
  
### post-training
```
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```
如果想得到更低的延迟,更少的内存占用.需要提供典型数据集.通过这个典型数据集来统计input和activation的范围.
```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

### during training
在训练代码里需要加上"fake-quantization"　node．
得到模型后,convert代码如下:
```
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
tflite_model = converter.convert()
```


### 已有模型的ckpt文件,想知道node名字
tensorflow中的node即operation．

<https://stackoverflow.com/questions/55757380/get-input-and-output-node-name-from-ckpt-and-meta-files-tensorflow>

### what is tensorflow checkpoint meta file
<https://stackoverflow.com/questions/36195454/what-is-the-tensorflow-checkpoint-meta-file>

### 已有.pb模型文件,获取node名字
```
import tensorflow as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = 'frozen_model.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)
```

### 转换得到量化模型
```
(env3.6) sc@sc:~/disk/keepgoing/model_format_convert$ ls tusimple_lanenet_vgg/ -lrth
总用量 531M
-rw-rw-r-- 1 sc sc 258M 5月  28  2019 tusimple_lanenet_vgg.ckpt.meta
-rw-rw-r-- 1 sc sc  11K 5月  28  2019 tusimple_lanenet_vgg.ckpt.index
-rw-rw-r-- 1 sc sc 273M 5月  28  2019 tusimple_lanenet_vgg.ckpt.data-00000-of-00001
-rw-rw-r-- 1 sc sc   57 5月  28  2019 checkpoint
```
1. checkpoint file --> .pb file
2. .pb --> .tflite

#### checkpoint file --> .pb file
``` python
#lanenet 
def convert_ckpt_into_pb_file(ckpt_file_path, pb_file_path):
    """

    :param ckpt_file_path:
    :param pb_file_path:
    :return:
    """
    # construct compute graph
    with tf.variable_scope('lanenet'):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    with tf.variable_scope('lanenet/'):
        binary_seg_ret = tf.cast(binary_seg_ret, dtype=tf.float32)
        binary_seg_ret = tf.squeeze(binary_seg_ret, axis=0, name='final_binary_output')
        instance_seg_ret = tf.squeeze(instance_seg_ret, axis=0, name='final_pixel_embedding_output')

    # create a session
    saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
    sess_config.gpu_options.allow_growth = False
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess, ckpt_file_path)

        converted_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=sess.graph.as_graph_def(),
            output_node_names=[
                'lanenet/input_tensor',
                'lanenet/final_binary_output',
                'lanenet/final_pixel_embedding_output'
            ]
        )

        with tf.gfile.GFile(pb_file_path, "wb") as f:
            f.write(converted_graph_def.SerializeToString())
```
分几步:
- 定义计算图,命名计算图的输入输出operation．
- 将ckpt文件包含的变量值恢复到session
- 用tf.graph_util.convert_variables_to_constants将变量值固定到计算图中

#### .pb --> .tflite
```　python
def pb_to_tflite(pb_path,tflite_path):
    input_arrays = ['lanenet/input_tensor']
    output_arrays = ['lanenet/final_binary_output','lanenet/final_pixel_embedding_output']
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays, output_arrays)
    tflite_model = converter.convert()
    f = open(tflite_path,'wb')
    f.write(tflite_model)

def pb_to_quant_tflite(pb_path,tflite_path):
    input_arrays = ['lanenet/input_tensor']
    output_arrays = ['lanenet/final_binary_output','lanenet/final_pixel_embedding_output']
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays, output_arrays)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    f = open(tflite_path,'wb')
    f.write(tflite_model)
```