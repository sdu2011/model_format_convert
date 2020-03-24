import tensorflow as tf
 
def freeze_graph(input_checkpoint, output_graph):
    output_node_names = "lanenet/input_tensor,lanenet/final_binary_output,lanenet/final_pixel_embedding_output" #获取的节点
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
 
from tensorflow.summary import FileWriter
def get_node_from_ckpt():
    sess = tf.Session()
    tf.train.import_meta_graph("./tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt.meta")
    FileWriter("__tb", sess.graph)
# get_node_from_ckpt()

def get_node_from_ckpt2():
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt.meta')
        graph_def = tf.get_default_graph().as_graph_def()
        node_list=[n.name for n in graph_def.node]
        print(node_list[-1])

# get_node_from_ckpt2()

def get_name_from_pb(pb_file_path):
    gf = tf.GraphDef()   
    m_file = open(pb_file_path,'rb')
    gf.ParseFromString(m_file.read())

    for n in gf.node:
        print(n.name)

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

if __name__ == '__main__':
    modelpath="./tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt"
    # freeze_graph(modelpath,"frozen.pb")
    # get_name_from_pb('./lanenet.pb')
    # pb_to_tflite('./lanenet.pb','./lanenet.tflite')
    pb_to_quant_tflite('./lanenet.pb','./lanenet_quant.tflite')

    print("finish!")
 