import tensorflow as tf
import onnxruntime as ort
import numpy as np
from onnx_tf.backend import prepare
import onnx
print('onnx.version:{}'.format(onnx.__version__))


# import tensorflow as tf
# import logging
# tf.get_logger().setLevel(logging.ERROR)

pfe = '/home/sc/disk/git/kitti_pretrained_point_pillars/pfe.onnx'
rpn = '/home/sc/disk/git/kitti_pretrained_point_pillars/rpn.onnx'

#测试输入
input_pillar_x = np.random.randn(1, 1, 12000, 100).astype(np.float32)
input_pillar_y = np.random.randn(1, 1, 12000, 100).astype(np.float32)
input_pillar_z = np.random.randn(1, 1, 12000, 100).astype(np.float32)
input_pillar_i = np.random.randn(1, 1, 12000, 100).astype(np.float32)
input_num_points_per_pillar = np.random.randn(1, 12000).astype(np.float32)
input_x_sub_shaped = np.random.randn(1, 1, 12000, 100).astype(np.float32)
input_y_sub_shaped = np.random.randn(1, 1, 12000, 100).astype(np.float32)
input_mask = np.random.randn(1, 1, 12000, 100).astype(np.float32)



###############打印运算图######################
def get_onnx_gragh(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print('*****************************')
    print(onnx.helper.printable_graph(model.graph))

# get_onnx_gragh(pfe)


################onnx inference##############
def do_onnx_inference(onnx_path, input):
    ort_session = ort.InferenceSession(onnx_path)
    outputs = ort_session.run(None, input)

    # print(outputs[0])
    # print(len(outputs),type(outputs[0]),outputs[0].shape)
    print(len(outputs))
    for output in outputs:
        print(output.shape)

    return outputs

pfe_input = {'pillar_x': input_pillar_x,
             'pillar_y': input_pillar_y,
             'pillar_z': input_pillar_z,
             'pillar_i': input_pillar_i,
             'num_points_per_pillar': input_num_points_per_pillar,
             'x_sub_shaped': input_x_sub_shaped,
             'y_sub_shaped': input_y_sub_shaped,
             'mask': input_mask
             }
pfe_output = do_onnx_inference(pfe,pfe_input)
# print(pfe_output[0][0][0][0])
pfe_output = np.asarray(pfe_output)

# out = pfe_output[0][:,:,496*432,:]
# print(out.shape)

rpn_input = {'input.1': np.random.randn(1, 64, 496, 432).astype(np.float32)}
# rpn_output = do_onnx_inference(rpn,rpn_input)


#################check input shape##################
# The model is represented as a protobuf structure and it can be accessed
# using the standard python-for-protobuf methods

# iterate through inputs of the graph
def check_input_shape(model):
    for input in model.graph.input:
        print(input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    print(d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                    # unknown dimension with symbolic name
                    print(d.dim_param, end=", ")
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")
        print()

# check_input_shape(onnx.load(pfe))
# check_input_shape(onnx.load(rpn))


#####################################################
def onnx_to_pb(onnx_path, pb_path, input):
    onnx_model = onnx.load(onnx_path)  # load onnx model
    tf_rep = prepare(onnx_model, strict=False)
    out_tf = tf_rep.run(input)
    print(out_tf.shape)
    tf_rep.export_graph

# pfe_onnx_model = onnx.load(pfe)  # load onnx model
# rpn_onnx_model = onnx.load(rpn)

# # 模拟激光雷达输入
# # input = [n,4] #4代表x,y,z,r
# input = np.random.normal(size=(10000,4))
# pfe_tf_rep =  prepare(pfe_onnx_model,strict=False)
# out_tf = pfe_tf_rep.run(input)

# # pfe_tf_rep.export_graph('pfe_graph.pb')


######################
def pb_to_quant_tflite(pb_path, tflite_path):
    input_arrays = ['lanenet/input_tensor']
    output_arrays = ['lanenet/final_binary_output',
                     'lanenet/final_pixel_embedding_output']
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays, output_arrays)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    f = open(tflite_path, 'wb')
    f.write(tflite_model)


def pb_inference(pb_path, input):
    tf.reset_default_graph()

    graph = tf.Graph()
    

    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)

    with graph.as_default():  # 在一个进程内用到多个计算图时必须这么写　在本脚本中并非必要
        pillar_x = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='pillar_x')
        pillar_y = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='pillar_y')
        pillar_z = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='pillar_z')
        pillar_i = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='pillar_i')
        num_points_per_pillar = tf.placeholder(np.float, shape=[1, 12000],name='num_points_per_pillar')
        x_sub_shaped = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='x_sub_shaped')
        y_sub_shaped = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='y_sub_shaped')
        mask = tf.placeholder(np.float, shape=[1, 1, 12000, 100],name='mask')

        print(num_points_per_pillar)

        #将参数恢复到运算图中
        # tf.import_graph_def(graph_def, {'pillar_x': pillar_x,
        #                                 'pillar_y': pillar_y,
        #                                 'pillar_z': pillar_z,
        #                                 'pillar_i': pillar_i,
        #                                 'num_points_per_pillar':num_points_per_pillar,
        #                                 'x_sub_shaped':x_sub_shaped,
        #                                 'y_sub_shaped':y_sub_shaped,
        #                                 'mask':mask
        # })
        tf.import_graph_def(graph_def)
        print(num_points_per_pillar)
                    
    
    print('Model loading complete!')

    # Get layer names
    # layers = [op.name for op in graph.get_operations()]
    # for layer in layers:
    #     print(layer)
    
    """
    # Check out the weights of the nodes
    weight_nodes = [n for n in graph_def.node if n.op == 'Const']
    for n in weight_nodes:
        print("Name of the node - %s" % n.name)
        # print("Value - " )
        # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
    """
    
    graph.finalize() #使得运算图变为只读,不能再加op了

    sess = tf.Session(graph=graph)
    output_tensor = graph.get_tensor_by_name("import/174:0")
    # out = sess.run(output_tensor,feed_dict={
    #          'import/pillar_x:0': np.random.randn(1, 1, 12000, 100).astype(np.float),
    #          'import/pillar_y:0': np.random.randn(1, 1, 12000, 100).astype(np.float),
    #          'import/pillar_z:0': np.random.randn(1, 1, 12000, 100).astype(np.float),
    #          'import/pillar_i:0': np.random.randn(1, 1, 12000, 100).astype(np.float),
    #          'import/num_points_per_pillar:0': np.random.randn(1, 12000).astype(np.float),
    #          'import/x_sub_shaped:0': np.random.randn(1, 1, 12000, 100).astype(np.float),
    #          'import/y_sub_shaped:0': np.random.randn(1, 1, 12000, 100).astype(np.float),
    #          'import/mask:0': np.random.randn(1, 1, 12000, 100).astype(np.float)
    #          })
    out = sess.run(output_tensor,feed_dict=input)
    # print(type(out))
    # print(out)

    return out


#注意, Tensor names must be of the form "<op_name>:<output_index>"
tf_pfe_input = {'import/pillar_x:0': input_pillar_x,
             'import/pillar_y:0': input_pillar_y,
             'import/pillar_z:0': input_pillar_z,
             'import/pillar_i:0': input_pillar_i,
             'import/num_points_per_pillar:0': input_num_points_per_pillar,
             'import/x_sub_shaped:0': input_x_sub_shaped,
             'import/y_sub_shaped:0': input_y_sub_shaped,
             'import/mask:0': input_mask
             }

print('*************************************')
print(np.array_equal(tf_pfe_input['import/pillar_x:0'],pfe_input['pillar_x']))

out_tf = pb_inference('./pfe_graph.pb',tf_pfe_input)
print(out_tf.shape,pfe_output.shape)

out_onnx = np.squeeze(pfe_output,axis=0)
print(out_onnx.shape)

print(out_tf[0][3][19][0])
print(out_onnx[0][3][19][0])

# diff = out_onnx - out_tf
#判断两个ndarray是否近似
is_same = np.isclose(out_onnx,out_tf,atol=1e-02)
# print(np.all(is_same))
# print(is_same)

# result = np.where(is_same == False)  #等价于np.nonzero(is_same == False)
result = np.nonzero(is_same == False)
print(result) #result是一个tuple 依次为每个维度满足条件的下标
listOfCoordinates= list(zip(result[0], result[1],result[2], result[3]))
for cord in listOfCoordinates:
    # print(cord)
    print(out_onnx[cord[0]][cord[1]][cord[2]][cord[3]])
    print(out_tf[cord[0]][cord[1]][cord[2]][cord[3]])

def get_name_from_pb(pb_file_path):
    gragh_def = tf.GraphDef()
    m_file = open(pb_file_path, 'rb')
    gragh_def.ParseFromString(m_file.read())

    #将pb文件内包含的信息导入到当前计算图.
    tf.import_graph_def(gragh_def)
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name,'\n')
    # for n in gragh_def.node:
    #     print(n.name)


# get_name_from_pb('./pfe_graph.pb')


# import tensorflow as tf

# x = tf.placeholder(tf.string)
# y = tf.placeholder(tf.int32)
# z = tf.placeholder(tf.float32)

# with tf.Session() as sess:
#     output = sess.run(x, feed_dict = {x :'Hello World', y:123, z:45.67})
#     print(output)
#     output = sess.run(y, feed_dict = {x :'Hello World', y:123, z:45.67})
#     print(output)
#     output = sess.run(z, feed_dict = {x :'Hello World', y:123, z:45.67})
# print(output)