# torch2onnx
``` python
def export(model, args, f, export_params=True, verbose=False, training=False,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None):
    r"""
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported;
    at the moment, it supports a limited set of dynamic models (e.g., RNNs.)

    See also: :ref:`onnx-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Tensor arguments will
            be hard-coded into the exported model; any Tensor arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Tensor, this is equivalent
            to having called it with a 1-ary tuple of that Tensor.
            (Note: passing keyword arguments to the model is not currently
            supported.  Give us a shout if you need it.)
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        training (bool, default False): export the model in training mode.  At
            the moment, ONNX is oriented towards exporting models for inference
            only, so you will generally not need to set this to True.
        input_names(list of strings, default empty list): names to assign to the
            input nodes of the graph, in order
        output_names(list of strings, default empty list): names to assign to the
            output nodes of the graph, in order
        aten (bool, default False): [DEPRECATED. use operator_export_type] export the
            model in aten mode. If using aten mode, all the ops original exported
            by the functions in symbolic.py are exported as ATen ops.
        export_raw_ir (bool, default False): [DEPRECATED. use operator_export_type]
            export the internal IR directly instead of converting it to ONNX ops.
        operator_export_type (enum, default OperatorExportTypes.ONNX):
            OperatorExportTypes.ONNX: all ops are exported as regular ONNX ops.
            OperatorExportTypes.ONNX_ATEN: all ops are exported as ATen ops.
            OperatorExportTypes.ONNX_ATEN_FALLBACK: if symbolic is missing,
                                                    fall back on ATen op.
            OperatorExportTypes.RAW: export raw ir.
    """
```

对于简单的模型,一般不会遇到什么问题.直接调用
```
torch.onnx.export
```
就完事.  注意对于dynamic shape的输入,转换时要通过aynamic_axes参数注明会动态变化的维度.比如:
```
input_names = ['voxels','num_points','coords']

# torch.onnx.export(net.voxel_feature_extractor,
#                 (voxels,num_points,coords),
#                 "vfe_dynamic_shape.onnx",
#                 verbose=False,
#                 input_names=input_names,
#                 dynamic_axes={'voxels':{0:'effective_pillar_num'},
#                               'num_points':{0:'effective_pillar_num'},
#                               'coords':{0:'effective_pillar_num'}
#                              }   
#                 )
```
模型的输入为３个tensor．3个tensor的第一个维度是不固定的.

## ATen
[aten](https://github.com/pytorch/pytorch/tree/master/aten)即“A Tensor Library for C++11”.
里面实现了很多的算子.有些pytorch里的函数或者类可能使用了ATen里面的算子.

pytorch模型所使用到的算子必须都在onnx里有相应的实现,才能够正确的完成torch到onnx的模型转换.

Aten op用来加速运算.比如`Ａ*B + C`,要两次内核调用,一次算乘法,一次算加法.用ATen里的 `at::addmm(C, B, A)`只需要一次内核调用.



参考:<https://leimao.github.io/blog/PyTorch-ATen-ONNX/>


在转一个并不复杂的模型的时候出现错误.模型并不存在什么复杂的算子.
```
RuntimeError: tuple appears in op that does not forward tuples (VisitNode at /pytorch/torch/csrc/jit/passes/lower_tuples.cpp:109)
frame #0: std::function<std::string ()>::operator()() const + 0x11 (0x7f27f31b3fe1 in /home/train/.local/lib/python3.5/site-packages/torch/lib/libc10.so)
frame #1: c10::Error::Error(c10::SourceLocation, std::string const&) + 0x2a (0x7f27f31b3dfa in /home/train/.local/lib/python3.5/site-packages/torch/lib/libc10.so)
frame #2: <unknown function> + 0x6da2e1 (0x7f27def7f2e1 in /home/train/.local/lib/python3.5/site-packages/torch/lib/libtorch.so.1)
frame #3: <unknown function> + 0x6da534 (0x7f27def7f534 in /home/train/.local/lib/python3.5/site-packages/torch/lib/libtorch.so.1)

```
搜索后发现和[这个错误](https://github.com/pytorch/pytorch/issues/13397)类似.
把`DataParallel`移除即可.
``` python
model = mobilenetv2()
model = torch.nn.DataParallel(model).cuda()
```
改为
``` python
model = mobilenetv2().cuda()
```
