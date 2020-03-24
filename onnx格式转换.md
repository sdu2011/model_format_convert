# onnx

## 打印运算图
``` python
# Load the ONNX model
model = onnx.load(pfe)
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print('*****************************')
print(onnx.helper.printable_graph(model.graph))
```
输出
```
graph torch-jit-export (
  %pillar_x[FLOAT, 1x1x12000x100]
  %pillar_y[FLOAT, 1x1x12000x100]
  %pillar_z[FLOAT, 1x1x12000x100]
  %pillar_i[FLOAT, 1x1x12000x100]
  %num_points_per_pillar[FLOAT, 1x12000]
  %x_sub_shaped[FLOAT, 1x1x12000x100]
  %y_sub_shaped[FLOAT, 1x1x12000x100]
  %mask[FLOAT, 1x1x12000x100]
) optional inputs with matching initializers (
  %8[INT64, 1]
  %9[FLOAT, 64x9]
  %10[FLOAT, 64]
  %11[FLOAT, 64]
  %12[FLOAT, 64]
  %13[FLOAT, 64]
  %14[INT64, scalar]
  %15[FLOAT, 64x9x1x1]
  %16[FLOAT, 64]
  %17[FLOAT, 1x100x1x1]
  %18[FLOAT, 1]
  %19[FLOAT, 100x1x1x8]
  %20[FLOAT, 1]
  %21[FLOAT, 64x64x1x34]
  %22[FLOAT, 64]
  %23[FLOAT, 64x64x3x3]
  %24[FLOAT, 64]
  %25[FLOAT, 64]
  %26[FLOAT, 64]
  %27[FLOAT, 64]
  %28[INT64, scalar]
  %29[FLOAT, 64x64x3x3]
  %30[FLOAT, 64]
  %31[FLOAT, 64]
  %32[FLOAT, 64]
  %33[FLOAT, 64]
  %34[INT64, scalar]
  %35[FLOAT, 64x64x3x3]
  %36[FLOAT, 64]
  %37[FLOAT, 64]
  %38[FLOAT, 64]
  %39[FLOAT, 64]
  %40[INT64, scalar]
  %41[FLOAT, 64x64x3x3]
  %42[FLOAT, 64]
  %43[FLOAT, 64]
  %44[FLOAT, 64]
  %45[FLOAT, 64]
  %46[INT64, scalar]
  %47[FLOAT, 64x128x1x1]
  %48[FLOAT, 128]
  %49[FLOAT, 128]
  %50[FLOAT, 128]
  %51[FLOAT, 128]
  %52[INT64, scalar]
  %53[FLOAT, 128x64x3x3]
  %54[FLOAT, 128]
  %55[FLOAT, 128]
  %56[FLOAT, 128]
  %57[FLOAT, 128]
  %58[INT64, scalar]
  %59[FLOAT, 128x128x3x3]
  %60[FLOAT, 128]
  %61[FLOAT, 128]
  %62[FLOAT, 128]
  %63[FLOAT, 128]
  %64[INT64, scalar]
  %65[FLOAT, 128x128x3x3]
  %66[FLOAT, 128]
  %67[FLOAT, 128]
  %68[FLOAT, 128]
  %69[FLOAT, 128]
  %70[INT64, scalar]
  %71[FLOAT, 128x128x3x3]
  %72[FLOAT, 128]
  %73[FLOAT, 128]
  %74[FLOAT, 128]
  %75[FLOAT, 128]
  %76[INT64, scalar]
  %77[FLOAT, 128x128x3x3]
  %78[FLOAT, 128]
  %79[FLOAT, 128]
  %80[FLOAT, 128]
  %81[FLOAT, 128]
  %82[INT64, scalar]
  %83[FLOAT, 128x128x3x3]
  %84[FLOAT, 128]
  %85[FLOAT, 128]
  %86[FLOAT, 128]
  %87[FLOAT, 128]
  %88[INT64, scalar]
  %89[FLOAT, 128x128x2x2]
  %90[FLOAT, 128]
  %91[FLOAT, 128]
  %92[FLOAT, 128]
  %93[FLOAT, 128]
  %94[INT64, scalar]
  %95[FLOAT, 256x128x3x3]
  %96[FLOAT, 256]
  %97[FLOAT, 256]
  %98[FLOAT, 256]
  %99[FLOAT, 256]
  %100[INT64, scalar]
  %101[FLOAT, 256x256x3x3]
  %102[FLOAT, 256]
  %103[FLOAT, 256]
  %104[FLOAT, 256]
  %105[FLOAT, 256]
  %106[INT64, scalar]
  %107[FLOAT, 256x256x3x3]
  %108[FLOAT, 256]
  %109[FLOAT, 256]
  %110[FLOAT, 256]
  %111[FLOAT, 256]
  %112[INT64, scalar]
  %113[FLOAT, 256x256x3x3]
  %114[FLOAT, 256]
  %115[FLOAT, 256]
  %116[FLOAT, 256]
  %117[FLOAT, 256]
  %118[INT64, scalar]
  %119[FLOAT, 256x256x3x3]
  %120[FLOAT, 256]
  %121[FLOAT, 256]
  %122[FLOAT, 256]
  %123[FLOAT, 256]
  %124[INT64, scalar]
  %125[FLOAT, 256x256x3x3]
  %126[FLOAT, 256]
  %127[FLOAT, 256]
  %128[FLOAT, 256]
  %129[FLOAT, 256]
  %130[INT64, scalar]
  %131[FLOAT, 256x128x4x4]
  %132[FLOAT, 128]
  %133[FLOAT, 128]
  %134[FLOAT, 128]
  %135[FLOAT, 128]
  %136[INT64, scalar]
  %137[FLOAT, 2x384x1x1]
  %138[FLOAT, 2]
  %139[FLOAT, 14x384x1x1]
  %140[FLOAT, 14]
  %141[FLOAT, 4x384x1x1]
  %142[FLOAT, 4]
  %143[FLOAT, 1]
  %144[FLOAT, 1]
  %145[FLOAT, 1]
  %146[FLOAT, 1]
  %147[FLOAT, 1]
  %148[FLOAT, 1]
  %149[FLOAT, 7]
  %150[FLOAT, 7]
  %151[FLOAT, 7]
  %152[FLOAT, 7]
  %153[FLOAT, 1]
  %154[FLOAT, 1]
  %155[FLOAT, 1]
  %156[FLOAT, 1]
  %157[FLOAT, 1]
  %158[FLOAT, 1]
) {
  %159 = Concat[axis = 1](%pillar_x, %pillar_y, %pillar_z)
  %160 = ReduceSum[axes = [3], keepdims = 1](%159)
  %161 = Constant[value = <Tensor>]()
  %162 = Reshape(%num_points_per_pillar, %161)
  %163 = Div(%160, %162)
  %164 = Sub(%159, %163)
  %165 = Sub(%pillar_x, %x_sub_shaped)
  %166 = Sub(%pillar_y, %y_sub_shaped)
  %167 = Concat[axis = 1](%165, %166)
  %168 = Concat[axis = 1](%pillar_x, %pillar_y, %pillar_z, %pillar_i)
  %169 = Concat[axis = 1](%168, %164, %167)
  %170 = Mul(%169, %mask)
  %171 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%170, %15, %16)
  %172 = BatchNormalization[epsilon = 0.00100000004749745, momentum = 1](%171, %10, %11, %12, %13)
  %173 = Relu(%172)
  %174 = Conv[dilations = [1, 3], group = 1, kernel_shape = [1, 34], pads = [0, 0, 0, 0], strides = [1, 1]](%173, %21, %22)
  return %174
}

```


##　测试onnx模型推理结果
需要安装onnxruntime:<https://github.com/microsoft/onnxruntime#installation>
onnxruntime-1.2.0
``` python
import onnxruntime as ort
def do_inference(onnx_path):
    ort_session = ort.InferenceSession(pfe)
    input = {'pillar_x':np.random.randn(1, 1, 12000, 100).astype(np.float32),
             'pillar_y':np.random.randn(1, 1, 12000, 100).astype(np.float32),
             'pillar_z':np.random.randn(1, 1, 12000, 100).astype(np.float32),
             'pillar_i':np.random.randn(1, 1, 12000, 100).astype(np.float32),
             'num_points_per_pillar':np.random.randn(1,12000).astype(np.float32),
             'x_sub_shaped':np.random.randn(1, 1, 12000, 100).astype(np.float32),
             'y_sub_shaped':np.random.randn(1, 1, 12000, 100).astype(np.float32),
             'mask':np.random.randn(1, 1, 12000, 100).astype(np.float32)
            }
    outputs = ort_session.run(None, input)

    # print(outputs[0])
    print(len(outputs),type(outputs[0]),outputs[0].shape)
    return outputs

pfe = '/home/sc/disk/git/kitti_pretrained_point_pillars/pfe.onnx'
do_inference(pfe)
```



### 查看onnx模型输入的 shape
``` python
import onnx

model = onnx.load(r"model.onnx")

# The model is represented as a protobuf structure and it can be accessed
# using the standard python-for-protobuf methods

# iterate through inputs of the graph
for input in model.graph.input:
    print (input.name, end=": ")
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape:
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                print (d.dim_value, end=", ")  # known dimension
            elif (d.HasField("dim_param")):
                print (d.dim_param, end=", ")  # unknown dimension with symbolic name
            else:
                print ("?", end=", ")  # unknown dimension with no name
    else:
        print ("unknown rank", end="")
    print()
```



maybe可以参考:<https://github.com/kindlychung/demo-load-pb-tensorflow>  
没有真的试过