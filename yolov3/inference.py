# encoding=utf-8
import onnxruntime as ort
import data_process
from data_process import draw_bboxes, ALL_CATEGORIES
import cv2 as cv


# ###############onnx inference##############
# input是个dict key为node name用netron打开模型可以查看
def do_onnx_inference(onnx_path, input):
    ort_session = ort.InferenceSession(onnx_path)
    outputs = ort_session.run(None, input)

    # print(outputs[0])
    # print(len(outputs),type(outputs[0]),outputs[0].shape)
    print(len(outputs))
    for output in outputs:
        print(output.shape)

    return outputs


img_path = './data/lishui_0902/lishui_tl.png'
input_resolution_yolov3_HW = (416, 416)
preprocessor = data_process.PreprocessYOLO(input_resolution_yolov3_HW)
image_raw, image_preprocessed = preprocessor.process(img_path)
x, y = 169, 196
for c in range(3):
    print(255*image_preprocessed[0][c][y][x])

shape_orig_WH = image_raw.size

model_input = {'000_net': image_preprocessed}
model_outputs = do_onnx_inference('./yolov3.onnx', model_input)

postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                      "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                       (59, 119), (116, 90), (156, 198), (373, 326)],
                      "obj_threshold": 0.6,         # 对象覆盖的阈值，[0,1]之间
                      "nms_threshold": 0.5,       # nms的阈值，[0,1]之间
                      "yolo_input_resolution": input_resolution_yolov3_HW}

postprocessor = data_process.PostprocessYOLO(**postprocessor_args)
boxes, classes, scores = postprocessor.process(model_outputs, (shape_orig_WH))
obj_detected_img = draw_bboxes(
    image_raw, boxes, scores, classes, ALL_CATEGORIES)
output_image_path = 'lights_bboxes.png'
obj_detected_img.save(output_image_path, 'PNG')

print('Saved image with bounding boxes of detected objects to {}.'.format(
    output_image_path))
