# 使用说明

**注意把cfg文件中的batch_size修改为1**

# 改进版yolov3
python darknet2onnx_test.py ./data/lishui_0902/yolov3-giou-lishui0902.cfg ./data/lishui_0902/yolov3-giou-lishui0902_last.weights ./data/lishui_0902/trafficlights2.names --input_size 416 -o data/lishui_0902/yolov3.onnx


# 标准版yolov3
python darknet2onnx.py ./data/lishui_0908/yolov3-giou.cfg ./data/lishui_0908/yolov3-giou_last.weights ./data/lishui_0908/trafficlights2.names --input_size 416 -o data/lishui_0908/yolov3.onnx

# 改进版yolov3
python darknet2onnx_0914.py ./data/lishui_0914/yolov3-giou.cfg ./data/lishui_0914/yolov3-giou_last.weights ./data/lishui_0914/trafficlights2.names --input_size 416 -o data/lishui_0914/yolov3.onnx
