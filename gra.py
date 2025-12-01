import gradio as gr
from PIL import Image
import torch
from torchvision.transforms import functional as F
import numpy as np
import cv2
from ultralytics import YOLO

# 加载 YOLO 模型
model_path = r"D:\Study\ultralytics-main\runs\detect\train4\weights\best.pt"
model = YOLO(model_path)  # 使用 ultralytics 库加载模型

def detect_objects(image):
    """
    使用 YOLO 模型对输入图片进行目标检测，并返回带有目标框的图片。
    """
    # 将 PIL 图像转换为 numpy 数组
    image_np = np.array(image)

    # 使用模型进行推理
    results = model(image_np)

    # 获取检测结果
    detections = results[0].boxes  # 获取检测框对象

    if detections is not None and len(detections) > 0:
        # 提取检测框的坐标、置信度和类别
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 获取坐标
            conf = box.conf[0].item()  # 获取置信度
            cls = int(box.cls[0].item())  # 获取类别索引
            label = f"{model.names[cls]} {conf:.2f}"

            # 在图片上绘制目标框和标签
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将 numpy 数组转换回 PIL 图像
    result_image = Image.fromarray(image_np)
    return result_image

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🎯 YOLOv11 目标检测界面")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 上传图片")
            input_image = gr.Image(label="输入图片", type="pil")
        with gr.Column(scale=1):
            gr.Markdown("## 检测结果")
            output_image = gr.Image(label="目标检测结果", type="pil")

    # 按钮触发检测
    detect_button = gr.Button("开始检测")
    detect_button.click(fn=detect_objects, inputs=input_image, outputs=output_image)

# 启动 Gradio 应用
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)