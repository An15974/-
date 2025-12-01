import os
import re
import time
import cv2
import yaml
import base64
import tempfile
import logging
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import shutil

# ------------------ 配置优化 ------------------
MODEL_PATH = os.getenv("GARBAGE_MODEL_PATH", "garbage_detection/double_label_train6/weights/best.pt")
YAML_PATH = os.getenv("GARBAGE_YAML_PATH", "lajifenlei.yaml")
MODEL_PATH = str(Path(MODEL_PATH).resolve())
YAML_PATH = str(Path(YAML_PATH).resolve())

TEMP_DIR = os.getenv("GARBAGE_TEMP_DIR", tempfile.gettempdir())
PROCESSED_VIDEO_DIR = os.path.join(TEMP_DIR, "processed_videos")
DEBUG_SAVE_DIR = os.path.join(TEMP_DIR, "debug_frames")
for dir_path in [PROCESSED_VIDEO_DIR, DEBUG_SAVE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

DEBUG_SAVE_FRAME = os.getenv("DEBUG_SAVE_FRAME", "False").lower() == "true"
MAX_HISTORY_LENGTH = 20

# ------------------ 日志配置 ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

log_file = os.path.join(TEMP_DIR, "garbage_detection.log")
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# ------------------ 全局变量 ------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000"]}})

model = None
model_loaded = False
big_category_mapping = {}
big_category_names = {}
small_category_names = {}
history_messages = []
is_detecting_flag = False

# ------------------ 前端页面模板 ------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>垃圾分类检测系统</title>
</head>
<body>
    <h1>垃圾分类检测系统</h1>
    <p>后端服务运行中</p>
</body>
</html>
"""


# ------------------ 工具函数 ------------------
def base64_to_cv2(base64_str):
    try:
        if not base64_str or not isinstance(base64_str, str):
            return None, "Base64非法（为空或非字符串）"

        if "," in base64_str:
            header, base64_data = base64_str.split(",", 1)
            if not re.match(r'^data:image/(jpeg|png|webp);base64$', header):
                logger.warning(f"非标准图像头: {header[:30]}...")
        else:
            base64_data = base64_str

        base64_data = base64_data.rstrip('=')
        pad_count = 4 - (len(base64_data) % 4)
        if pad_count < 4:
            base64_data += "=" * pad_count

        img_bytes = base64.b64decode(base64_data, validate=True)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return None, "无法解码为图像（格式错误或损坏）"

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return None, f"图像尺寸过小: {w}x{h}（最小20x20）"
        if h > 4000 or w > 4000:
            return None, f"图像尺寸过大: {w}x{h}（最大4000x4000）"

        return img, "ok"
    except base64.binascii.Error as e:
        logger.error(f"Base64解码错误: {str(e)}")
        return None, f"Base64格式错误: {str(e)}"
    except Exception as e:
        logger.exception("base64_to_cv2 异常")
        return None, f"处理失败: {str(e)}"


def cv2_to_base64(img_bgr, quality=85):
    try:
        if img_bgr is None:
            logger.error("cv2_to_base64: 输入图像为空")
            return None

        h, w = img_bgr.shape[:2]
        max_size = 2000
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
        pil = Image.fromarray(img_rgb)
        buf = BytesIO()
        pil.save(buf, format='JPEG', quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/jpeg;base64," + b64
    except Exception as e:
        logger.exception("cv2_to_base64 异常")
        return None


def draw_detection_results(pil_img, img_bgr, results):
    draw = ImageDraw.Draw(pil_img)
    font_paths = ["C:/Windows/Fonts/simhei.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 18)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
        logger.warning("未找到中文字体，使用默认字体")

    detected = []
    img_height, img_width = img_bgr.shape[:2]

    for res in results:
        try:
            boxes = getattr(res, "boxes", [])
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                x1_abs, y1_abs, x2_abs, y2_abs = map(int, xyxy)
                x1 = round(x1_abs / img_width, 4) if img_width != 0 else 0.0
                y1 = round(y1_abs / img_height, 4) if img_height != 0 else 0.0
                x2 = round(x2_abs / img_width, 4) if img_width != 0 else 0.0
                y2 = round(y2_abs / img_height, 4) if img_height != 0 else 0.0
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
                cls_idx = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], 'cpu') else int(box.cls[0])

                small_name = small_category_names.get(cls_idx, f"未知类别({cls_idx})")
                big_idx = big_category_mapping.get(cls_idx, -1)
                big_name = big_category_names.get(big_idx, f"未知大类({big_idx})") if big_idx != -1 else "未知大类"

                # 后端不再生成AI介绍，改为空字符串，由前端填充
                detected.append({
                    "label": f"{big_name}/{small_name}",
                    "big_name": big_name,
                    "small_name": small_name,
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "introduction": ""  # 前端填充介绍
                })

                draw.rectangle([(x1_abs, y1_abs), (x2_abs, y2_abs)], outline=(0, 255, 0), width=3)
                text = f"{big_name}/{small_name} {conf:.2f}"
                text_bbox = draw.textbbox((x1_abs, y1_abs - 22), text, font=font)
                draw.rectangle(text_bbox, fill=(0, 255, 0))
                draw.text((x1_abs, y1_abs - 22), text, fill=(0, 0, 0), font=font)
        except Exception as e:
            logger.exception(f"绘制检测结果失败: {str(e)}")
            continue

    return pil_img, detected


# ------------------ 公共检测逻辑 ------------------
def process_media(media_bgr, is_video_frame=False):
    global model, history_messages
    if model is None or not model_loaded:
        return None, "模型未加载，请先初始化"

    try:
        imgsz = 640 if not is_video_frame else 480
        results = model(media_bgr, conf=0.5, imgsz=imgsz, verbose=False)
    except Exception as e:
        err_msg = f"模型推理失败: {str(e)}"
        logger.exception(err_msg)
        return None, err_msg

    media_rgb = cv2.cvtColor(media_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(media_rgb)
    pil_drawn, detected = draw_detection_results(pil_img, media_bgr, results)

    intro = "暂无检测物体" if not detected else ""  # 前端填充介绍
    detected_label = detected[0]['label'] if detected else ""
    if detected:
        history_type = "[实时检测]" if is_video_frame else "[图片检测]"
        history_item = f"{history_type} {detected[0]['label']}（置信度{detected[0]['confidence']:.2f}）\n"
        history_messages.append(history_item)
        if len(history_messages) > MAX_HISTORY_LENGTH:
            history_messages.pop(0)

    buf = BytesIO()
    pil_drawn.save(buf, format='JPEG', quality=85)
    annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "annotated_base64": annotated_base64,
        "introduction": intro,
        "history": "\n".join(history_messages[-10:]),
        "detected_label": detected_label,
        "detections": detected
    }, "ok"


# ------------------ Flask路由 ------------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/init_model', methods=['GET'])
def init_model_endpoint():
    global model, model_loaded, big_category_mapping, big_category_names, small_category_names
    if model_loaded:
        return jsonify({"code": 0, "msg": "✅ 模型已加载，可直接开始检测"})

    try:
        if not os.path.exists(MODEL_PATH):
            err_msg = f"模型文件不存在: {MODEL_PATH}"
            logger.error(err_msg)
            return jsonify({"code": 1, "msg": err_msg})
        if not os.path.exists(YAML_PATH):
            err_msg = f"YAML配置不存在: {YAML_PATH}"
            logger.error(err_msg)
            return jsonify({"code": 2, "msg": err_msg})

        try:
            model = YOLO(MODEL_PATH)
            logger.info(f"模型加载成功: {MODEL_PATH}")
        except Exception as e:
            err_msg = f"模型加载失败: {str(e)}"
            logger.exception(err_msg)
            return jsonify({"code": 3, "msg": err_msg})

        try:
            warmup_img = np.zeros((480, 640, 3), dtype=np.uint8)
            model.predict(source=warmup_img, conf=0.5, imgsz=640, stream=False, verbose=False)
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {str(e)}")

        try:
            with open(YAML_PATH, 'r', encoding='utf-8-sig') as f:
                cfg = yaml.safe_load(f)
            big_category_mapping = cfg.get("big_category_mapping", {})
            big_category_names = cfg.get("big_category_names", {})
            small_category_names = cfg.get("names", {})
            logger.info(f"YAML配置加载成功，包含{len(small_category_names)}个小类别")
        except Exception as e:
            err_msg = f"YAML解析失败: {str(e)}"
            logger.exception(err_msg)
            return jsonify({"code": 4, "msg": err_msg})

        model_loaded = True
        return jsonify({"code": 0, "msg": "✅ 模型加载成功，可开始检测"})
    except Exception as e:
        logger.exception("模型初始化异常")
        return jsonify({"code": 500, "msg": f"初始化失败: {str(e)}"})


@app.route('/start', methods=['POST'])
def start_endpoint():
    global is_detecting_flag
    is_detecting_flag = True
    logger.info("检测启动")
    return jsonify({"code": 0, "msg": "started"})


@app.route('/stop', methods=['POST'])
def stop_endpoint():
    global is_detecting_flag
    is_detecting_flag = False
    logger.info("检测停止")
    return jsonify({"code": 0, "msg": "stopped"})


@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    try:
        if not request.is_json:
            return jsonify({"code": 400, "msg": "请求格式错误，请使用application/json"})
        data = request.json or {}
        frame_base64 = data.get('frame_base64', '').strip()
        if not frame_base64:
            return jsonify({"code": 400, "msg": "缺少frame_base64字段"})

        frame_bgr, err = base64_to_cv2(frame_base64)
        if frame_bgr is None:
            return jsonify({"code": 400, "msg": f"帧解码失败: {err}"})

        if DEBUG_SAVE_FRAME:
            try:
                save_path = os.path.join(DEBUG_SAVE_DIR, f"frame_{int(time.time())}.jpg")
                cv2.imwrite(save_path, frame_bgr)
            except Exception as e:
                logger.warning(f"调试帧保存失败: {e}")

        result, msg = process_media(frame_bgr, is_video_frame=True)
        if result is None:
            return jsonify({"code": 500, "msg": msg})

        return jsonify({"code": 0, "msg": "ok", "data": result})
    except Exception as e:
        logger.exception("帧处理异常")
        return jsonify({"code": 500, "msg": f"帧处理失败: {str(e)}"})


@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    try:
        if not request.is_json:
            return jsonify({"code": 400, "msg": "请求格式错误"})
        data = request.json or {}
        image_base64 = data.get('image_base64', '').strip()
        if not image_base64:
            return jsonify({"code": 400, "msg": "缺少image_base64字段"})

        img_bgr, err = base64_to_cv2(image_base64)
        if img_bgr is None:
            return jsonify({"code": 400, "msg": f"图片解码失败: {err}"})

        result, msg = process_media(img_bgr, is_video_frame=False)
        if result is None:
            return jsonify({"code": 500, "msg": msg})

        return jsonify({"code": 0, "msg": "ok", "data": result})
    except Exception as e:
        logger.exception("图片处理异常")
        return jsonify({"code": 500, "msg": f"图片处理失败: {str(e)}"})


@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    global history_messages
    try:
        if 'video' not in request.files:
            return jsonify({"code": 400, "msg": "缺少视频文件"})

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"code": 400, "msg": "视频文件名为空"})

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_file.filename)[1], delete=False) as tmp:
            input_path = tmp.name
            video_file.save(tmp)
        logger.info(f"视频保存成功: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            os.unlink(input_path)
            return jsonify({"code": 500, "msg": "无法打开视频"})

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        logger.info(f"视频信息: {width}x{height}, {fps:.1f}FPS")

        out_name = f"processed_{int(time.time())}_{os.path.splitext(video_file.filename)[0]}.mp4"
        out_path = os.path.join(PROCESSED_VIDEO_DIR, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            os.unlink(input_path)
            return jsonify({"code": 500, "msg": "视频写入器初始化失败"})

        detected_set = {}
        frame_idx = 0
        video_duration = total_frames / fps if fps > 0 else 0
        sample_interval = max(1, int(fps / (10 if video_duration < 60 else 5 if video_duration < 300 else 2)))
        logger.info(f"视频采样间隔: {sample_interval}帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_interval != 0:
                out.write(frame)
                continue

            try:
                results = model(frame, conf=0.5, imgsz=480, verbose=False)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_drawn, detected = draw_detection_results(pil_img, frame, results)
                out_frame = cv2.cvtColor(np.array(pil_drawn), cv2.COLOR_RGB2BGR)
                out.write(out_frame)

                for d in detected:
                    if d['label'] not in detected_set or d['confidence'] > detected_set[d['label']]['confidence']:
                        detected_set[d['label']] = d
            except Exception as e:
                logger.warning(f"处理第{frame_idx}帧失败: {str(e)}")
                out.write(frame)
                continue

        cap.release()
        out.release()
        os.unlink(input_path)
        logger.info(f"视频处理完成: {out_path}")

        detections = sorted(list(detected_set.values()), key=lambda x: x['confidence'], reverse=True)
        if detections:
            detected_labels = [d['label'] for d in detections]
            final_intro = ""  # 前端填充介绍
            full_intro = ""
        else:
            final_intro = "未检测到物体"
            full_intro = final_intro
            detections = []

        history_item = f"[视频检测] {os.path.basename(video_file.filename)}\n{final_intro}\n"
        history_messages.append(history_item)
        if len(history_messages) > MAX_HISTORY_LENGTH:
            history_messages.pop(0)

        return jsonify({
            "code": 0,
            "msg": "视频处理完成",
            "data": {
                "download_url": f"/download_processed_video/{out_name}",
                "introduction": full_intro,
                "history": "\n".join(history_messages[-10:]),
                "detections": detections,
                "detected_labels": detected_labels if detections else []
            }
        })
    except Exception as e:
        logger.exception("视频处理异常")
        return jsonify({"code": 500, "msg": f"视频处理失败: {str(e)}"})


@app.route('/download_processed_video/<filename>')
def download_processed_video(filename):
    try:
        safe_filename = os.path.basename(filename)
        video_path = os.path.join(PROCESSED_VIDEO_DIR, safe_filename)
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            logger.error(f"视频不存在: {video_path}")
            return jsonify({"code": 404, "msg": "视频文件不存在"}), 404
        if not video_path.endswith('.mp4'):
            logger.warning(f"禁止下载非视频文件: {video_path}")
            return jsonify({"code": 403, "msg": "不支持的文件类型"}), 403
        return send_from_directory(PROCESSED_VIDEO_DIR, safe_filename, as_attachment=True)
    except Exception as e:
        logger.exception("视频下载异常")
        return jsonify({"code": 500, "msg": f"下载失败: {str(e)}"}), 500


# ------------------ 启动配置 ------------------
def cleanup_temp_files():
    try:
        if os.path.exists(DEBUG_SAVE_DIR):
            files = sorted(Path(DEBUG_SAVE_DIR).glob('*.jpg'), key=os.path.getmtime)
            for f in files[:-100]:
                os.remove(f)
        if os.path.exists(PROCESSED_VIDEO_DIR):
            now = time.time()
            for f in Path(PROCESSED_VIDEO_DIR).glob('*.mp4'):
                if now - os.path.getmtime(f) > 86400:
                    os.remove(f)
        logger.info("临时文件清理完成")
    except Exception as e:
        logger.warning(f"临时文件清理失败: {str(e)}")


if __name__ == '__main__':
    try:
        cleanup_temp_files()
        logger.info(f"服务启动，视频保存目录: {PROCESSED_VIDEO_DIR}")
        logger.info(f"后端服务地址: http://0.0.0.0:5000")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    finally:
        cleanup_temp_files()