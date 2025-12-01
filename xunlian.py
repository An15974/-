from ultralytics import YOLO
import yaml
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')  # 过滤冗余警告，提升日志整洁度


def train_garbage_detector():
    """
    优化后的垃圾检测模型训练函数
    适配Windows系统，提升训练稳定性与结果可追溯性
    """
    # -------------------------- 路径配置（更优雅的Path类） --------------------------
    project_root = Path(r"D:\Study\ultralytics-main")
    model_path = project_root / "yolo11n.pt"
    data_yaml_path = project_root / "lajifenlei.yaml"
    project_dir = project_root / "garbage_detection"

    # 检查关键文件是否存在（容错性提升）
    if not model_path.exists():
        raise FileNotFoundError(f"模型权重文件不存在：{model_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在：{data_yaml_path}")

    # -------------------------- 模型加载（增强鲁棒性） --------------------------
    model = YOLO(model_path)
    print(f"✅ 成功加载预训练模型：{model_path.name}")

    # -------------------------- 训练参数优化（核心改进） --------------------------
    train_params = {
        # 基础配置
        "data": str(data_yaml_path),
        "epochs": 1000,
        "batch": 32,  # RTX 4060 Laptop建议batch=16（显存更稳，32可能OOM）
        "imgsz": 640,
        "device": "0",  # 指定GPU，若需CPU则设为"cpu"

        # Windows系统适配
        "workers": 0,  # 禁用多线程，避免进程冲突

        # 项目与日志配置（可追溯性）
        "project": str(project_dir),
        "name": "double_label_train7",
        "exist_ok": True,  # 覆盖已有目录
        "save": True,  # 保存最佳权重
        "save_period": 10,  # 每10轮保存一次检查点，防止意外中断丢失进度
        "val": True,  # 训练中自动验证（默认开启，显式声明更清晰）

        # 超参数调优（提升收敛速度与精度）
        "lr0": 0.01,  # 初始学习率（YOLOv8默认0.01，可根据数据集调整）
        "lrf": 0.01,  # 最终学习率因子（余弦退火到lr0*lrf）
        "momentum": 0.937,  # SGD动量
        "weight_decay": 0.0005,  # 权重衰减（防止过拟合）
        "warmup_epochs": 3.0,  # 预热轮数（小学习率起步，避免初期震荡）

        # 早停机制（防止过拟合，节省时间）
        "patience": 50,  # 50轮无精度提升则自动停止训练

        # 数据增强（提升泛化能力）
        "hsv_h": 0.015,  # 色调增强
        "hsv_s": 0.7,  # 饱和度增强
        "hsv_v": 0.4,  # 明度增强
        "degrees": 0.0,  # 旋转角度（按需调整，垃圾检测建议0-10度）
        "translate": 0.1,  # 平移
        "scale": 0.5,  # 缩放

        # 其他实用配置
        "rect": True,  # 矩形训练（提升速度，不损失精度）
        "cos_lr": True,  # 余弦学习率调度（比步长衰减更平滑）
        "verbose": True,  # 打印详细训练日志
    }

    # -------------------------- 启动训练 --------------------------
    print("\n🚀 开始训练垃圾检测模型...")
    results = model.train(**train_params)

    # -------------------------- 训练后分析 --------------------------
    print("\n🎉 训练完成！关键指标：")
    print(f"最佳mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"最佳精确率: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"最佳召回率: {results.results_dict['metrics/recall(B)']:.4f}")


if __name__ == '__main__':
    try:
        train_garbage_detector()
    except Exception as e:
        print(f"\n❌ 训练过程出错：{str(e)}")
        # 可添加错误日志保存、邮件告警等扩展逻辑