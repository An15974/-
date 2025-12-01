import os
from PIL import Image


def normalize_labels(data_dir):
    """
    将标签文件从原始像素坐标转换为归一化坐标（YOLO 格式）
    :param data_dir: 数据集根目录（包含 images/ 和 labels/）
    """
    # 支持的图片格式
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    for split in ['train', 'val']:
        labels_dir = os.path.join(data_dir, 'labels', split)
        images_dir = os.path.join(data_dir, 'images', split)

        # 检查目录是否存在
        if not os.path.exists(labels_dir):
            print(f"警告：标签目录不存在 {labels_dir}")
            continue
        if not os.path.exists(images_dir):
            print(f"警告：图片目录不存在 {images_dir}")
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            label_path = os.path.join(labels_dir, label_file)

            # 获取标签文件名（不含后缀）
            label_name = os.path.splitext(label_file)[0]

            # 寻找匹配的图片文件（处理_original后缀、多格式）
            img_path = None
            # 1. 先尝试带_original的后缀
            for ext in img_extensions:
                candidate = os.path.join(images_dir, f"{label_name}_original{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            # 2. 再尝试原名称
            if not img_path:
                for ext in img_extensions:
                    candidate = os.path.join(images_dir, f"{label_name}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

            # 图片不存在则跳过并提示
            if not img_path:
                print(f"跳过：未找到 {label_file} 对应的图片（{images_dir}/{label_name}.*）")
                continue

            # 获取图片尺寸
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"错误：无法打开图片 {img_path} → {e}")
                continue

            # 读取并修正标签
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:  # 跳过无效行
                    print(f"警告：{label_file} 中无效行 → {line}")
                    continue

                # 解析坐标（兼容整数/浮点数）
                try:
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:5])
                except ValueError:
                    print(f"警告：{label_file} 中坐标格式错误 → {line}")
                    continue

                # 归一化坐标（YOLO要求：x_center/y_center/w/h 相对于图片宽高，范围0-1）
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # 确保坐标在合法范围
                x_center_norm = max(0.0, min(1.0, x_center_norm))
                y_center_norm = max(0.0, min(1.0, y_center_norm))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))

                new_lines.append(
                    f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

            # 保存修正后的标签
            with open(label_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"已修正：{label_path}（对应图片：{os.path.basename(img_path)}）")


# 使用示例（修改为您的数据集路径）
if __name__ == "__main__":
    normalize_labels(r"D:\Study\ultralytics-main\datasets")