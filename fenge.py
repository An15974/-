import os
import random
import shutil
from tqdm import tqdm  # 进度条显示

# 配置路径
img_dir = r"D:\Study\ultralytics-main\MyDataset\JPEGImages"  # 原始图片文件夹路径
label_dir = r"D:\Study\ultralytics-main\MyDataset\labels"  # 原始标注文件夹路径
output_dir = r"D:\Study\ultralytics-main\datasets"  # 输出根目录

# 创建输出目录结构
os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)

# 获取所有图片文件列表（支持多种格式）
img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
img_files = [f for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in img_exts]

# 过滤存在对应标注文件的图片
valid_files = []
for img_file in img_files:
    base_name = os.path.splitext(img_file)[0]
    label_file = f"{base_name}.txt"
    label_path = os.path.join(label_dir, label_file)

    if os.path.exists(label_path):
        valid_files.append((img_file, label_file))
    else:
        print(f"警告：{img_file} 没有对应的标注文件，已跳过")

# 设置随机种子保证可重复性
random.seed(42)
random.shuffle(valid_files)

# 计算分割点
split_idx = int(len(valid_files) * 0.8)
train_files = valid_files[:split_idx]
val_files = valid_files[split_idx:]


def copy_files(file_list, subset):
    """复制文件到目标目录"""
    for img_file, label_file in tqdm(file_list, desc=f"处理 {subset} 集"):
        # 复制图片
        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(output_dir, f"images/{subset}", img_file)
        shutil.copy(src_img, dst_img)

        # 复制标注
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(output_dir, f"labels/{subset}", label_file)
        shutil.copy(src_label, dst_label)


# 执行复制
copy_files(train_files, "train")
copy_files(val_files, "val")

# 输出统计信息
print(f"\n分割完成！")
print(f"总有效样本: {len(valid_files)}")
print(f"训练集数量: {len(train_files)} ({len(train_files)})")
print(f"验证集数量: {len(val_files)} ({len(val_files)})")
print(f"输出目录结构：")
print(f"├── {output_dir}/images/train")
print(f"├── {output_dir}/images/val")
print(f"├── {output_dir}/labels/train")
print(f"└── {output_dir}/labels/val")