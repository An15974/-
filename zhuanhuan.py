import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_xml(xml_path):
    """解析单个XML文件，返回图片信息和目标列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片基本信息
    size = root.find('size')
    width = float(size.find('width').text)  # 改为浮点数
    height = float(size.find('height').text)  # 改为浮点数
    filename = root.find('filename').text

    # 解析目标框（支持浮点数）
    objects = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        bbox = obj.find('bndbox')
        # 将坐标转换为浮点数
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        objects.append({'class': cls, 'bbox': (xmin, ymin, xmax, ymax)})

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_bbox_to_yolo(bbox, width, height):
    """将VOC的边界框转换为YOLO格式（浮点数处理）"""
    xmin, ymin, xmax, ymax = bbox
    # 计算中心点和宽高（使用浮点数）
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height
    return [x_center, y_center, bbox_width, bbox_height]


def generate_class_map(xml_dir):
    """从XML文件中提取所有类别生成映射字典"""
    class_counter = defaultdict(int)
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        data = parse_xml(xml_path)
        for obj in data['objects']:
            class_counter[obj['class']] += 1

    # 按字母顺序生成类别ID
    classes = sorted(class_counter.keys())
    class_id_map = {cls: idx for idx, cls in enumerate(classes)}
    return class_id_map, classes


def convert_xml_to_yolo(xml_dir, output_dir, class_map=None):
    """批量转换XML文件为YOLO格式（支持浮点数）"""
    os.makedirs(output_dir, exist_ok=True)

    if not class_map:
        class_map, classes = generate_class_map(xml_dir)
        print(f"检测到{len(classes)}个类别：{classes}")
        # 保存类别列表到文件
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            f.write('\n'.join(classes))
    else:
        classes = list(class_map.keys())

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        data = parse_xml(xml_path)

        # 生成YOLO格式的标注（保留4位小数）
        yolo_lines = []
        for obj in data['objects']:
            cls = obj['class']
            if cls not in class_map:
                print(f"警告：类别{cls}未在映射中，跳过该标注")
                continue
            bbox = obj['bbox']
            yolo_bbox = convert_bbox_to_yolo(bbox, data['width'], data['height'])
            # 格式化输出（保留4位小数）
            yolo_line = f"{class_map[cls]} " + " ".join([f"{v:.4f}" for v in yolo_bbox]) + "\n"
            yolo_lines.append(yolo_line)

        # 保存为TXT文件
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        output_path = os.path.join(output_dir, txt_filename)
        with open(output_path, 'w') as f:
            f.writelines(yolo_lines)

    print(f"转换完成，输出路径：{output_dir}")


if __name__ == "__main__":
    # 配置参数
    xml_dir = r"D:\Study\ultralytics-main\MyDataset\Annotations"  # XML文件所在目录
    output_dir = r"D:\Study\ultralytics-main\MyDataset\labels"  # 输出YOLO标注的目录

    # 自动检测类别并生成映射
    convert_xml_to_yolo(xml_dir, output_dir)