import tempfile

import gradio as gr
import yaml
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import cv2

# === 新增：详细垃圾分类介绍字典（已按要求详细扩展） ===
GARBAGE_INTRODUCTION = {
    # 其他垃圾（大类0）
    "其他垃圾/污损塑料": "其他垃圾是无法回收且无害的废弃物，长期堆积会占用土地资源。污损塑料指被污染、破损或难以清洁的塑料制品，如沾有油污的塑料袋、食物残渣的塑料容器等。其具体作用是临时包装或盛放物品，但一旦被污染，失去再利用价值。可回收性极低，因为污染会破坏塑料的再利用价值。危害性较低，但填埋占用土地，焚烧可能释放有害气体，长期影响环境质量。对环境影响：填埋占用土地并产生甲烷，对环境有轻微但持续的危害；长期积累可能释放微塑料，影响土壤和水体质量。对人类和动物影响：直接危害较小，但可能通过食物链间接影响健康。",

    "其他垃圾/烟蒂": "其他垃圾无回收价值，随意丢弃易造成视觉污染。烟蒂是香烟过滤嘴，主要成分是醋酸纤维素，含有尼古丁、焦油等化学物质。其具体作用是过滤吸烟时的有害物质，但本身是不可回收的垃圾。可回收性极低，因为含有化学物质且难以清洁。危害性较高，含有多种有害化学物质，对环境和健康都有潜在危害。对环境影响：烟蒂含尼古丁、焦油等化学物质，可能释放有害物质污染土壤和水体。对人类影响：可能引起火灾，对动物影响：可能造成误食中毒。",

    "其他垃圾/牙签": "其他垃圾难以回收利用，过量堆积会增加处理压力。牙签多为木质或塑料材质，一次性使用后难以回收。其具体作用是清洁牙齿缝隙，但使用后无法再利用。可回收性低，木质牙签难以回收，塑料牙签也因污染和材质问题难以回收。危害性较低，但可能影响土壤结构，塑料牙签被动物误食可能造成伤害。对环境影响：木质牙签可能影响土壤结构，塑料牙签可能被动物误食造成伤害。对人类影响：可能造成轻微割伤。",

    "其他垃圾/破碎花盆及碟碗": "其他垃圾回收价值极低，破碎陶瓷易划伤设备。破碎花盆及碟碗是陶瓷制品的废弃物，主要用途是盛放物品或作为装饰。其具体作用是日常使用，但破损后失去功能。可回收性极低，陶瓷制品通常无法回收。危害性较低，但可能造成割伤，填埋占用空间。对环境影响：破碎陶瓷制品难以降解，可能划伤土壤表面，影响土壤微生物活动。对人类和动物影响：可能造成割伤。",

    "其他垃圾/竹筷": "其他垃圾无法循环利用，易污染且处理成本高。竹筷是一次性竹制品，用于进餐的工具。其具体作用是辅助进食，但使用后无法再利用。可回收性低，竹筷虽然可生物降解，但因污染和材质问题通常不被视为可回收物。危害性较低，但填埋可能影响土壤。对环境影响：竹筷填埋需要较长时间分解，可能影响土壤结构。对人类和动物影响：危害较小，但大量填埋可能影响土壤质量。",

    "其他垃圾/一次性快餐盒": "其他垃圾难以清洁再生，易造成白色污染。一次性快餐盒是用于盛放快餐的容器，通常为塑料制成。其具体作用是盛放快餐。可回收性低，因为塑料快餐盒通常难以清洁和回收。危害性较低，但填埋占用空间，可能释放微塑料。对环境影响：一次性快餐盒多为塑料，填埋难降解，可能释放微塑料。对人类和动物影响：可能被动物误食。",

    # 厨余垃圾（大类1）
    "厨余垃圾/剩饭剩菜": "厨余垃圾是易腐烂生物质废弃物，不及时处理易发霉发臭污染环境。剩饭剩菜可通过堆肥或生物降解制有机肥，回收价值较高。投放前需沥干汤汁，挑出非厨余杂物。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生大量渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地，吸引害虫。",

    "厨余垃圾/大骨头": "厨余垃圾可生物降解，随意丢弃易滋生细菌。大骨头是食物残渣，主要来自肉类食品。其具体作用是提供营养，但食用后成为废弃物。可回收性低，因为大骨头难以降解，不适合常规堆肥。危害性低，但可能影响厨余垃圾处理。对环境影响：大骨头难以堆肥，填埋时可能影响堆肥质量，产生渗滤液。对人类和动物影响：对人类危害小，但可能影响厨余垃圾处理效率。",

    "厨余垃圾/水果果皮": "厨余垃圾易腐烂可循环，是优质堆肥原料，不分类会浪费资源。水果果皮可堆肥转化为有机肥料，回收价值高。投放时无需去除果蒂，大块可撕碎。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地。",

    "厨余垃圾/水果果肉": "厨余垃圾富含有机物，不处理易变质污染土壤水源。水果果肉可通过生物处理制沼气或有机肥，回收价值高。变质果肉仍属厨余垃圾，及时投放。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地。",

    "厨余垃圾/茶叶渣": "厨余垃圾可自然降解，是环保的生物质资源。茶叶渣富含纤维和矿物质，处理方式为堆肥或作为植物肥料，回收价值中等。无需清洗直接投放。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地。",

    "厨余垃圾/菜叶菜根": "厨余垃圾易分解可循环，不分类会增加填埋压力。菜叶菜根可堆肥制有机肥，回收价值高。投放前去除塑料绳等杂物，沥干水分。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地。",

    "厨余垃圾/蛋壳": "厨余垃圾可生物降解，能调节堆肥酸碱度。蛋壳主要成分为碳酸钙，处理方式为破碎后堆肥，回收价值中等。无需清洗，直接破碎投放。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地。",

    "厨余垃圾/鱼骨": "厨余垃圾易腐烂，是生物处理的优质原料。鱼骨（尤其是小鱼骨）质地较软，处理方式为堆肥或生物降解，回收价值高。与其他厨余垃圾一同投放即可。对环境影响：可生物降解，可堆肥处理，减少填埋污染。对人类和动物影响：若未分类，填埋会产生渗滤液和甲烷，污染水土环境，对人类可能传播疾病，对动物可能破坏栖息地。",

    # 可回收物（大类2）
    "可回收物/充电宝": "可回收物能循环利用，减少资源开采和环境污染。充电宝含锂、铜等可回收成分，处理方式为专业拆解回收，回收价值较高。投放时保持干燥，不拆解。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋浪费资源，焚烧可能释放有害气体，对环境有轻微危害，对人类有益，减少环境污染，对动物可能造成电池泄漏污染。",

    "可回收物/包": "可回收物循环利用可节约原材料，降低碳排放。包的皮革、帆布等材质可再生加工，回收价值中等。投放前清理内部杂物，破损严重仍可回收。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/化妆品瓶": "可回收物再生能减少塑料/玻璃消耗，保护资源。化妆品瓶（玻璃/塑料材质）可重新加工为容器，回收价值较高。投放前清空冲洗，取下非同类部件。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/塑料玩具": "可回收物循环利用可降低塑料污染，节约石油资源。塑料玩具多为PP/PE材质，处理方式为熔融再生，回收价值中等。投放前去除电池，清洗污渍。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/塑料碗盆": "可回收物再生能减少白色污染，提升资源利用率。塑料碗盆（PP/PE材质）可熔融重塑，回收价值中等。破损严重的按其他垃圾投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/塑料衣架": "可回收物循环利用可节约塑料原料，减少环境压力。塑料衣架可熔融再生为新塑料制品，回收价值较低。无需拆解，直接放入可回收物箱。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/快递纸袋": "可回收物再生能节约木材资源，减少森林砍伐。快递纸袋可重新制成再生纸，回收价值较高。投放前去除胶带，折叠压平，潮湿的需晾干。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/插头电线": "可回收物回收能节约金属资源，减少矿产开采。插头电线含铜、铝等金属，处理方式为拆解回收金属，回收价值较高。盘绕整齐后投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类可致电击风险，对动物可能造成中毒。",

    "可回收物/旧衣服": "可回收物再利用能减少纺织业污染，节约能源。旧衣服可捐赠、纤维化再生，回收价值中等。投放前清洗打包，破损的仍可回收。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/易拉罐": "可回收物再生能耗仅为原生产的5%，节约能源效果显著。易拉罐（铝/铁材质）可100%再生，回收价值高。冲洗干净，压扁投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/枕头": "可回收物再利用能减少资源浪费，降低填埋压力。枕头的布料、填充物可再生加工，回收价值较低。投放前拆解分类，破损的可整体投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/毛绒玩具": "可回收物再生能减少塑料和纺织废料污染。毛绒玩具的布料、PP棉可回收，处理方式为分类再生，回收价值较低。去除电池，清洗后投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/洗发水瓶": "可回收物再生能减少塑料垃圾，节约石油资源。洗发水瓶（PET材质）可制成纤维、塑料管材，回收价值中等。清空冲洗，泵头可一同投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/玻璃杯": "可回收物可无限循环利用，再生无环境污染。玻璃杯再生能节约石英砂等原料，回收价值较高。避免破碎，破损的需包裹投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/皮鞋": "可回收物再生能减少皮革和橡胶浪费。皮鞋的皮革、橡胶鞋底可加工为再生制品，回收价值较低。清理鞋内杂物后投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/砧板": "可回收物再生能提升资源利用率，减少木材/塑料消耗。木质砧板可作生物质原料，塑料砧板可熔融再生，回收价值中等。清洗干净后投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/纸板箱": "可回收物再生能节约木材，减少造纸污染。纸板箱可制成新纸箱或纸浆制品，回收价值较高。拆除胶带，折叠压平，潮湿的需晾干。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/调料瓶": "可回收物再生能减少玻璃/塑料垃圾，保护资源。调料瓶（玻璃/塑料材质）可重新利用，回收价值中等。清空冲洗，金属盖可一同投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/酒瓶": "可回收物再生能节约玻璃原料，降低能耗。酒瓶（尤其是无色玻璃）回收价值高，可循环制成新玻璃制品。冲洗干净，去除纸质标签。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/金属食品罐": "可回收物再生能减少矿产开采，节约能源。金属食品罐（铁/铝材质）再生利用率超90%，回收价值高。冲洗干净，压扁投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/锅": "可回收物再生能节约金属资源，减少冶炼污染。铁锅、铝锅等金属锅具可回炉重炼，回收价值较高。无需修复，直接投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/食用油桶": "可回收物再生能减少塑料污染，提升资源循环率。食用油桶（PET/HDPE材质）可制成非食品接触类产品，回收价值中等。彻底清洗后投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    "可回收物/饮料瓶": "可回收物再生能大幅减少塑料污染，节约资源。饮料瓶（PET材质）回收体系成熟，可制成纤维、容器等，回收价值高。清空冲洗，压扁投放。对环境影响：可回收再利用，节约资源，减少污染。对人类和动物影响：若未回收，填埋占用土地，焚烧可能释放有害气体，对环境有轻微危害，对人类和动物的危害较小。",

    # 有害垃圾（大类3）
    "有害垃圾/干电池": "有害垃圾含有毒物质，随意丢弃会污染土壤和水源。干电池含汞、镉等重金属，属于有害垃圾。其具体作用是提供电力，但含有有害物质。可回收性低，因为含有有害物质，需特殊处理。危害性高，含有重金属等有害物质。对环境影响：含剧毒物质，如重金属、化学品，污染环境和健康。对人类和动物影响：对人类可致病或中毒，对动物可致死。",

    "有害垃圾/软膏": "有害垃圾含化学物质，会危害生态环境和人体健康。过期软膏含残留化学成分，属于有害垃圾。其具体作用是治疗皮肤问题，但含有有害成分。可回收性低，因为含有药物成分，需特殊处理。危害性高，含有化学成分。对环境影响：含剧毒物质，如重金属、化学品，污染环境和健康。对人类和动物影响：对人类可致病或中毒，对动物可致死。",

    "有害垃圾/过期药物": "有害垃圾含变质成分，随意丢弃会污染土壤和水源，危害生物。过期药物需专业无害化处理，属于有害垃圾。其具体作用是治疗疾病，但过期后失去药效。可回收性低，因为含有化学成分，需特殊处理。危害性高，含有化学成分。对环境影响：含剧毒物质，如重金属、化学品，污染环境和健康。对人类和动物影响：对人类可致病或中毒，对动物可致死。"
}

# 固定模型和配置文件路径
MODEL_PATH = r"D:\Study\ultralytics-main\garbage_detection\double_label_train7\weights\best.pt"
YAML_PATH = r"D:\Study\ultralytics-main\lajifenlei.yaml"

# 全局变量：模型和标签映射
model = None
big_category_mapping = None
big_category_names = None
small_category_names = None

# 历史记录存储
history_messages = []


def init_model():
    """初始化模型和配置（页面加载时执行）"""
    global model, big_category_mapping, big_category_names, small_category_names
    try:
        if not os.path.exists(MODEL_PATH):
            return f"❌ 模型文件不存在：\n{MODEL_PATH}"
        if not os.path.exists(YAML_PATH):
            return f"❌ 配置文件不存在：\n{YAML_PATH}"

        model = YOLO(MODEL_PATH)
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        big_category_mapping = cfg["big_category_mapping"]
        big_category_names = cfg["big_category_names"]
        small_category_names = cfg["names"]
        return "✅ 模型加载成功，点击「开始检测」启用摄像头检测"
    except Exception as e:
        return f"❌ 初始化失败：{str(e)}"


def get_ai_introduction(big_name, small_name):
    """获取垃圾分类详细介绍（已替换为本地字典）"""
    key = f"{big_name}/{small_name}"
    intro = GARBAGE_INTRODUCTION.get(key, "未找到该垃圾类别介绍")

    # 为保持与原格式兼容，返回两个部分（实际介绍已包含完整信息）
    return f"{intro}/"


def process_frame(frame, is_detecting, last_detected_label, is_mirrored):
    """处理帧函数，新增镜像参数"""
    global model, big_category_mapping, big_category_names, small_category_names

    current_introduction = "暂无检测物体"
    current_label = last_detected_label

    if model is None or frame is None:
        return frame, current_introduction, current_label, ""

    try:
        # 如果需要镜像，先翻转画面
        if is_mirrored:
            frame = cv2.flip(frame, 1)  # 水平翻转

        # 格式转换：RGB（Gradio输入）→ BGR（模型处理）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 模型推理
        results = model(frame_bgr, conf=0.5, imgsz=640)

        # 绘制检测结果
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
        except:
            font = ImageFont.load_default()

        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0].cpu().numpy())
                small_idx = int(box.cls[0].cpu().numpy())
                big_idx = big_category_mapping.get(small_idx, -1)
                small_name = small_category_names.get(small_idx, "未知小类")
                big_name = big_category_names.get(big_idx, "未知大类") if big_idx != -1 else "未知大类"

                # 记录检测到的物体
                label = f"{big_name}/{small_name}"
                detected_objects.append({
                    "big_name": big_name,
                    "small_name": small_name,
                    "label": label,
                    "confidence": conf
                })

                # 绘制边界框
                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

                # 绘制标签背景
                display_label = f"{big_name}/{small_name} {conf:.2f}"
                text_bbox = draw.textbbox((x1, y1 - 25), display_label, font=font)
                draw.rectangle(text_bbox, fill=(0, 255, 0))

                # 绘制标签文字
                draw.text((x1, y1 - 25), display_label, font=font, fill=(0, 0, 0))

        # 如果有检测到的物体，获取第一个物体的AI介绍
        if detected_objects and is_detecting:
            # 按置信度排序，选择最可信的物体
            detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
            best_object = detected_objects[0]
            current_label = best_object["label"]

            # 只有当标签变化时才重新获取介绍
            if current_label != last_detected_label:
                current_introduction = get_ai_introduction(best_object["big_name"], best_object["small_name"])
                # 添加到历史记录
                history_messages.append(f"检测到: {best_object['label']}\n{current_introduction}\n")
            else:
                current_introduction = "正在获取介绍..."  # 保持原有介绍
        else:
            current_introduction = "暂无检测物体" if is_detecting else "检测已暂停"
            current_label = ""

        # 生成历史记录文本
        history_text = "\n".join(history_messages[-10:])  # 只保留最近10条

        return np.array(pil_img), current_introduction, current_label, history_text

    except Exception as e:
        print(f"处理帧时出错: {e}")
        return frame, f"处理错误: {str(e)}", "", ""


def process_image(image, is_mirrored):
    """处理上传的图片，支持镜像设置"""
    if image is None:
        return None, "请上传图片", ""

    try:
        # 转换图像格式
        if isinstance(image, str):
            # 如果是文件路径
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果是numpy数组
            image = np.array(image)

        # 应用镜像
        if is_mirrored:
            image = cv2.flip(image, 1)

        # 处理图像
        processed_image, introduction, _, history_text = process_frame(image, True, "", is_mirrored)

        return processed_image, introduction, history_text
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return image, f"处理图片时出错: {str(e)}", ""


def process_video(video_path, is_mirrored):
    """处理上传的视频，支持镜像设置"""
    if video_path is None:
        return None, "请上传视频", ""

    try:
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "无法打开视频文件", ""

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建临时输出文件
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"processed_{os.path.basename(video_path)}")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        detected_objects = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理每一帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 应用镜像
            if is_mirrored:
                frame_rgb = cv2.flip(frame_rgb, 1)

            processed_frame, introduction, _, _ = process_frame(frame_rgb, True, "", is_mirrored)

            # 转换回BGR并写入
            if processed_frame is not None:
                processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                out.write(processed_frame_bgr)

            frame_count += 1

            # 每10帧更新一次进度
            if frame_count % 1000 == 0:
                yield f"正在处理第 {frame_count} 帧...", ""

        cap.release()
        out.release()

        # 生成最终介绍
        if detected_objects:
            objects_str = ", ".join(detected_objects)
            final_introduction = f"视频中检测到的物体: {objects_str}\n请查看处理后的视频文件。"
        else:
            final_introduction = "视频处理完成，但未检测到特定物体。"

        # 添加到历史记录
        history_messages.append(f"视频处理完成: {os.path.basename(video_path)}\n{final_introduction}\n")
        history_text = "\n".join(history_messages[-10:])

        return output_path, final_introduction, history_text

    except Exception as e:
        print(f"处理视频时出错: {e}")
        return None, f"处理视频时出错: {str(e)}", ""


# 创建界面
with gr.Blocks(title="垃圾分类实时检测系统") as demo:
    # 状态变量
    is_detecting = gr.State(False)
    last_detected_label = gr.State("")
    is_mirrored = gr.State(False)  # 新增镜像状态变量

    gr.Markdown("""
    # 🗑️ 垃圾分类实时检测系统

    **使用说明：**
    1. 等待系统状态显示「模型加载成功」
    2. 点击「开始检测」按钮开启实时检测
    3. 检测结果将直接在画面中显示边界框和分类标签
    4. AI会自动介绍检测到的垃圾类别（包含详细环境影响和健康危害分析）
    5. 可以上传图片或视频进行检测
    """)

    # 系统状态和控制按钮区域
    with gr.Row():
        status = gr.Textbox(
            label="系统状态",
            value="初始化中...",
            interactive=False,
            lines=2,
            scale=4
        )
        with gr.Column(scale=1):
            start_btn = gr.Button("▶️ 开始检测", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ 停止检测", variant="secondary", size="lg")
            mirror_btn = gr.Button("🔄 切换镜像", variant="secondary", size="lg")  # 新增镜像按钮

    # 主要显示区域
    with gr.Row():
        # 摄像头显示和上传区域
        with gr.Column(scale=2):
            with gr.Tab("实时摄像头"):
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="实时检测画面",
                    height=400,
                    type="numpy"
                )

            with gr.Tab("图片上传"):
                image_input = gr.Image(
                    label="上传图片",
                    type="filepath",
                    height=400
                )
                image_output = gr.Image(
                    label="检测结果",
                    height=400
                )
                image_btn = gr.Button("🔍 检测图片", variant="primary")

            with gr.Tab("视频上传"):
                video_input = gr.Video(
                    label="上传视频",
                    height=400
                )
                video_output = gr.Video(
                    label="处理后的视频",
                    height=400
                )
                video_btn = gr.Button("🎬 处理视频", variant="primary")

        # AI介绍区域
        with gr.Column(scale=1):
            ai_introduction = gr.Textbox(
                label="🧠 详细垃圾分类介绍",
                value="等待检测物体...",
                interactive=False,
                lines=12,
                max_lines=15
            )

            ai_history = gr.Textbox(
                label="📜 历史记录",
                value="",
                interactive=False,
                lines=10,
                max_lines=15
            )

            gr.Markdown("""
            **介绍内容说明：**
            - 包含垃圾类别的详细环境影响分析
            - 说明对人类和动物的健康危害
            - 详细描述垃圾的具体作用和可回收性
            - 两部分信息已整合为完整介绍
            """)

    # 页面加载时初始化模型
    demo.load(init_model, inputs=[], outputs=status)


    # 视频流处理函数
    def video_stream(frame, detecting_state, last_label, mirror_state):
        """处理视频流，新增镜像参数"""
        return process_frame(frame, detecting_state, last_label, mirror_state)


    # 绑定视频流处理
    webcam.stream(
        video_stream,
        inputs=[webcam, is_detecting, last_detected_label, is_mirrored],
        outputs=[webcam, ai_introduction, last_detected_label, ai_history],
        show_progress="hidden"
    )


    # 按钮事件处理
    def start_detection():
        """开始检测"""
        return True, "🔴 检测中...实时识别已开启"


    def stop_detection():
        """停止检测"""
        return False, "✅ 检测已暂停，点击「开始检测」重新启用"


    def toggle_mirror(current_state):
        """切换镜像状态"""
        new_state = not current_state
        status_msg = "镜像模式已开启" if new_state else "镜像模式已关闭"
        return new_state, f"ℹ️ {status_msg}"


    # 图片处理
    def process_image_wrapper(image, mirror_state):
        """包装图片处理函数，传递镜像参数"""
        return process_image(image, mirror_state)


    # 视频处理
    def process_video_wrapper(video, mirror_state):
        """包装视频处理函数，传递镜像参数"""
        if video is None:
            return None, "请上传视频", ""
        return process_video(video, mirror_state)


    # 绑定事件
    start_btn.click(
        start_detection,
        inputs=[],
        outputs=[is_detecting, status]
    )

    stop_btn.click(
        stop_detection,
        inputs=[],
        outputs=[is_detecting, status]
    )

    # 绑定镜像切换按钮事件
    mirror_btn.click(
        toggle_mirror,
        inputs=[is_mirrored],
        outputs=[is_mirrored, status]
    )

    image_btn.click(
        process_image_wrapper,
        inputs=[image_input, is_mirrored],
        outputs=[image_output, ai_introduction, ai_history]
    )

    video_btn.click(
        process_video_wrapper,
        inputs=[video_input, is_mirrored],
        outputs=[video_output, ai_introduction, ai_history]
    )

# 启动界面
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )