import os
from yolov5.train import run


def main():
    # 数据配置 YAML 文件，路径相对于 yolov5 目录
    data_yaml = os.path.abspath('data/workwear.yaml')
    run(
        imgsz=300,  # 输入的图片大小
        batch=4,  # 批量大小
        epochs=1,  # 训练轮次
        data=data_yaml,  # 数据集配置文件路径
        weights='yolov5s.pt',  # 预训练权重
        project='runs/train',  # 保存结果文件夹
        name='workwear_exp',  # 本次训练实验名称
        exist_ok=True,  # 同名文件夹允许覆盖
        workers=0,
        device=0,
        save=True
    )


if __name__ == '__main__':
    main()

import sys
import torch
# print("当前Python路径:", sys.executable)
# print("PyTorch版本:", torch.__version__)
# print("CUDA可用:", torch.cuda.is_available())
# print("CUDA设备数:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("CUDA设备名:", torch.cuda.get_device_name(0))
# else:
#     print("未检测到CUDA设备")
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # 把yolov5加入路径，这样Python能找到相应代码包
# sys.path.insert(0, os.path.abspath('yolov5'))
