import torch

# 模型路径
weights = 'runs/train/workwear_exp/weights/best.pt'
img_path = r"D:\Desktop\3953.jpg_wh860.jpg"

# 加载模型
model = torch.hub.load('yolov5', 'custom', path=weights, source='local')

# 推理图片
results = model(img_path)

# 打印检测结果
results.print()

# 显示检测图像
results.show()
