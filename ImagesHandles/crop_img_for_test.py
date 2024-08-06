import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 定义输入输出文件夹路径
input_folder = "E:/DATASETS/image_resolution/REDS_test/HR"
output_folder = "E:/DATASETS/image_resolution/REDS_test/HR_crop32"
os.makedirs(output_folder, exist_ok=True)

# 定义裁剪比例
scales = 32

# 遍历输入文件夹中的所有图像文件
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path).convert('RGB')

    # 计算新的尺寸，确保宽度和高度都是32的倍数
    w, h = np.array(img.size) // scales * scales

    # 随机裁剪图像
    cropped_img = transforms.RandomCrop((h, w))(img)

    # 保存裁剪后的图像到输出文件夹
    output_path = os.path.join(output_folder, img_name)
    cropped_img.save(output_path)

print("所有图像已裁剪并保存到新的文件夹。")
