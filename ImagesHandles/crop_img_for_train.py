import os
from PIL import Image

def split_image(image_path, output_folder, image_name):
    image = Image.open(image_path)
    width, height = image.size

    # 计算每块的尺寸
    piece_width = width // 2
    piece_height = height // 2

    pieces = [
        (0, 0, piece_width, piece_height),
        (piece_width, 0, width, piece_height),
        (0, piece_height, piece_width, height),
        (piece_width, piece_height, width, height),
    ]

    for i, box in enumerate(pieces, start=1):
        piece = image.crop(box)
        piece.save(os.path.join(output_folder, f"{image_name}_{i}.png"))

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_name)
            base_name = os.path.splitext(image_name)[0]
            split_image(image_path, output_folder, base_name)

# 输入文件夹和输出文件夹路径
input_folder = 'C:/DATASETS/DIV2K/DIV2K_train_HR'
output_folder = 'C:/DATASETS/DIV2K_Crop/DIV2K_Crop_train_HR'
# 处理文件夹中的图像
process_folder(input_folder, output_folder)
