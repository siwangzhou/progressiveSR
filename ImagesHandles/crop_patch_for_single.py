from PIL import Image

# 打开图像
img_name = "1_local_SR"
image_path = "./results_0817_local_b64/out_local_v17/" + img_name + ".png"  # 替换为你的图像路径
out_patch = "./results_0817_local_b64/out_local_v17/" + img_name + "_patch1.png"
image = Image.open(image_path)

# 定义裁剪区域的坐标
# 0817 (1070, 600, 1160, 690)   1sr

# 裁剪图像
cropped_image = image.crop((1070, 600, 1160, 690))

# 保存裁剪后的图像
cropped_image.save(out_patch)

