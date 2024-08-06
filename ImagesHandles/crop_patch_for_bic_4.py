from PIL import Image
import os

input_path = "./results_v2_bicubic/out_v2_div2k/"
output_path = "./results_v2_bicubic/out_v2_div2k_patch/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 打开图像
for i in range(1, 10):
    img_name = str(i)+"sr_bic"
    image_path = input_path + img_name + ".png"  # 替换为你的图像路径
    out_patch = output_path + "bic_" + img_name + "_patch1.png"
    image = Image.open(image_path)

    # 定义裁剪区域的坐标
    # 0817 (1070, 600, 1160, 690)   1sr
    # 0831 (300, 775, 390, 865)     2sr
    # 0873 (360, 330, 450, 420)     3sr
    # 0825 (1023, 905, 1113, 995)   9sr
    p = []
    if img_name in ["1sr_bic", "1hr"]:
        p = [1070, 600, 1160, 690]
    elif img_name in ["2sr_bic", "2hr"]:
        p = [300, 775, 390, 865]
    elif img_name in ["3sr_bic", "3hr"]:
        p = [360, 330, 450, 420]
    elif img_name in ["9sr_bic", "9hr"]:
        p = [1023, 905, 1113, 995]

    # 裁剪图像
    if len(p)!=0:
        cropped_image = image.crop((p[0], p[1], p[2], p[3]))
        # 保存裁剪后的图像
        cropped_image.save(out_patch)
