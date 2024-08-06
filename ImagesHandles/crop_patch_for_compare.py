from PIL import Image

# 打开图像
for i in range(1, 10):
    img_name = str(i)+"sr"
    image_path = "./results_diff_visual_quant_222/out_v17_div2k/" + img_name + ".png"  # 替换为你的图像路径
    out_patch = "./results_diff_visual_quant_222/out_v17_div2k/" + img_name + "_patch1.png"
    image = Image.open(image_path)

    # 定义裁剪区域的坐标
    # 0817 (1070, 600, 1170, 690)   1sr
    # 0831 (335, 779, 425, 860)     2sr patch1
    # 0831 (536, 920, 616, 992)     2sr patch2
    # 0873 (360, 330, 460, 420)     3sr
    # 0891 (90, 1210, 230, 1265)    4sr
    # 0867 (825, 1957, 1028, 2012)  5sr
    # 0871 (1308, 1035, 1378, 1098) 6sr
    # 0837 (830, 530, 1025, 610)    7sr
    # 0832 (1558, 580, 1658, 670)   8sr
    # 0825 (1038, 918, 1098, 972)   9sr
    p = []
    if img_name in ["1sr", "1hr"]:
        p = [1070, 600, 1170, 690]
    elif img_name in ["2sr", "2hr"]:
        p = [335, 779, 425, 860]
    elif img_name in ["3sr", "3hr"]:
        p = [360, 330, 460, 420]
    elif img_name in ["4sr", "4hr"]:
        p = [90, 1210, 230, 1265]
    elif img_name in ["5sr", "5hr"]:
        p = [825, 1957, 1028, 2012]
    elif img_name in ["6sr", "6hr"]:
        p = [1308, 1035, 1378, 1098]
    elif img_name in ["7sr", "7hr"]:
        p = [830, 530, 1025, 610]
    elif img_name in ["8sr", "8hr"]:
        p = [1558, 580, 1658, 670]
    elif img_name in ["9sr", "9hr"]:
        p = [1038, 918, 1098, 972]

    # 裁剪图像
    cropped_image = image.crop((p[0], p[1], p[2], p[3]))

    # 保存裁剪后的图像
    cropped_image.save(out_patch)

