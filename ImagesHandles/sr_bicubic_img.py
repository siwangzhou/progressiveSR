import os
import time
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from skimage.color import rgb2ycbcr
import math
from skimage.metrics import structural_similarity as ssim_get


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

photo_path = './results_v2_bicubic/out_v2_div2k'
save_path = photo_path  # 假设这是保存放大图像的路径

flag_metric = False
flag_visual = True

test_list = [f for f in os.listdir(photo_path) if f.endswith(('lr.png', 'hr.png'))]

# 分离 lr 和 hr 图像的文件名
lr_imgs = [f for f in test_list if f.endswith('lr.png')]
hr_imgs = [f.replace('lr.png', 'hr.png') for f in lr_imgs if f.replace('lr.png', 'hr.png') in test_list]

# 确保 hr 图像存在
hr_imgs = [img for img in hr_imgs if os.path.exists(os.path.join(photo_path, img))]

y_psnr_sum = 0
y_ssim_sum = 0
count = 0
t = time.time()

for lr_name, hr_name in zip(lr_imgs, hr_imgs):
    lr_path = os.path.join(photo_path, lr_name)
    hr_path = os.path.join(photo_path, hr_name)

    # 读取图像
    img_lr = Image.open(lr_path).convert('RGB')
    img_hr = Image.open(hr_path).convert('RGB')

    # 双线性插值放大 lr 图像
    # img_lr4 = img_lr.resize((img_hr.size[0], img_hr.size[1]), Image.BICUBIC)
    h, w = img_lr.size
    img_sr = img_lr.resize((int(h * 4), int(w * 4)), Image.BICUBIC)

    # 保存放大后的图像
    if flag_visual:
        save_name = lr_name.replace('lr.png', 'sr_bic.png')
        save_path_full = os.path.join(save_path, save_name)
        img_sr.save(save_path_full)

    # 转换为 numpy 数组以计算 PSNR 和 SSIM
    i1 = np.array(img_hr, dtype=np.float32) / 255.0  # 归一化到0-1
    i2 = np.array(img_sr, dtype=np.float32) / 255.0  # 假设img_sr是你的超分辨率图像或插值放大后的图像

    # 如果你需要YCbCr色彩空间的Y通道
    y_i1 = rgb2ycbcr(i1)[:, :, 0]  # 提取Y通道
    y_i2 = rgb2ycbcr(i2)[:, :, 0]  # 提取Y通道

    y_psnr = psnr_get(y_i1, y_i2)
    y_ssim = ssim_get(y_i1, y_i2, data_range=255)

    y_psnr_sum += y_psnr
    y_ssim_sum += y_ssim

    count += 1
print('avg Y PSNR:%.3f' % (y_psnr_sum / count))
print('avg Y SSIM:%.3f' % (y_ssim_sum / count))
print('time:%.3f s' % (time.time() - t))

if flag_metric:
    str_write = 'bicubic result: avg_Y_PSRN:{0:.2f}    avg_Y_SSIM:{1:.4f}    time={2:.3f} s'.format((
                y_psnr_sum / count), (y_ssim_sum / count), (time.time() - t)) + '\n'
    fp = open(photo_path+'/test_log.txt', 'a+')
    fp.write(str_write)
    fp.close()
