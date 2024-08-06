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

# 局部
# hr_path = './results_diff_visual_quant_222/out_v17_div2k_patch/v17_1hr_patch1.png'
# # sr_path = './results_quant/out_v2_div2k_patch/v2_1sr_patch1.png'
# sr_path = './results_local/out_local_v17/1_local_SR_patch1.png'

# 整图渐进
# hr_path = './results_diff_visual_quant_222/out_v17_div2k/1hr.png'
# # sr_path = './results_quant/out_v2_div2k/1sr.png'
# sr_path = './results_diff_visual_quant_222/out_v17_div2k/1sr.png'

# 整图传统
# hr_path = './results_quant/out_v2_div2k/1hr.png'
# # sr_path = './results_quant/out_v2_div2k/1sr.png'
# sr_path = './results_quant/out_v12_div2k/1sr.png'

# 局部渐进
hr_path = './results_diff_visual_quant_222/out_v17_div2k_patch/v17_1hr_patch1.png'
# sr_path = './results_quant/out_v2_div2k_patch/v2_1sr_patch1.png'
sr_path = './results_diff_visual_quant_222/out_v4_div2k_patch/v4_1sr_patch1.png'

# 局部传统
# hr_path = './results_diff_visual_quant_222/out_v17_div2k_patch/v17_1hr_patch1.png'
# # sr_path = './results_quant/out_v2_div2k_patch/v2_1sr_patch1.png'
# sr_path = './results_quant/out_v12_div2k_patch/v12_1sr_patch1.png'

# hr_path = './results_diff_visual_quant_222/out_v17_div2k_patch/v17_1hr_patch1.png'
# # sr_path = './results_quant/out_v2_div2k_patch/v2_1sr_patch1.png'
# sr_path = './results_local_b2000/out_local_v17/1_local_SR_patch1.png'

# hr_path = './results_diff_visual_quant_222/out_v17_div2k/1hr.png'
# sr_path = './results_diff_visual_quant_222/out_v17_div2k/1sr.png'

# hr_path = './results_diff_visual_quant_222/out_v17_div2k_patch/v17_1hr_patch1.png'
# # sr_path = './results_quant/out_v2_div2k_patch/v2_1sr_patch1.png'
# sr_path = './results_0817_local_b64/out_local_v17/1_local_SR_patch1.png'

t = time.time()

# 读取图像
img_hr = Image.open(hr_path).convert('RGB')
img_sr = Image.open(sr_path).convert('RGB')

# 转换为 numpy 数组以计算 PSNR 和 SSIM
i1 = np.array(img_hr, dtype=np.float32) / 255.0  # 归一化到0-1
i2 = np.array(img_sr, dtype=np.float32) / 255.0  # 假设img_sr是你的超分辨率图像或插值放大后的图像

# 如果你需要YCbCr色彩空间的Y通道
y_i1 = rgb2ycbcr(i1)[:, :, 0]  # 提取Y通道
y_i2 = rgb2ycbcr(i2)[:, :, 0]  # 提取Y通道

y_psnr = psnr_get(y_i1, y_i2)
y_ssim = ssim_get(y_i1, y_i2, data_range=255)

print('Y PSNR:%.2f' % y_psnr)
print('Y SSIM:%.4f' % y_ssim)
print('time:%.3f s' % (time.time() - t))

str_write = 'psnr:{0:.2f}, ssim:{1:.4f}'.format(y_psnr, y_ssim) + '\n'
fp = open('./results_0817_local_b64/test_log.txt', 'a+')
fp.write(str_write)
fp.close()
