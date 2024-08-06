import sys
from utils import *
import copy
import torchvision
import time
import sys
from torch.utils.checkpoint import checkpoint
import torch
from ops.OmniSR_visual_quant_222 import LR_SR_x4_v13_quant
import numpy as np
from skimage.metrics import structural_similarity as ssim_get
from data_v12v13 import Test
from torchvision.utils import save_image


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


kwards = {'upsampling': 1,
              'res_num': 5,
              'block_num': 1,
              'bias': True,
              'block_script_name': 'OSA',
              'block_class_name': 'OSA_Block',
              'window_size': 8,
              'pe': True,
              'ffn_bias': True}


net_G = LR_SR_x4_v13_quant(kwards=kwards).cuda()
chpoint= torch.load("train_log_v13_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1056.pt")
net_G.load_state_dict(chpoint['model_state_dict'])
test_epoch = chpoint['epoch']
net_G.eval()
for param in net_G.parameters():
    param.requires_grad = False

# ["E:/DATASETS/image_resolution/Set5/HR_crop32"]
# ['E:/DATASETS/image_resolution/DIV2K/DIV2K_valid_HR_crop32']
# ['E:/DATASETS/image_resolution/DIV2K/DIV2K_valid_HR_crop32_visual']

# ["E:/DATASETS/image_resolution/Set14/HR_crop32"]
# ["E:/DATASETS/image_resolution/BSDS100/HR_crop32"]
# ["E:/DATASETS/image_resolution/Urban100/HR_crop32"]

data_test = Test(["E:/DATASETS/image_resolution/Urban100/HR_crop32"])
test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0,
                                        pin_memory=True)
flag_metric = True
flag_visual = False

# Set5
y_psnr_sum = 0
y_ssim_sum = 0
count = 0
t = time.time()
index = 0

for test_batch, (_, hrimg) in enumerate(test_data, start=0):
    index = index + 1
    hrimg = hrimg.cuda()

    lr_img, hr_img = net_G(hrimg)

    i1 = hrimg / 2 + 0.5
    i2 = hr_img / 2 + 0.5
    i3 = lr_img / 2 + 0.5

    i1 = i1.clamp(0, 1)
    i2 = i2.clamp(0, 1)
    i3 = i3.clamp(0, 1)

    if flag_visual:
        save_image(i1, './results_diff_visual_quant_222/out_v13_div2k/' + str(index) + 'hr.png')
        save_image(i2, './results_diff_visual_quant_222/out_v13_div2k/' + str(index) + 'sr.png')
        save_image(i3, './results_diff_visual_quant_222/out_v13_div2k/' + str(index) + 'lr.png')

    i1 = i1.cpu().detach().numpy()[0]
    i2 = i2.cpu().detach().numpy()[0]
    i3 = i3.cpu().detach().numpy()[0]

    y_i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
    y_i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16

    y_psnr = psnr_get(y_i1, y_i2)
    y_ssim = ssim_get(y_i1, y_i2, data_range=255)

    y_psnr_sum += y_psnr
    y_ssim_sum += y_ssim

    count += 1

print('avg Y PSNR:%.3f' % (y_psnr_sum / count))
print('avg Y SSIM:%.3f' % (y_ssim_sum / count))
print('time:%.3f s' % (time.time() - t))

if flag_metric:
    str_write = 'test_epoch:{0}    avg_Y_PSRN:{1:.2f}    avg_Y_SSIM:{2:.4f}    time={3:.3f} s'.format(test_epoch, (
                y_psnr_sum / count), (y_ssim_sum / count), (time.time() - t)) + '\n'
    fp = open('out_v13/test_log.txt', 'a+')
    fp.write(str_write)
    fp.close()
