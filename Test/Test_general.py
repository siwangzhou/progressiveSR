import sys
from utils import *
import copy
import torchvision
import time
import sys
from torch.utils.checkpoint import checkpoint
import torch
from ops.OmniSR_visual_quant_222_general import LR_SR_x4_general
import numpy as np
from skimage.metrics import structural_similarity as ssim_get
from data_general import Test
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


net_G = LR_SR_x4_general(kwards=kwards).cuda()
chpoint= torch.load("train_log_general_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_293.pt")
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
y_psnr_sum_v2 = 0
y_ssim_sum_v2 = 0
y_psnr_sum_v4 = 0
y_ssim_sum_v4 = 0
y_psnr_sum_v7 = 0
y_ssim_sum_v7 = 0
y_psnr_sum_v9 = 0
y_ssim_sum_v9 = 0
y_psnr_sum_v11 = 0
y_ssim_sum_v11 = 0
y_psnr_sum_v13 = 0
y_ssim_sum_v13 = 0
y_psnr_sum_v15 = 0
y_ssim_sum_v15 = 0
y_psnr_sum_v17 = 0
y_ssim_sum_v17 = 0

count = 0
t = time.time()
index = 0

for test_batch, (_, hrimg) in enumerate(test_data, start=0):
    index = index + 1
    hrimg = hrimg.cuda()

    lr_img_list, hr_img_list = net_G(hrimg)

    i1 = hrimg / 2 + 0.5
    i2 = hr_img_list[0] / 2 + 0.5
    i4 = hr_img_list[1] / 2 + 0.5
    i7 = hr_img_list[2] / 2 + 0.5
    i9 = hr_img_list[3] / 2 + 0.5
    i11 = hr_img_list[4] / 2 + 0.5
    i13 = hr_img_list[5] / 2 + 0.5
    i15 = hr_img_list[6] / 2 + 0.5
    i17 = hr_img_list[7] / 2 + 0.5

    i1 = i1.clamp(0, 1)
    i2 = i2.clamp(0, 1)
    i4 = i4.clamp(0, 1)
    i7 = i7.clamp(0, 1)
    i9 = i9.clamp(0, 1)
    i11 = i11.clamp(0, 1)
    i13 = i13.clamp(0, 1)
    i15 = i15.clamp(0, 1)
    i17 = i17.clamp(0, 1)


    if flag_visual:
        save_image(i1, './out_general/' + str(index) + '_hr.png')
        save_image(i2, './out_general/' + str(index) + '_sr_v2.png')
        save_image(i4, './out_general/' + str(index) + '_sr_v4.png')
        save_image(i7, './out_general/' + str(index) + '_sr_v7.png')
        save_image(i9, './out_general/' + str(index) + '_sr_v9.png')
        save_image(i11, './out_general/' + str(index) + '_sr_v11.png')
        save_image(i13, './out_general/' + str(index) + '_sr_v13.png')
        save_image(i15, './out_general/' + str(index) + '_sr_v15.png')
        save_image(i17, './out_general/' + str(index) + '_sr_v17.png')

    i1 = i1.cpu().detach().numpy()[0]
    i2 = i2.cpu().detach().numpy()[0]
    i4 = i4.cpu().detach().numpy()[0]
    i7 = i7.cpu().detach().numpy()[0]
    i9 = i9.cpu().detach().numpy()[0]
    i11 = i11.cpu().detach().numpy()[0]
    i13 = i13.cpu().detach().numpy()[0]
    i15 = i15.cpu().detach().numpy()[0]
    i17 = i17.cpu().detach().numpy()[0]

    y_i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
    y_i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
    y_i4 = 65.481 * i4[0, :, :] + 128.553 * i4[1, :, :] + 24.966 * i4[2, :, :] + 16
    y_i7 = 65.481 * i7[0, :, :] + 128.553 * i7[1, :, :] + 24.966 * i7[2, :, :] + 16
    y_i9 = 65.481 * i9[0, :, :] + 128.553 * i9[1, :, :] + 24.966 * i9[2, :, :] + 16
    y_i11 = 65.481 * i11[0, :, :] + 128.553 * i11[1, :, :] + 24.966 * i11[2, :, :] + 16
    y_i13 = 65.481 * i13[0, :, :] + 128.553 * i13[1, :, :] + 24.966 * i13[2, :, :] + 16
    y_i15 = 65.481 * i15[0, :, :] + 128.553 * i15[1, :, :] + 24.966 * i15[2, :, :] + 16
    y_i17 = 65.481 * i17[0, :, :] + 128.553 * i17[1, :, :] + 24.966 * i17[2, :, :] + 16

    y_psnr_v2 = psnr_get(y_i1, y_i2)
    y_ssim_v2 = ssim_get(y_i1, y_i2, data_range=255)
    y_psnr_v4 = psnr_get(y_i1, y_i4)
    y_ssim_v4 = ssim_get(y_i1, y_i4, data_range=255)
    y_psnr_v7 = psnr_get(y_i1, y_i7)
    y_ssim_v7 = ssim_get(y_i1, y_i7, data_range=255)
    y_psnr_v9 = psnr_get(y_i1, y_i9)
    y_ssim_v9 = ssim_get(y_i1, y_i9, data_range=255)
    y_psnr_v11 = psnr_get(y_i1, y_i11)
    y_ssim_v11 = ssim_get(y_i1, y_i11, data_range=255)
    y_psnr_v13 = psnr_get(y_i1, y_i13)
    y_ssim_v13 = ssim_get(y_i1, y_i13, data_range=255)
    y_psnr_v15 = psnr_get(y_i1, y_i15)
    y_ssim_v15 = ssim_get(y_i1, y_i15, data_range=255)
    y_psnr_v17 = psnr_get(y_i1, y_i17)
    y_ssim_v17 = ssim_get(y_i1, y_i17, data_range=255)

    y_psnr_sum_v2 += y_psnr_v2
    y_ssim_sum_v2 += y_ssim_v2
    y_psnr_sum_v4 += y_psnr_v4
    y_ssim_sum_v4 += y_ssim_v4
    y_psnr_sum_v7 += y_psnr_v7
    y_ssim_sum_v7 += y_ssim_v7
    y_psnr_sum_v9 += y_psnr_v9
    y_ssim_sum_v9 += y_ssim_v9
    y_psnr_sum_v11 += y_psnr_v11
    y_ssim_sum_v11 += y_ssim_v11
    y_psnr_sum_v13 += y_psnr_v13
    y_ssim_sum_v13 += y_ssim_v13
    y_psnr_sum_v15 += y_psnr_v15
    y_ssim_sum_v15 += y_ssim_v15
    y_psnr_sum_v17 += y_psnr_v17
    y_ssim_sum_v17 += y_ssim_v17

    count += 1

psnr_list_8 = [y_psnr_sum_v2/count, y_psnr_sum_v4/count, y_psnr_sum_v7/count, y_psnr_sum_v9/count, y_psnr_sum_v11/count, y_psnr_sum_v13/count, y_psnr_sum_v15/count, y_psnr_sum_v17/count]
psnr_list_8_str = ', '.join([f'{psnr:.2f}' for psnr in psnr_list_8])

ssim_list_8 = [y_ssim_sum_v2/count, y_ssim_sum_v4/count, y_ssim_sum_v7/count, y_ssim_sum_v9/count, y_ssim_sum_v11/count, y_ssim_sum_v13/count, y_ssim_sum_v15/count, y_ssim_sum_v17/count]
ssim_list_8_str = ', '.join([f'{ssim:.4f}' for ssim in ssim_list_8])

print("avg Y PSNR:", psnr_list_8_str)
print("avg Y SSIM:", ssim_list_8_str)
print('time:%.3f s' % (time.time() - t))

if flag_metric:
    str_write = 'test epoch:{0}, psnr_list:{1}, ssim_list:{2}'.format(test_epoch, psnr_list_8_str, ssim_list_8_str) + '\n'
    fp = open('./out_general/test_log.txt', 'a+')
    fp.write(str_write)
    fp.close()
