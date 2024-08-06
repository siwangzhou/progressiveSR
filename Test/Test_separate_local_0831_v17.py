import sys
from utils import *
import copy
import torchvision
import time
import sys
from torch.utils.checkpoint import checkpoint
import torch
from ops.OmniSR_visual_quant_222 import *
from ops.OmniSR_visual_quant_222_general import LR_SR_x4_general
import numpy as np
from data_v1v2 import Test
from torchvision.utils import save_image
from einops import rearrange


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
chpoint= torch.load("../train_log_general_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_293.pt")
net_G.load_state_dict(chpoint['model_state_dict'])
net_G.eval()
for param in net_G.parameters():
    param.requires_grad = False

net_v2 = LR_SR_x4_v2_quant(kwards=kwards).cuda()
chpoint_v2= torch.load("../train_log_v2_quant/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1316.pt")
net_v2.load_state_dict(chpoint_v2['model_state_dict'])
net_v2.eval()
for param in net_v2.parameters():
    param.requires_grad = False

net_v17 = LR_SR_x4_v17_quant(kwards=kwards).cuda()
chpoint_v17= torch.load("../train_log_v17_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1340.pt")
net_v17.load_state_dict(chpoint_v17['model_state_dict'])
net_v17.eval()
for param in net_v17.parameters():
    param.requires_grad = False

data_test = Test(["../local_img_0831"])
test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Set5
t = time.time()
index = 0

for test_batch, (_, hrimg) in enumerate(test_data, start=0):
    index = index + 1
    hrimg = hrimg.cuda()

    v2_LR = net_G.v2_downsample(hrimg)
    v2_LR_quant = net_G.v2_quant(v2_LR)
    v2_SR_temp = net_G.v2_upsample(v2_LR_quant)

    v2_LR_feat = net_G.conv1(v2_LR_quant)
    v2_local_feat = v2_LR_feat[:, :, 24:27, 16:20]
    v4_LR_diff = net_G.v4_downsample(hrimg)
    v4_LR_diff_quant = net_G.v4_quant(v4_LR_diff)
    v4_local_diff_feat = v4_LR_diff_quant[:, :, 24:27, 16:20]
    v4_local_cat = torch.cat((v2_local_feat, v4_local_diff_feat), dim=1)
    v4_local_LR = rearrange(v4_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)

    v4_local_feat = net_G.conv2(v4_local_LR)
    v7_LR_diff = net_G.v7_downsample(hrimg)
    v7_LR_diff_quant = net_G.v7_quant(v7_LR_diff)
    v7_local_diff_feat = v7_LR_diff_quant[:, :, 24:27, 16:20]
    v7_local_cat = torch.cat((v4_local_feat, v7_local_diff_feat), dim=1)
    v7_local_LR = rearrange(v7_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)

    v7_local_feat = net_G.conv3(v7_local_LR)
    v9_LR_diff = net_G.v9_downsample(hrimg)
    v9_LR_diff_quant = net_G.v9_quant(v9_LR_diff)
    v9_local_diff_feat = v9_LR_diff_quant[:, :, 24:27, 16:20]
    v9_local_cat = torch.cat((v7_local_feat, v9_local_diff_feat), dim=1)
    v9_local_LR = rearrange(v9_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)

    v9_local_feat = net_G.conv4(v9_local_LR)
    v11_LR_diff = net_G.v11_downsample(hrimg)
    v11_LR_diff_quant = net_G.v11_quant(v11_LR_diff)
    v11_local_diff_feat = v11_LR_diff_quant[:, :, 24:27, 16:20]
    v11_local_cat = torch.cat((v9_local_feat, v11_local_diff_feat), dim=1)
    v11_local_LR = rearrange(v11_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)

    v11_local_feat = net_G.conv5(v11_local_LR)
    v13_LR_diff = net_G.v13_downsample(hrimg)
    v13_LR_diff_quant = net_G.v13_quant(v13_LR_diff)
    v13_local_diff_feat = v13_LR_diff_quant[:, :, 24:27, 16:20]
    v13_local_cat = torch.cat((v11_local_feat, v13_local_diff_feat), dim=1)
    v13_local_LR = rearrange(v13_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=13, a2=13, c=3)

    v13_local_feat = net_G.conv6(v13_local_LR)
    v15_LR_diff = net_G.v15_downsample(hrimg)
    v15_LR_diff_quant = net_G.v15_quant(v15_LR_diff)
    v15_local_diff_feat = v15_LR_diff_quant[:, :, 24:27, 16:20]
    v15_local_cat = torch.cat((v13_local_feat, v15_local_diff_feat), dim=1)
    v15_local_LR = rearrange(v15_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=14, a2=14, c=3)

    v15_local_feat = net_G.conv7(v15_local_LR)
    v17_LR_diff = net_G.v17_downsample(hrimg)
    v17_LR_diff_quant = net_G.v17_quant(v17_LR_diff)
    v17_local_diff_feat = v17_LR_diff_quant[:, :, 24:27, 16:20]
    v17_local_cat = torch.cat((v15_local_feat, v17_local_diff_feat), dim=1)
    v17_local_LR = rearrange(v17_local_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=15, a2=15, c=3)

    v17_local_SR_temp = net_G.v17_upsample(v17_local_LR)
    v17_local_SR_temp_feat = rearrange(v17_local_SR_temp, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=32, a2=32, c=3)

    v2_SR = net_v2.layer4(v2_SR_temp)
    local_SR_temp = net_v17.layer3(v17_local_SR_temp)
    v2_SR_feat = rearrange(v2_SR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=32, a2=32, c=3)
    local_SR_temp_feat = rearrange(local_SR_temp, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=32, a2=32, c=3)
    v2_SR_feat[:, :, 24:27, 16:20] = local_SR_temp_feat
    local_SR = rearrange(v2_SR_feat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

    i1 = hrimg / 2 + 0.5
    i2 = local_SR / 2 + 0.5

    i1 = i1.clamp(0, 1)
    i2 = i2.clamp(0, 1)

    save_image(i1, '../out_local_v17/' + str(index) + '_hr.png')
    save_image(i2, '../out_local_v17/' + str(index) + '_local_SR.png')
