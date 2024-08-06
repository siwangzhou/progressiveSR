from ops.OmniSR_visual_quant_222_general import LR_SR_x4_general
import torch
import numpy as np
import math
import copy
from torch.utils.checkpoint import checkpoint
from data_general import MyDataset, Test
import time


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)



if __name__ == '__main__':
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

    chpoint1 = torch.load("./train_log_v2_quant/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1316.pt")
    v2_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint1['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v2_upsample_state_dict = {k.replace('layer3.', ''): v for k, v in chpoint1['model_state_dict'].items() if
                                k.startswith('layer3.')}
    net_G.v2_downsample.load_state_dict(v2_downsample_state_dict)
    net_G.v2_upsample.load_state_dict(v2_upsample_state_dict)

    chpoint2 = torch.load("./train_log_v4_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1370.pt")
    v4_conv1_state_dict = {k.replace('conv1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                           k.startswith('conv1.')}
    v4_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v4_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                                k.startswith('layer2.')}
    net_G.conv1.load_state_dict(v4_conv1_state_dict)
    net_G.v4_downsample.load_state_dict(v4_downsample_state_dict)
    net_G.v4_upsample.load_state_dict(v4_upsample_state_dict)

    chpoint3 = torch.load("./train_log_v7_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1260.pt")
    v7_conv2_state_dict = {k.replace('conv2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                           k.startswith('conv2.')}
    v7_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v7_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                                k.startswith('layer2.')}
    net_G.conv2.load_state_dict(v7_conv2_state_dict)
    net_G.v7_downsample.load_state_dict(v7_downsample_state_dict)
    net_G.v7_upsample.load_state_dict(v7_upsample_state_dict)

    chpoint4 = torch.load("./train_log_v9_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1282.pt")
    v9_conv3_state_dict = {k.replace('conv3.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                           k.startswith('conv3.')}
    v9_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v9_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                                k.startswith('layer2.')}
    net_G.conv3.load_state_dict(v9_conv3_state_dict)
    net_G.v9_downsample.load_state_dict(v9_downsample_state_dict)
    net_G.v9_upsample.load_state_dict(v9_upsample_state_dict)

    chpoint5 = torch.load("./train_log_v11_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1260.pt")
    v11_conv4_state_dict = {k.replace('conv4.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                           k.startswith('conv4.')}
    v11_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v11_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                                 k.startswith('layer2.')}
    net_G.conv4.load_state_dict(v11_conv4_state_dict)
    net_G.v11_downsample.load_state_dict(v11_downsample_state_dict)
    net_G.v11_upsample.load_state_dict(v11_upsample_state_dict)

    chpoint6 = torch.load("./train_log_v13_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1056.pt")
    v13_conv5_state_dict = {k.replace('conv5.', ''): v for k, v in chpoint6['model_state_dict'].items() if
                            k.startswith('conv5.')}
    v13_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint6['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v13_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint6['model_state_dict'].items() if
                                 k.startswith('layer2.')}
    net_G.conv5.load_state_dict(v13_conv5_state_dict)
    net_G.v13_downsample.load_state_dict(v13_downsample_state_dict)
    net_G.v13_upsample.load_state_dict(v13_upsample_state_dict)

    chpoint7 = torch.load("./train_log_v15_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1014.pt")
    v15_conv6_state_dict = {k.replace('conv6.', ''): v for k, v in chpoint7['model_state_dict'].items() if
                            k.startswith('conv6.')}
    v15_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint7['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v15_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint7['model_state_dict'].items() if
                                 k.startswith('layer2.')}
    net_G.conv6.load_state_dict(v15_conv6_state_dict)
    net_G.v15_downsample.load_state_dict(v15_downsample_state_dict)
    net_G.v15_upsample.load_state_dict(v15_upsample_state_dict)

    chpoint8 = torch.load("./train_log_v17_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1340.pt")
    v17_conv7_state_dict = {k.replace('conv7.', ''): v for k, v in chpoint8['model_state_dict'].items() if
                            k.startswith('conv7.')}
    v17_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint8['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v17_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint8['model_state_dict'].items() if
                                 k.startswith('layer2.')}
    net_G.conv7.load_state_dict(v17_conv7_state_dict)
    net_G.v17_downsample.load_state_dict(v17_downsample_state_dict)
    net_G.v17_upsample.load_state_dict(v17_upsample_state_dict)

    for param in net_G.parameters():
        param.requires_grad = False
    for param in net_G.Omni.parameters():
        param.requires_grad = True

    # for name,param in net_G.named_parameters():
    #     print(name, param.requires_grad)

    data_train = MyDataset()
    data_test = Test()
    train_data = torch.utils.data.DataLoader(dataset=data_train, batch_size=6, shuffle=True, num_workers=2,
                                             pin_memory=True)
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0,
                                            pin_memory=True)

    max_iterations_per_epoch = len(train_data) // 4
    print(max_iterations_per_epoch)

    lossmse = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0005, weight_decay=1e-4, betas=[0.9, 0.999])
    # 学习率调度器
    epochs = 400

    for i in range(0, epochs):
        iteration = 0

        psnr_list = []
        psnr_list2 = []
        psnr_list4 = []
        psnr_list7 = []
        psnr_list9 = []
        psnr_list11 = []
        psnr_list13 = []
        psnr_list15 = []
        psnr_list17 = []
        psnr_list_8 = []
        net_list = []
        if i == 150:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.00025, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 200:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.000125, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 250:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 300:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625 / 2, weight_decay=1e-4,
                                            betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 350:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625 / 4, weight_decay=1e-4,
                                            betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])

        sum_loss = 0
        losses = []
        t1 = time.time()
        t2 = time.time()
        save_num = int(max_iterations_per_epoch / 12)

        for batch, (cropimg_list, sourceimg) in enumerate(train_data, start=0):

            if iteration >= max_iterations_per_epoch:
                break

            cropimg_list = [lr.cuda() for lr in cropimg_list]
            sourceimg = sourceimg.cuda()

            optimizer_G.zero_grad()

            lr_img_list, hr_img_list = net_G(sourceimg)

            loss_v2_sr = lossL1(hr_img_list[0], sourceimg)
            loss_v4_sr = lossL1(hr_img_list[1], sourceimg)
            loss_v7_sr = lossL1(hr_img_list[2], sourceimg)
            loss_v9_sr = lossL1(hr_img_list[3], sourceimg)
            loss_v11_sr = lossL1(hr_img_list[4], sourceimg)
            loss_v13_sr = lossL1(hr_img_list[5], sourceimg)
            loss_v15_sr = lossL1(hr_img_list[6], sourceimg)
            loss_v17_sr = lossL1(hr_img_list[7], sourceimg)

            loss2 = (loss_v2_sr + loss_v4_sr + loss_v7_sr + loss_v9_sr + loss_v11_sr + loss_v13_sr + loss_v15_sr + loss_v17_sr ) / 8

            loss_G = loss2
            sum_loss += loss_G

            loss_G.backward()
            optimizer_G.step()

            iteration += 1

            # 判断最高psnr并保存
            if batch % save_num == 0:
                net_G.eval()
                psnr2_sum = 0
                psnr4_sum = 0
                psnr7_sum = 0
                psnr9_sum = 0
                psnr11_sum = 0
                psnr13_sum = 0
                psnr15_sum = 0
                psnr17_sum = 0
                psnr_sum = 0
                a = copy.deepcopy(net_G.state_dict())
                net_list.append(a)
                with torch.no_grad():
                    for test_batch, (lrimg_list, hrimg) in enumerate(test_data, start=0):
                        lrimg_list = [lr.cuda() for lr in lrimg_list]
                        hrimg = hrimg.cuda()

                        lr_img_list, hr_img_list = net_G(hrimg)

                        i1 = hrimg.cpu().detach().numpy()[0]
                        i2 = hr_img_list[0].cpu().detach().numpy()[0]
                        i4 = hr_img_list[1].cpu().detach().numpy()[0]
                        i7 = hr_img_list[2].cpu().detach().numpy()[0]
                        i9 = hr_img_list[3].cpu().detach().numpy()[0]
                        i11 = hr_img_list[4].cpu().detach().numpy()[0]
                        i13 = hr_img_list[5].cpu().detach().numpy()[0]
                        i15 = hr_img_list[6].cpu().detach().numpy()[0]
                        i17 = hr_img_list[7].cpu().detach().numpy()[0]

                        i1 = (i1 + 1.0) / 2.0
                        i1 = np.clip(i1, 0.0, 1.0)
                        i2 = (i2 + 1.0) / 2.0
                        i2 = np.clip(i2, 0.0, 1.0)
                        i4 = (i4 + 1.0) / 2.0
                        i4 = np.clip(i4, 0.0, 1.0)
                        i7 = (i7 + 1.0) / 2.0
                        i7 = np.clip(i7, 0.0, 1.0)
                        i9 = (i9 + 1.0) / 2.0
                        i9 = np.clip(i9, 0.0, 1.0)
                        i11 = (i11 + 1.0) / 2.0
                        i11 = np.clip(i11, 0.0, 1.0)
                        i13 = (i13 + 1.0) / 2.0
                        i13 = np.clip(i13, 0.0, 1.0)
                        i15 = (i15 + 1.0) / 2.0
                        i15 = np.clip(i15, 0.0, 1.0)
                        i17 = (i17 + 1.0) / 2.0
                        i17 = np.clip(i17, 0.0, 1.0)

                        i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
                        i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
                        i4 = 65.481 * i4[0, :, :] + 128.553 * i4[1, :, :] + 24.966 * i4[2, :, :] + 16
                        i7 = 65.481 * i7[0, :, :] + 128.553 * i7[1, :, :] + 24.966 * i7[2, :, :] + 16
                        i9 = 65.481 * i9[0, :, :] + 128.553 * i9[1, :, :] + 24.966 * i9[2, :, :] + 16
                        i11 = 65.481 * i11[0, :, :] + 128.553 * i11[1, :, :] + 24.966 * i11[2, :, :] + 16
                        i13 = 65.481 * i13[0, :, :] + 128.553 * i13[1, :, :] + 24.966 * i13[2, :, :] + 16
                        i15 = 65.481 * i15[0, :, :] + 128.553 * i15[1, :, :] + 24.966 * i15[2, :, :] + 16
                        i17 = 65.481 * i17[0, :, :] + 128.553 * i17[1, :, :] + 24.966 * i17[2, :, :] + 16

                        psnr2 = psnr_get(i1, i2)
                        psnr4 = psnr_get(i1, i4)
                        psnr7 = psnr_get(i1, i7)
                        psnr9 = psnr_get(i1, i9)
                        psnr11 = psnr_get(i1, i11)
                        psnr13 = psnr_get(i1, i13)
                        psnr15 = psnr_get(i1, i15)
                        psnr17 = psnr_get(i1, i17)

                        psnr2_sum += psnr2
                        psnr4_sum += psnr4
                        psnr7_sum += psnr7
                        psnr9_sum += psnr9
                        psnr11_sum += psnr11
                        psnr13_sum += psnr13
                        psnr15_sum += psnr15
                        psnr17_sum += psnr17

                        psnr_sum += (psnr2 + psnr4 + psnr7 +psnr9 + psnr11 + psnr13 + psnr15 +psnr17 ) / 8
                psnr_sum = psnr_sum / (test_batch + 1)
                psnr2_sum = psnr2_sum / (test_batch + 1)
                psnr4_sum = psnr4_sum / (test_batch + 1)
                psnr7_sum = psnr7_sum / (test_batch + 1)
                psnr9_sum = psnr9_sum / (test_batch + 1)
                psnr11_sum = psnr11_sum / (test_batch + 1)
                psnr13_sum = psnr13_sum / (test_batch + 1)
                psnr15_sum = psnr15_sum / (test_batch + 1)
                psnr17_sum = psnr17_sum / (test_batch + 1)

                psnr_list.append(psnr_sum)
                psnr_list2.append(psnr2_sum)
                psnr_list4.append(psnr4_sum)
                psnr_list7.append(psnr7_sum)
                psnr_list9.append(psnr9_sum)
                psnr_list11.append(psnr11_sum)
                psnr_list13.append(psnr13_sum)
                psnr_list15.append(psnr15_sum)
                psnr_list17.append(psnr17_sum)

                net_G.train()

        psnr_max = max(psnr_list)
        index = psnr_list.index(psnr_max)
        psnr_list_8 = [psnr_list2[index], psnr_list4[index], psnr_list7[index], psnr_list9[index], psnr_list11[index], psnr_list13[index], psnr_list15[index], psnr_list17[index]]
        psnr_list_8_str = ', '.join([f'{psnr:.3f}' for psnr in psnr_list_8])
        if (i + 1) % 1 == 0:
            checkpoint = {
                'epoch': i+1,
                'model_state_dict': net_list[index],
                'optimizer_state_dict': optimizer_G.state_dict(),
            }
            torch.save(checkpoint, './train_log_general_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_{0}.pt'.format(i + 1))
            del net_list
        print('{0}|{1}    avg_loss={2:.6f}    time={3:.3f}min   psnr_max:{4:.3f}   psnr_list:{5}'.format(i + 1, epochs, sum_loss / batch, (time.time() - t1) / 60, psnr_max, psnr_list_8_str))
        str_list = [str(item) for item in psnr_list]
        # 使用join方法将新列表中的元素连接成一个字符串，元素之间由空格分隔
        str_list = ' '.join(str_list)
        str_write = '{0}|{1}    avg_loss={2:.6f}    time={3:.3f}min   psnr_max:{4:.3f}   psnr_list:{5}'.format(i + 1, epochs, sum_loss / batch, (time.time() - t1) / 60, psnr_max, psnr_list_8_str) + '\n'
        fp = open('./train_log_general_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4.txt', 'a+')
        fp.write(str_write)
        fp.close()
