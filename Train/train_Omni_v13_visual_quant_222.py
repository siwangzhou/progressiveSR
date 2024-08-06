from ops.OmniSR_visual_quant_222 import LR_SR_x4_v13_quant
import torch
import numpy as np
import math
import copy
from torch.utils.checkpoint import checkpoint
from data_v12v13 import MyDataset, Test
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

    net_G = LR_SR_x4_v13_quant(kwards=kwards).cuda()

    chpoint = torch.load("./train_log_v2_quant/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1316.pt")
    v2_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint['model_state_dict'].items() if
                                k.startswith('layer1.')}
    net_G.v2_downsample.load_state_dict(v2_downsample_state_dict)

    chpoint2 = torch.load("./train_log_v4_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1370.pt")
    v4_conv1_state_dict = {k.replace('conv1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                           k.startswith('conv1.')}
    v4_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                                k.startswith('layer1.')}
    net_G.conv1.load_state_dict(v4_conv1_state_dict)
    net_G.v4_downsample.load_state_dict(v4_downsample_state_dict)

    chpoint3 = torch.load("./train_log_v7_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1260.pt")
    v7_conv2_state_dict = {k.replace('conv2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                                k.startswith('conv2.')}
    v7_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                                k.startswith('layer1.')}
    net_G.conv2.load_state_dict(v7_conv2_state_dict)
    net_G.v7_downsample.load_state_dict(v7_downsample_state_dict)

    chpoint4 = torch.load("./train_log_v9_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1282.pt")
    v9_conv3_state_dict = {k.replace('conv3.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                           k.startswith('conv3.')}
    v9_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                                k.startswith('layer1.')}
    net_G.conv3.load_state_dict(v9_conv3_state_dict)
    net_G.v9_downsample.load_state_dict(v9_downsample_state_dict)

    chpoint5 = torch.load("./train_log_v11_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_1260.pt")
    v11_conv4_state_dict = {k.replace('conv4.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                           k.startswith('conv4.')}
    v11_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                                k.startswith('layer1.')}
    net_G.conv4.load_state_dict(v11_conv4_state_dict)
    net_G.v11_downsample.load_state_dict(v11_downsample_state_dict)

    for param in net_G.v2_downsample.parameters():
        param.requires_grad = False
    for param in net_G.conv1.parameters():
        param.requires_grad = False
    for param in net_G.v4_downsample.parameters():
        param.requires_grad = False
    for param in net_G.conv2.parameters():
        param.requires_grad = False
    for param in net_G.v7_downsample.parameters():
        param.requires_grad = False
    for param in net_G.conv3.parameters():
        param.requires_grad = False
    for param in net_G.v9_downsample.parameters():
        param.requires_grad = False
    for param in net_G.conv4.parameters():
        param.requires_grad = False
    for param in net_G.v11_downsample.parameters():
        param.requires_grad = False

    # for name,param in net_G.named_parameters():
    #     print(name,param.requires_grad)

    data_train = MyDataset()
    data_test = Test()
    train_data = torch.utils.data.DataLoader(dataset=data_train, batch_size=16, shuffle=True, num_workers=2,
                                             pin_memory=True)
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0,
                                            pin_memory=True)

    max_iterations_per_epoch = len(train_data) // 4
    print(max_iterations_per_epoch)

    lossmse = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0005, weight_decay=1e-4, betas=[0.9, 0.999])
    # 学习率调度器
    epochs = 1500

    for i in range(0, epochs):
        iteration = 0

        psnr_list = []
        net_list = []
        if i == 250:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.00025, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 500:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.000125, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 750:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625, weight_decay=1e-4, betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 1000:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625 / 2, weight_decay=1e-4,
                                            betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])
        elif i == 1250:
            optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0000625 / 4, weight_decay=1e-4,
                                            betas=[0.9, 0.999])
            for param_group in optimizer_G.param_groups:
                print(param_group["lr"])

        sum_loss = 0
        losses = []
        t1 = time.time()
        t2 = time.time()
        save_num = int(max_iterations_per_epoch / 12)

        for batch, (cropimg, sourceimg) in enumerate(train_data, start=0):

            if iteration >= max_iterations_per_epoch:
                break

            cropimg = cropimg.cuda()
            sourceimg = sourceimg.cuda()

            optimizer_G.zero_grad()

            lr_img, hr_img = net_G(sourceimg)

            loss1 = lossmse(lr_img, cropimg)
            loss2 = lossL1(hr_img, sourceimg)

            loss_G = loss1 + loss2
            sum_loss += loss_G

            loss_G.backward()
            optimizer_G.step()

            iteration += 1

            # 判断最高psnr并保存
            if batch % save_num == 0 or cropimg is None:
                net_G.eval()
                psnr1_sum = 0
                a = copy.deepcopy(net_G.state_dict())
                net_list.append(a)
                with torch.no_grad():
                    for test_batch, (lrimg, hrimg) in enumerate(test_data, start=0):
                        lrimg = lrimg.cuda()
                        hrimg = hrimg.cuda()

                        lr_img, hr_img = net_G(hrimg)

                        i1 = hrimg.cpu().detach().numpy()[0]
                        i2 = hr_img.cpu().detach().numpy()[0]
                        i1 = (i1 + 1.0) / 2.0
                        i1 = np.clip(i1, 0.0, 1.0)
                        i2 = (i2 + 1.0) / 2.0
                        i2 = np.clip(i2, 0.0, 1.0)

                        i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
                        i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
                        psnr1 = psnr_get(i1, i2)
                        psnr1_sum += psnr1
                psnr1_sum = psnr1_sum / (test_batch + 1)
                psnr_list.append(psnr1_sum)
                net_G.train()

        psnr_max = max(psnr_list)
        index = psnr_list.index(psnr_max)
        if (i + 1) >= 800:
            checkpoint = {
                'epoch': i+1,
                'model_state_dict': net_list[index],
                'optimizer_state_dict': optimizer_G.state_dict(),
            }
            torch.save(checkpoint, './train_log_v13_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4_{0}.pt'.format(i + 1))
            del net_list
        print('{0}|{1}    avg_loss={2:.6f}    time={3:.3f}min   psnr_max:{4:.3f}'.format(i + 1, epochs, sum_loss / batch, (time.time() - t1) / 60,  psnr_max))
        str_list = [str(item) for item in psnr_list]
        # 使用join方法将新列表中的元素连接成一个字符串，元素之间由空格分隔
        str_list = ' '.join(str_list)
        str_write = '{0}|{1}    avg_loss={2:.6f}    time={3:.3f}min   psnr_max:{4:.3f}'.format(i + 1, epochs, sum_loss / batch, (time.time() - t1) / 60, psnr_max) + '\n'
        fp = open('train_log_v13_visual_quant_222/CS_Omni_P2P_x4/CS_Omni_P2P_x4.txt', 'a+')
        fp.write(str_write)
        fp.close()
