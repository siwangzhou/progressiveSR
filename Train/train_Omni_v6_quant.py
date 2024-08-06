from ops.OmniSR import LR_SR_x4_v6_quant
import torch
import numpy as np
import math
import copy
from torch.utils.checkpoint import checkpoint
from data_v6v7 import MyDataset, Test
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

    net_G = LR_SR_x4_v6_quant(kwards=kwards).cuda()

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
        if (i + 1) % 1 == 0:
            checkpoint = {
                'epoch': i+1,
                'model_state_dict': net_list[index],
                'optimizer_state_dict': optimizer_G.state_dict(),
            }
            torch.save(checkpoint, './train_log_v6_quant/CS_Omni_P2P_x4/CS_Omni_P2P_x4_{0}.pt'.format(i + 1))
            del net_list
        print('{0}|{1}    avg_loss={2:.6f}    time={3:.3f}min   psnr_max:{4:.3f}'.format(i + 1, epochs, sum_loss / batch, (time.time() - t1) / 60,  psnr_max))
        str_list = [str(item) for item in psnr_list]
        # 使用join方法将新列表中的元素连接成一个字符串，元素之间由空格分隔
        str_list = ' '.join(str_list)
        str_write = '{0}|{1}    avg_loss={2:.6f}    time={3:.3f}min   psnr_max:{4:.3f}'.format(i + 1, epochs, sum_loss / batch, (time.time() - t1) / 60, psnr_max) + '\n'
        fp = open('./train_log_v6_quant/CS_Omni_P2P_x4/CS_Omni_P2P_x4.txt', 'a+')
        fp.write(str_write)
        fp.close()
