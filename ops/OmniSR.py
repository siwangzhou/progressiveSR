import torch
import torch.nn as nn
from ops.OSAG import OSAG
from einops import rearrange
from ops.pixelshuffle import pixelshuffle_block
import torch.nn.functional as F
from ops.Quantization import Quantization_RS


class CS_DownSample_x4(nn.Module):
    def __init__(self):
        super(CS_DownSample_x4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8*8*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_DownSample_x3_77(nn.Module):
    def __init__(self):
        super(CS_DownSample_x3_77, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9*8*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=9, a2=8, c=3)

        return x


class CS_DownSample_x3_55(nn.Module):
    def __init__(self):
        super(CS_DownSample_x3_55, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9*9*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)

        return x


class CS_DownSample_x3_2(nn.Module):
    def __init__(self):
        super(CS_DownSample_x3_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10*10*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)

        return x


class CS_DownSample_x2_91(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_91, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=11*11*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)

        return x


class CS_DownSample_x2_66(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_66, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12*12*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)

        return x


class CS_DownSample_x2_46(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_46, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=13*13*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=13, a2=13, c=3)

        return x



class CS_DownSample_xDiff_v4(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(9*9-8*8)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v7(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(10*10-8*8)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v9(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(11*11-8*8)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v11(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(12*12-8*8)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_UpSample_x2_46(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2_46, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=13, stride=13, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x2_66(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2_66, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=12, stride=12, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x2_91(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2_91, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=11, stride=11, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x3_2(nn.Module):
    def __init__(self):
        super(CS_UpSample_x3_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=10, stride=10, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x3_55(nn.Module):
    def __init__(self):
        super(CS_UpSample_x3_55, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=9, stride=9, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x3_77(nn.Module):
    def __init__(self):
        super(CS_UpSample_x3_77, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=(9, 8), stride=(9, 8), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x4(nn.Module):
    def __init__(self):
        super(CS_UpSample_x4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=8, stride=8, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class OmniSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR, self).__init__()
        kwargs = kwargs['kwards']
        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                                bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out


class Enhancemodule(nn.Module):
    def __init__(self, num_channels=3, num_features=64):
        super(Enhancemodule, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)

        # Additional convolution layers for feature enhancement
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        # Final convolution layer to reduce back to 3 channels
        self.conv5 = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class LR_SR_x4_v1_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v1_quant, self).__init__()
        self.layer1 = CS_DownSample_x4()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR


class LR_SR_x4_v2_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v2_quant, self).__init__()
        self.layer1 = CS_DownSample_x4()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x4()
        self.layer4 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        HR = self.layer4(HR)
        return LR, HR


class LR_SR_x4_v3_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v3_quant, self).__init__()
        self.layer1 = CS_DownSample_x3_55()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x3_55()
        self.layer4 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        HR = self.layer4(HR)
        return LR, HR


class LR_SR_x4_v4_quant(nn.Module):
    def __init__(self, kwards, net_v2_pretrained):
        super(LR_SR_x4_v4_quant, self).__init__()

        self.net_v2_pretrained = net_v2_pretrained
        self.layer1 = CS_DownSample_xDiff_v4()  # 32卷积核的下采样
        self.layer2 = CS_UpSample_x3_55()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.net_v2_pretrained.layer2(self.net_v2_pretrained.layer1(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff = self.layer1(x)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)

        HR = self.layer2(new_LR)
        HR = self.layer3(HR)
        return new_LR, HR


class LR_SR_x4_v6_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v6_quant, self).__init__()
        self.layer1 = CS_DownSample_x3_2()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x3_2()
        self.layer4 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        HR = self.layer4(HR)
        return LR, HR


class LR_SR_x4_v7_quant(nn.Module):
    def __init__(self, kwards, net_v2_pretrained):
        super(LR_SR_x4_v7_quant, self).__init__()

        self.net_v2_pretrained = net_v2_pretrained
        self.layer1 = CS_DownSample_xDiff_v7()  # 32卷积核的下采样
        self.layer2 = CS_UpSample_x3_2()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.net_v2_pretrained.layer2(self.net_v2_pretrained.layer1(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff = self.layer1(x)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)

        HR = self.layer2(new_LR)
        HR = self.layer3(HR)
        return new_LR, HR


class LR_SR_x4_v8_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v8_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_91()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x2_91()
        self.layer4 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        HR = self.layer4(HR)
        return LR, HR


class LR_SR_x4_v9_quant(nn.Module):
    def __init__(self, kwards, net_v2_pretrained):
        super(LR_SR_x4_v9_quant, self).__init__()

        self.net_v2_pretrained = net_v2_pretrained
        self.layer1 = CS_DownSample_xDiff_v9()  # 32卷积核的下采样
        self.layer2 = CS_UpSample_x2_91()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.net_v2_pretrained.layer2(self.net_v2_pretrained.layer1(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff = self.layer1(x)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)

        HR = self.layer2(new_LR)
        HR = self.layer3(HR)
        return new_LR, HR


class LR_SR_x4_v10_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v10_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_66()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x2_66()
        self.layer4 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        HR = self.layer4(HR)
        return LR, HR


class LR_SR_x4_v11_quant(nn.Module):
    def __init__(self, kwards, net_v2_pretrained):
        super(LR_SR_x4_v11_quant, self).__init__()

        self.net_v2_pretrained = net_v2_pretrained
        self.layer1 = CS_DownSample_xDiff_v11()  # 32卷积核的下采样
        self.layer2 = CS_UpSample_x2_66()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.net_v2_pretrained.layer2(self.net_v2_pretrained.layer1(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff = self.layer1(x)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)

        HR = self.layer2(new_LR)
        HR = self.layer3(HR)
        return new_LR, HR


class LR_SR_x4_v12_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v12_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_46()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x2_46()
        self.layer4 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        HR = self.layer4(HR)
        return LR, HR