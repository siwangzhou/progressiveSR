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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(10*10-9*9)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v9(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(11*11-10*10)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v11(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(12*12-11*11)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v13(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v13, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(13*13-12*12)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v15(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v15, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(14*14-13*13)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v17(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v17, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(15*15-14*14)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v19(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v19, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(16*16-15*15)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_UpSample_x2(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=16, stride=16, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x


class CS_UpSample_x2_13(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2_13, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=15, stride=15, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        return x

class CS_UpSample_x2_28(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2_28, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32*32*3, kernel_size=14, stride=14, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

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


class LR_SR_x4_general(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_general, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()
        self.v2_upsample = CS_UpSample_x4()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=8, stride=8, padding=0)
        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()
        self.v4_upsample = CS_UpSample_x3_55()

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9 * 9 * 3, kernel_size=9, stride=9, padding=0)
        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()
        self.v7_upsample = CS_UpSample_x3_2()

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=10 * 10 * 3, kernel_size=10, stride=10, padding=0)
        self.v9_downsample = CS_DownSample_xDiff_v9()
        self.v9_quant = Quantization_RS()
        self.v9_upsample = CS_UpSample_x2_91()

        self.conv4 = nn.Conv2d(in_channels=3, out_channels=11 * 11 * 3, kernel_size=11, stride=11, padding=0)
        self.v11_downsample = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()
        self.v11_upsample = CS_UpSample_x2_66()

        self.conv5 = nn.Conv2d(in_channels=3, out_channels=12 * 12 * 3, kernel_size=12, stride=12, padding=0)
        self.v13_downsample = CS_DownSample_xDiff_v13()
        self.v13_quant = Quantization_RS()
        self.v13_upsample = CS_UpSample_x2_46()

        self.conv6 = nn.Conv2d(in_channels=3, out_channels=13 * 13 * 3, kernel_size=13, stride=13, padding=0)
        self.v15_downsample = CS_DownSample_xDiff_v15()
        self.v15_quant = Quantization_RS()
        self.v15_upsample = CS_UpSample_x2_28()

        self.conv7 = nn.Conv2d(in_channels=3, out_channels=14 * 14 * 3, kernel_size=14, stride=14, padding=0)
        self.v17_downsample = CS_DownSample_xDiff_v17()
        self.v17_quant = Quantization_RS()
        self.v17_upsample = CS_UpSample_x2_13()

        self.Omni = OmniSR(kwards=kwards)

    def forward(self, x):
        v2_LR = self.v2_downsample(x)
        v2_LR_quant = self.v2_quant(v2_LR)
        v2_SR_temp = self.v2_upsample(v2_LR_quant)
        v2_SR = self.Omni(v2_SR_temp)

        v2_LR_feat = self.conv1(v2_LR_quant)
        v4_LR_diff = self.v4_downsample(x)
        v4_LR_diff_quant = self.v4_quant(v4_LR_diff)
        v4_LR_cat = torch.cat((v2_LR_feat, v4_LR_diff_quant), dim=1)
        v4_LR = rearrange(v4_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)
        v4_SR_temp = self.v4_upsample(v4_LR)
        v4_SR = self.Omni(v4_SR_temp)

        v4_LR_feat = self.conv2(v4_LR)
        v7_LR_diff = self.v7_downsample(x)
        v7_LR_diff_quant = self.v7_quant(v7_LR_diff)
        v7_LR_cat = torch.cat((v4_LR_feat, v7_LR_diff_quant), dim=1)
        v7_LR = rearrange(v7_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)
        v7_SR_temp = self.v7_upsample(v7_LR)
        v7_SR = self.Omni(v7_SR_temp)

        v7_LR_feat = self.conv3(v7_LR)
        v9_LR_diff = self.v9_downsample(x)
        v9_LR_diff_quant = self.v9_quant(v9_LR_diff)
        v9_LR_cat = torch.cat((v7_LR_feat, v9_LR_diff_quant), dim=1)
        v9_LR = rearrange(v9_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)
        v9_SR_temp = self.v9_upsample(v9_LR)
        v9_SR = self.Omni(v9_SR_temp)

        v9_LR_feat = self.conv4(v9_LR)
        v11_LR_diff = self.v11_downsample(x)
        v11_LR_diff_quant = self.v11_quant(v11_LR_diff)
        v11_LR_cat = torch.cat((v9_LR_feat, v11_LR_diff_quant), dim=1)
        v11_LR = rearrange(v11_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)
        v11_SR_temp = self.v11_upsample(v11_LR)
        v11_SR = self.Omni(v11_SR_temp)

        v11_LR_feat = self.conv5(v11_LR)
        v13_LR_diff = self.v13_downsample(x)
        v13_LR_diff_quant = self.v13_quant(v13_LR_diff)
        v13_LR_cat = torch.cat((v11_LR_feat, v13_LR_diff_quant), dim=1)
        v13_LR = rearrange(v13_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=13, a2=13, c=3)
        v13_SR_temp = self.v13_upsample(v13_LR)
        v13_SR = self.Omni(v13_SR_temp)

        v13_LR_feat = self.conv6(v13_LR)
        v15_LR_diff = self.v15_downsample(x)
        v15_LR_diff_quant = self.v15_quant(v15_LR_diff)
        v15_LR_cat = torch.cat((v13_LR_feat, v15_LR_diff_quant), dim=1)
        v15_LR = rearrange(v15_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=14, a2=14, c=3)
        v15_SR_temp = self.v15_upsample(v15_LR)
        v15_SR = self.Omni(v15_SR_temp)

        v15_LR_feat = self.conv7(v15_LR)
        v17_LR_diff = self.v17_downsample(x)
        v17_LR_diff_quant = self.v17_quant(v17_LR_diff)
        v17_LR_cat = torch.cat((v15_LR_feat, v17_LR_diff_quant), dim=1)
        v17_LR = rearrange(v17_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=15, a2=15, c=3)
        v17_SR_temp = self.v17_upsample(v17_LR)
        v17_SR = self.Omni(v17_SR_temp)

        return [v2_LR, v4_LR, v7_LR, v9_LR, v11_LR, v13_LR, v15_LR, v17_LR], [v2_SR, v4_SR, v7_SR, v9_SR, v11_SR, v13_SR, v15_SR, v17_SR]