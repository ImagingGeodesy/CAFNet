
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
import math

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class Weight_allocation(nn.Module):
    def __init__(self):
        super(Weight_allocation, self).__init__()
    def forward(self, xh, xl):
        Weight_att = torch.sigmoid(xh)
        xh_w = Weight_att * xh
        xl_w = (1 - Weight_att) * xl
        x_out = torch.cat((xh_w, xl_w), dim=1)
        return x_out

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class ECAAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        kernel_size = int(abs((math.log2(channels) / 2) + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2, bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0,2,1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0,2,1).unsqueeze(-1)
        return x*y.expand_as(x)

class Down_wt_WDB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt_WDB, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.RECA = ECAAttention(128)
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x_w = self.RECA(x)
        x = x + x_w
        x = self.conv_bn_relu(x)

        return x

class MSA(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        self.pre_project_l = nn.Conv2d(dim_xl, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size , data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size )
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size , data_format='channels_first'),
            nn.Conv2d(group_size , group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size )
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size , data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size , group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 , data_format='channels_first'),
            nn.Conv2d(dim_xl * 2, dim_xl, 1)
        )
        self.wa = Weight_allocation()

    def forward(self, xh, xl):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xl = self.pre_project_l(xl)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(self.wa(xh[0], xl[0]))
        x1 = self.g1(self.wa(xh[1], xl[1]))
        x2 = self.g2(self.wa(xh[2], xl[2]))
        x3 = self.g3(self.wa(xh[3], xl[3]))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x

class CUB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(CUB, self).__init__()
        pad0 = int((kernel_size - 1) / 2)
        pad1 = int((kernel_size - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                                 padding=(pad0, 0))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.Up2 = nn.Upsample(scale_factor=2)

        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.Up2(x)

        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r

        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

def double_conv(in_channels, out_channels):  # 双层卷积模型，神经网络最基本的框架
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3指kernel_size，即卷积核3*3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_my(in_channels, out_channels):  # 双层卷积模型，神经网络最基本的框架
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding=0),
        nn.BatchNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
        nn.ReLU(inplace=True),
    )

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Encoder_WDB(nn.Module):

    def __init__(self,):
        super().__init__()

        self.dconv_down0 = double_conv(3, 32)
        self.dconv_down1 = double_conv(32, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)

        # self.maxpool = nn.MaxPool2d(2)
        self.pool1 = Down_wt_WDB(32,32)
        self.pool2 = Down_wt_WDB(64, 64)
        self.pool3 = Down_wt_WDB(128, 128)

    def forward(self, x):

        # encode
        conv0 = self.dconv_down0(x)
        x = self.pool1(conv0)

        conv1 = self.dconv_down1(x)
        x = self.pool2(conv1)

        conv2 = self.dconv_down2(x)
        x = self.pool3(conv2)

        conv3 = self.dconv_down3(x)

        return conv0, conv1, conv2, conv3
class my_CAFNet(nn.Module):

    def __init__(self,n_classes=3):
        super().__init__()

        self.backbone = Encoder_WDB()
        # self.decoder = Decoder()
        self.dconv_up3 = double_conv(128 + 128, 128)
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(64, 32)

        self.ecub3 = CUB(128, 128)
        self.ecub2 = CUB(64, 64)
        self.ecub1 = CUB(32, 32)

        self.MSA3 = MSA(256, 128)
        self.MSA2 = MSA(128, 64)
        self.MSA1 = MSA(64, 32)

        self.conv_last = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):

        # encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        g3 = self.MSA3(x4,x3)
        d3 = self.dconv_up3(x4)
        d3 = self.ecub3(d3)
        d3 = torch.add(d3,g3)

        g2 = self.MSA2(g3, x2)
        d2 = self.dconv_up2(d3)
        d2 = self.ecub2(d2)
        d2 = torch.add(d2,g2)

        g1 = self.MSA1(g2, x1)
        d1 = self.dconv_up1(d2)
        d1 = self.ecub1(d1)
        d1 = torch.add(d1,g1)

        out = self.conv_last(d1)

        return out

if __name__ == "__main__":
    model = my_CAFNet(n_classes=2)
    model.eval()
    input = torch.rand(1, 3, 128, 128)
    from ptflops import get_model_complexity_info
    flops1, params1 = get_model_complexity_info(model, (3, 128, 128), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs:  {flops1}")
    print(f"Parameters:  {params1}")
