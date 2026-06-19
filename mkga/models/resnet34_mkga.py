import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiKernelFeatureExtractor(nn.Module):
    def __init__(self, in_c, out_c, kernel_combo="3_5"):
        super().__init__()
        if kernel_combo == "1_3":
            self.branch1 = nn.Conv2d(in_c, out_c // 2, kernel_size=1, padding=0)
            self.branch2 = nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1)
        elif kernel_combo == "3_7":
            self.branch1 = nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1)
            self.branch2 = nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=3, dilation=3)
        else:
            self.branch1 = nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1)
            self.branch2 = nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=2, dilation=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.branch1(x), self.branch2(x)], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


class MKGABlock(nn.Module):
    def __init__(self, in_c_high, in_c_skip, out_c, use_gate=True, use_multi_kernel=True, kernel_combo="3_5"):
        super().__init__()
        self.use_gate = use_gate
        self.use_multi_kernel = use_multi_kernel
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_c_high, out_c, 1)
        if self.use_multi_kernel:
            self.skip_cleaner = MultiKernelFeatureExtractor(in_c_skip, out_c, kernel_combo=kernel_combo)
        else:
            self.skip_cleaner = nn.Sequential(
                nn.Conv2d(in_c_skip, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )
        if self.use_gate:
            self.attn = AttentionGate(F_g=out_c, F_l=out_c, F_int=out_c // 2)
        self.res_block = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )

    def forward(self, high, skip):
        high_up = self.reduce(self.up(high))
        if high_up.shape[-2:] != skip.shape[-2:]:
            high_up = F.interpolate(high_up, size=skip.shape[-2:], mode='bilinear')
        skip_clean = self.skip_cleaner(skip)
        if self.use_gate:
            skip_gated = self.attn(g=high_up, x=skip_clean)
        else:
            skip_gated = skip_clean
        return self.res_block(torch.cat([high_up, skip_gated], dim=1))


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1)
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(in_c, out_c, 1)
        self.fuse = nn.Sequential(nn.Conv2d(out_c * 4, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = F.interpolate(self.pool_conv(self.pool(x)), size=x.shape[-2:], mode='bilinear')
        return self.fuse(torch.cat([x1, x2, x3, x4], dim=1))


class ResNet34_MKGA(nn.Module):
    def __init__(self, num_classes=2, ablation_dict=None):
        super().__init__()
        use_gate = True
        use_multi_kernel = True
        kernel_combo = "3_5"
        if ablation_dict is not None:
            use_gate = ablation_dict.get('use_gate', True)
            use_multi_kernel = ablation_dict.get('use_multi_kernel', True)
            kernel_combo = ablation_dict.get('kernel_combo', "3_5")

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        self.aspp = ASPP(512, 256)
        self.dec4 = MKGABlock(256, 256, 128, use_gate=use_gate, use_multi_kernel=use_multi_kernel, kernel_combo=kernel_combo)
        self.dec3 = MKGABlock(128, 128, 64, use_gate=use_gate, use_multi_kernel=use_multi_kernel, kernel_combo=kernel_combo)
        self.dec2 = MKGABlock(64, 64, 32, use_gate=use_gate, use_multi_kernel=use_multi_kernel, kernel_combo=kernel_combo)
        self.dec1 = MKGABlock(32, 64, 16, use_gate=use_gate, use_multi_kernel=use_multi_kernel, kernel_combo=kernel_combo)
        self.head_seg = nn.Conv2d(16, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_pos = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.head_tirads = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, img):
        x1 = self.enc1(img)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        bridge = self.aspp(x5)
        d4 = self.dec4(bridge, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        mask = self.head_seg(d1)
        if mask.shape[-2:] != img.shape[-2:]:
            mask = F.interpolate(mask, size=img.shape[-2:], mode='bilinear')
        global_feat = self.gap(bridge).flatten(1)
        return mask, self.head_pos(global_feat), self.head_tirads(global_feat)
