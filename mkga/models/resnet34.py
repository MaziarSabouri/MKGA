import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import random


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig
        indices = torch.randperm(B).to(x.device)
        mu_perm, sig_perm = mu[indices], sig[indices]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        mu_mix = lam * mu + (1 - lam) * mu_perm
        sig_mix = lam * sig + (1 - lam) * sig_perm
        return x_normed * sig_mix + mu_mix


class ResNet34(nn.Module):
    def __init__(self, num_classes=2, use_feedback=False, use_mixstyle=False):
        super().__init__()
        self.use_feedback = use_feedback
        self.use_mixstyle = use_mixstyle

        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        self.upconv4 = self._up_block(512, 256)
        self.upconv3 = self._up_block(256, 128)
        self.upconv2 = self._up_block(128, 64)
        self.upconv1 = self._up_block(64, 64)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head_pos = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 3))
        self.head_tirads = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, 2), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, img):
        x1 = self.enc1(img)
        x2 = self.enc2(x1)
        if self.use_mixstyle:
            x2 = self.mixstyle(x2)
        x3 = self.enc3(x2)
        if self.use_mixstyle:
            x3 = self.mixstyle(x3)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d4 = self.upconv4(x5)
        d4 = F.interpolate(d4, x4.shape[-2:]) + x4
        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, x3.shape[-2:]) + x3
        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, x2.shape[-2:]) + x2
        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, x1.shape[-2:]) + x1

        mask_logits = self.final_upsample(d1)
        if mask_logits.shape[-1] != img.shape[-1]:
            mask_logits = F.interpolate(mask_logits, size=img.shape[-2:], mode='bilinear')

        img_feat = self.gap(x5).view(x5.size(0), -1)
        return mask_logits, self.head_pos(img_feat), self.head_tirads(img_feat)
