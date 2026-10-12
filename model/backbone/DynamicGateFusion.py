import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Enhanced_SE_SemanticFusionVGG16']



class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)

        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        return x * spatial_att



class DynamicGateFusion(nn.Module):
    def __init__(self, channel_g, channel_a):
        super(DynamicGateFusion, self).__init__()
        self.gate_g = nn.Sequential(
            nn.Conv2d(channel_g + channel_a, channel_g, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Conv2d(channel_g + channel_a, channel_a, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, gyro, acce):
        combined = torch.cat([gyro, acce], dim=1)
        gate_g = self.gate_g(combined)
        gate_a = self.gate_a(combined)
        return gyro * gate_g + acce * gate_a, acce * (1 - gate_a) + gyro * (1 - gate_g)



class Enhanced_SE_SemanticFusionVGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(Enhanced_SE_SemanticFusionVGG16, self).__init__()
        
        self.g_layer1 = self._make_cbam_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.g_layer2 = self._make_cbam_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.g_layer3 = self._make_cbam_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.g_layer4 = self._make_cbam_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.g_layer5 = self._make_cbam_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        
        self.a_layer1 = self._make_cbam_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.a_layer2 = self._make_cbam_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.a_layer3 = self._make_cbam_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.a_layer4 = self._make_cbam_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.a_layer5 = self._make_cbam_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        
        self.fusion_layers = nn.ModuleList([
            DynamicGateFusion(64, 64),
            DynamicGateFusion(128, 128),
            DynamicGateFusion(256, 256),
            DynamicGateFusion(512, 512),
            DynamicGateFusion(512, 512),
            DynamicGateFusion(768, 768),
            DynamicGateFusion(768, 768),
            DynamicGateFusion(n_classes, n_classes)
        ])

        
        self.g_layer6 = self._make_fc_layer(7 * 7 * 512, 768)
        self.g_layer7 = nn.Linear(768, 768)
        self.g_layer8 = nn.Linear(768, n_classes)
        self.a_layer6 = self._make_fc_layer(7 * 7 * 512, 768)
        self.a_layer7 = nn.Linear(768, 768)
        self.a_layer8 = nn.Linear(768, n_classes)

    def _make_cbam_conv_block(self, in_list, out_list, k_list, p_list, pooling_k, pooling_s):
        layers = []
        for i in range(len(in_list)):
            layers += [
                nn.Conv2d(in_list[i], out_list[i], kernel_size=k_list[i], padding=p_list[i]),
                nn.BatchNorm2d(out_list[i]),
                CBAMLayer(out_list[i]),  
                nn.ReLU()
            ]
        layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
        return nn.Sequential(*layers)

    def _make_fc_layer(self, size_in, size_out):
        return nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.ReLU()
        )

    def forward(self, gyro, acce):
        
        g_out1 = self.g_layer1(gyro)
        a_out1 = self.a_layer1(acce)
        g_out1, a_out1 = self.fusion_layers[0](g_out1, a_out1)

        g_out2 = self.g_layer2(g_out1)
        a_out2 = self.a_layer2(a_out1)
        g_out2, a_out2 = self.fusion_layers[1](g_out2, a_out2)

        g_out3 = self.g_layer3(g_out2)
        a_out3 = self.a_layer3(a_out2)
        g_out3, a_out3 = self.fusion_layers[2](g_out3, a_out3)

        g_out4 = self.g_layer4(g_out3)
        a_out4 = self.a_layer4(a_out3)
        g_out4, a_out4 = self.fusion_layers[3](g_out4, a_out4)

        g_out5 = self.g_layer5(g_out4)
        a_out5 = self.a_layer5(a_out4)
        g_out5, a_out5 = self.fusion_layers[4](g_out5, a_out5)

        
        g_flat = g_out5.view(g_out5.size(0), -1)
        a_flat = a_out5.view(a_out5.size(0), -1)

        g_out6 = self.g_layer6(g_flat)
        a_out6 = self.a_layer6(a_flat)
        g_out6, a_out6 = self.fusion_layers[5](g_out6, a_out6)

        g_out7 = self.g_layer7(g_out6)
        a_out7 = self.a_layer7(a_out6)
        g_out7, a_out7 = self.fusion_layers[6](g_out7, a_out7)

        g_out8 = self.g_layer8(g_out7)
        a_out8 = self.a_layer8(a_out7)
        return g_out8, a_out8