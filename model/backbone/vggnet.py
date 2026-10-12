import torchvision
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
import torch.nn.init as init
__all__ = ['VggNet16','SeVggNet16','SeFusionVGG16','SeFusionVGG16_Berkeley','SemanticFusionVGG16','SemanticFusionVGG16_Berkeley']


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()           
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),   
            nn.ReLU(inplace=True),                                  
            nn.Linear(channel // reduction, channel, bias=False),   
            nn.Sigmoid()                                            
        )

    def forward(self, x):
        b, c, _, _ = x.size()                   
        y = self.avg_pool(x).view(b, c)         
        y = self.fc(y).view(b, c, 1, 1)         
        return x * y.expand_as(x)               


def init_conv(conv, glu=True):                
    init.xavier_uniform_(conv.weight)         
    if conv.bias is not None:                 
        conv.bias.data.zero_()                


class SelfAttention(nn.Module):
    



    def __init__(self, in_dim, activation=F.relu):       
        super(SelfAttention, self).__init__()                  
        self.chanel_in = in_dim                          
        self.activation = activation                     
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)   
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)   
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)        
        
        self.gamma = nn.Parameter(torch.zeros(1))       

        self.softmax  = nn.Softmax(dim=-1)              
        
        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
        
    def forward(self, x):
        






        m_batchsize, C, width, height = x.size()

        
        f = self.f(x).view(m_batchsize, -1, width * height) 
        g = self.g(x).view(m_batchsize, -1, width * height) 
        h = self.h(x).view(m_batchsize, -1, width * height) 
        
        attention = torch.bmm(f.permute(0, 2, 1), g) 
        attention = self.softmax(attention)
        
        self_attetion = torch.bmm(h, attention) 
        self_attetion = self_attetion.view(m_batchsize, C, width, height) 
        
        out = self.gamma * self_attetion + x
        return out


class SE_Fusion(nn.Module):
    def __init__(self, channel_g=128, channel_a=256, reduction=4):   
        super(SE_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc_g = nn.Sequential(
            nn.Linear((channel_g + channel_a), (channel_g + channel_a) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_g + channel_a) // reduction, channel_g, bias=True),
            nn.Sigmoid()
        )
        self.fc_a = nn.Sequential(
            nn.Linear((channel_g + channel_a), (channel_g + channel_a) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_g + channel_a) // reduction, channel_a, bias=True),
            nn.Sigmoid()
        )
        
        self.scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel_g + channel_a, (channel_g + channel_a) // (2 ** i), 3, padding=2 ** i,
                          dilation=2 ** i),
                nn.BatchNorm2d((channel_g + channel_a) // (2 ** i)),
                nn.ReLU()
            ) for i in range(3)  
        ])
        
        self.scale_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel_g + channel_a, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, gyro, acce):   
        
        b_g, c_g, _, _ = gyro.size()
        y_g = self.avg_pool(gyro).view(b_g, c_g)
        y_a = self.avg_pool(acce).view(b_g, -1)
        z = torch.cat((y_g, y_a), 1)

        
        
        combined = torch.cat([gyro, acce], dim=1)
        scale_feats = [conv(combined) for conv in self.scale_conv]

        
        scale_weights = self.scale_att(combined)  

        
        fused_feat = sum(w * f for w, f in zip(scale_weights.unbind(1), scale_feats))

        
        
        y_g = self.fc_g(fused_feat.view(b_g, -1))
        y_a = self.fc_a(fused_feat.view(b_g, -1))

        return 2 * gyro * y_g.view_as(gyro), 2 * acce * y_a.view_as(acce)


class SE_semantic_Fusion(nn.Module):
    def __init__(self, channel_g=128, channel_a=256, reduction=4):
        super(SE_semantic_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_g = nn.Sequential(
            nn.Linear((channel_g + channel_a), (channel_g + channel_a) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_g + channel_a) // reduction, channel_g, bias=True),
            nn.Sigmoid()
        )
        self.fc_g_add = nn.Sequential(
            nn.Linear(channel_g, channel_g // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel_g // reduction, channel_g, bias=True),
            nn.Sigmoid()
        )
        self.fc_g_mul = nn.Sequential(
            nn.Linear(channel_g, channel_g // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel_g // reduction, channel_g, bias=True),
            nn.Sigmoid()
        )
        self.fc_a = nn.Sequential(
            nn.Linear((channel_g + channel_a), (channel_g + channel_a) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_g + channel_a) // reduction, channel_a, bias=True),
            nn.Sigmoid()
        )
        self.fc_a_add = nn.Sequential(
            nn.Linear(channel_a, channel_a // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel_a // reduction, channel_a, bias=True),
            nn.Sigmoid()
        )
        self.fc_a_mul = nn.Sequential(
            nn.Linear(channel_a, channel_a // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel_a // reduction, channel_a, bias=True),
            nn.Sigmoid()
        )

    def forward(self, gyro, acce):

        
        if len(gyro.size()) < 4:
            b_g, c_g = gyro.size()
            y_g = gyro
            fm_g = gyro.view(gyro.size(0), -1)
            G_g = torch.mm(fm_g, fm_g.t())
            norm_G_g = F.normalize(G_g, p=2, dim=1)

            b_a, c_a = acce.size()
            y_a = acce
            fm_a = acce.view(acce.size(0), -1)
            G_a = torch.mm(fm_a, fm_a.t())
            norm_G_a = F.normalize(G_a, p=2, dim=1)
        else:
            b_g, c_g, _, _ = gyro.size()
            fm_g = gyro.view(gyro.size(0), -1)
            G_g = torch.mm(fm_g, fm_g.t())
            norm_G_g = F.normalize(G_g, p=2, dim=1)
            y_g = self.avg_pool(gyro).view(b_g, c_g)

            b_a, c_a, _, _ = acce.size()
            fm_a = acce.view(acce.size(0), -1)
            G_a = torch.mm(fm_a, fm_a.t())
            norm_G_a = F.normalize(G_a, p=2, dim=1)
            y_a = self.avg_pool(acce).view(b_a, c_a)

        
        z_add = torch.add(torch.mm(norm_G_g, y_g), torch.mm(norm_G_a, y_a))   
        z_mul = torch.mul(torch.mm(norm_G_g, y_g), torch.mm(norm_G_a, y_a))   
        z = torch.cat((torch.mm(norm_G_g, y_g), torch.mm(norm_G_a, y_a)), 1)   

        
        if len(gyro.size()) < 4:
            y_g = self.fc_g(z).view(b_g, c_g)
            y_g_add = self.fc_g_add(z_add).view(b_g, c_g)
            y_g_mul = self.fc_g_mul(z_mul).view(b_g, c_g)
            y_a = self.fc_a(z).view(b_a, c_a)
            y_a_add = self.fc_a_add(z_add).view(b_a, c_a)
            y_a_mul = self.fc_a_mul(z_mul).view(b_a, c_a)
        else:
            y_g = self.fc_g(z).view(b_g, c_g, 1, 1)
            y_g_add = self.fc_g_add(z_add).view(b_g, c_g, 1, 1)
            y_g_mul = self.fc_g_mul(z_mul).view(b_g, c_g, 1, 1)
            y_a = self.fc_a(z).view(b_a, c_a, 1, 1)
            y_a_add = self.fc_a_add(z_add).view(b_a, c_a, 1, 1)
            y_a_mul = self.fc_a_mul(z_mul).view(b_a, c_a, 1, 1)

        
        return torch.mul((F.relu(y_g) + F.relu(y_g_add) + F.relu(y_g_mul)).expand_as(gyro), gyro), \
               torch.mul((F.relu(y_a) + F.relu(y_a_add) + F.relu(y_a_mul)).expand_as(acce), acce)

class SMMRF(nn.Module):
    

    MODES = ('mean', 'concat', 'concat_gate', 'full')

    def __init__(self, channels, reduction=4, mode='full'):
        super(SMMRF, self).__init__()
        if mode not in self.MODES:
            raise ValueError('Unknown SMMRF mode: %s' % mode)
        self.mode = mode
        hidden = max(channels // reduction, 8)
        self.concat_relation = nn.Linear(2 * channels, hidden)
        self.gated_relation = nn.Linear(channels, hidden)
        self.cross_relation = nn.Linear(channels, hidden)
        self.gyro_excitation = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.acc_excitation = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        for layer in list(self.gyro_excitation) + list(self.acc_excitation):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _context(features):
        with torch.cuda.amp.autocast(enabled=False):
            flattened = features.float().flatten(1)
            similarity = F.normalize(flattened @ flattened.t(), p=2, dim=1)
            pooled = features.float().mean(dim=(2, 3)) if features.ndim == 4 else features.float()
            return similarity @ pooled

    def forward(self, gyro, acc):
        if gyro.shape[:2] != acc.shape[:2] or gyro.ndim != acc.ndim:
            raise ValueError('SMMRF needs aligned sensor batches with equal channels')
        if self.mode == 'mean':
            fused = 0.5 * (gyro + acc)
            return fused, fused
        g = self._context(gyro)
        a = self._context(acc)
        con = F.relu(self.concat_relation(torch.cat((g, a), dim=1)))
        mix = torch.sigmoid(g + a)
        gated = F.relu(self.gated_relation(mix * g + (1 - mix) * a))
        crossed = F.relu(self.cross_relation(g * a - 0.5 * (g + a)))
        active_relations = {
            'concat': 1,
            'concat_gate': 2,
            'full': 3,
        }[self.mode]
        relations = (con, gated, crossed)[:active_relations]
        g_signal = sum(self.gyro_excitation[i](value) for i, value in enumerate(relations))
        a_signal = sum(self.acc_excitation[i](value) for i, value in enumerate(relations))
        
        g_gate = 1 + 0.5 * torch.tanh(g_signal / 3)
        a_gate = 1 + 0.5 * torch.tanh(a_signal / 3)
        if gyro.ndim == 4:
            g_gate = g_gate[:, :, None, None]
            a_gate = a_gate[:, :, None, None]
        return gyro * g_gate.to(gyro.dtype), acc * a_gate.to(acc.dtype)


class SE_semantic_Fusion_Berkeley(nn.Module):
    def __init__(self,channel_1=128,channel_2=128,channel_3=128,channel_4=256,channel_5=128,channel_6=256,reduction=4):
        super(SE_semantic_Fusion_Berkeley, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_a1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_a1_add = nn.Sequential(
            nn.Linear((channel_1), (channel_1) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_a1_mul = nn.Sequential(
            nn.Linear((channel_1), (channel_1) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_a2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_a2_add = nn.Sequential(
            nn.Linear((channel_2), (channel_2) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_2) // reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_a2_mul = nn.Sequential(
            nn.Linear((channel_2), (channel_2) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_2) // reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_a3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_a3_add = nn.Sequential(
            nn.Linear((channel_3), (channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_3) // reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_a3_mul = nn.Sequential(
            nn.Linear((channel_3), (channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_3) // reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_a4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.fc_a4_add = nn.Sequential(
            nn.Linear((channel_4), (channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_4) // reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.fc_a4_mul = nn.Sequential(
            nn.Linear((channel_4), (channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_4) // reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.fc_a5 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_5, bias=True),
            nn.Sigmoid()
        )
        self.fc_a5_add = nn.Sequential(
            nn.Linear((channel_5), (channel_5) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_5) // reduction, channel_5, bias=True),
            nn.Sigmoid()
        )
        self.fc_a5_mul = nn.Sequential(
            nn.Linear((channel_5), (channel_5) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_5) // reduction, channel_5, bias=True),
            nn.Sigmoid()
        )
        self.fc_a6 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_6, bias=True),
            nn.Sigmoid()
        )
        self.fc_a6_add = nn.Sequential(
            nn.Linear((channel_6), (channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_6) // reduction, channel_6, bias=True),
            nn.Sigmoid()
        )
        self.fc_a6_mul = nn.Sequential(
            nn.Linear((channel_6), (channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_6) // reduction, channel_6, bias=True),
            nn.Sigmoid()
        )

    def forward(self, a1, a2, a3, a4, a5, a6):

        
        if len(a1.size())<4:
            b_a1, c_a1= a1.size()
            b_a2, c_a2= a2.size()
            b_a3, c_a3= a3.size()
            b_a4, c_a4= a4.size()
            b_a5, c_a5= a5.size()
            b_a6, c_a6= a6.size()

            y_a1 = a1
            fm_a1 = a1.view(a1.size(0), -1)
            G_a1  = torch.mm(fm_a1, fm_a1.t())
            norm_G_a1 = F.normalize(G_a1, p=2, dim=1)

            y_a2 = a2
            fm_a2= a2.view(a2.size(0), -1)
            G_a2  = torch.mm(fm_a2, fm_a2.t())
            norm_G_a2 = F.normalize(G_a2, p=2, dim=1)

            y_a3 = a3
            fm_a3 = a3.view(a3.size(0), -1)
            G_a3  = torch.mm(fm_a3, fm_a3.t())
            norm_G_a3 = F.normalize(G_a3, p=2, dim=1)

            y_a4 = a4
            fm_a4= a4.view(a4.size(0), -1)
            G_a4  = torch.mm(fm_a4, fm_a4.t())
            norm_G_a4 = F.normalize(G_a4, p=2, dim=1)

            y_a5 = a5
            fm_a5 = a5.view(a5.size(0), -1)
            G_a5  = torch.mm(fm_a5, fm_a5.t())
            norm_G_a5 = F.normalize(G_a5, p=2, dim=1)

            y_a6 = a6
            fm_a6= a6.view(a6.size(0), -1)
            G_a6  = torch.mm(fm_a6, fm_a6.t())
            norm_G_a6 = F.normalize(G_a6, p=2, dim=1)
        else:
            b_a1, c_a1, _, _ = a1.size()
            fm_a1 = a1.view(a1.size(0), -1)
            G_a1  = torch.mm(fm_a1, fm_a1.t())
            norm_G_a1 = F.normalize(G_a1, p=2, dim=1)
            y_a1 = self.avg_pool(a1).view(b_a1, c_a1)

            b_a2, c_a2, _, _ = a2.size()
            fm_a2 = a2.view(a2.size(0), -1)
            G_a2  = torch.mm(fm_a2, fm_a2.t())
            norm_G_a2 = F.normalize(G_a2, p=2, dim=1)
            y_a2 = self.avg_pool(a2).view(b_a2, c_a2)

            b_a3, c_a3, _, _ = a3.size()
            fm_a3 = a3.view(a3.size(0), -1)
            G_a3  = torch.mm(fm_a3, fm_a3.t())
            norm_G_a3 = F.normalize(G_a3, p=2, dim=1)
            y_a3 = self.avg_pool(a3).view(b_a3, c_a3)

            b_a4, c_a4, _, _ = a4.size()
            fm_a4 = a4.view(a4.size(0), -1)
            G_a4  = torch.mm(fm_a4, fm_a4.t())
            norm_G_a4 = F.normalize(G_a4, p=2, dim=1)
            y_a4 = self.avg_pool(a4).view(b_a4, c_a4)

            b_a5, c_a5, _, _ = a5.size()
            fm_a5 = a5.view(a5.size(0), -1)
            G_a5  = torch.mm(fm_a5, fm_a5.t())
            norm_G_a5 = F.normalize(G_a5, p=2, dim=1)
            y_a5 = self.avg_pool(a5).view(b_a5, c_a5)

            b_a6, c_a6, _, _ = a6.size()
            fm_a6 = a6.view(a6.size(0), -1)
            G_a6  = torch.mm(fm_a6, fm_a6.t())
            norm_G_a6 = F.normalize(G_a6, p=2, dim=1)
            y_a6 = self.avg_pool(a6).view(b_a6, c_a6)

        
        z=torch.cat((torch.mm(norm_G_a1,y_a1),torch.mm(norm_G_a2,y_a2),torch.mm(norm_G_a3,y_a3),torch.mm(norm_G_a4,y_a4),torch.mm(norm_G_a5,y_a5),torch.mm(norm_G_a6,y_a6)),1)
        z_add=(torch.mm(norm_G_a1,y_a1)+torch.mm(norm_G_a2,y_a2)+torch.mm(norm_G_a3,y_a3)+torch.mm(norm_G_a4,y_a4)+torch.mm(norm_G_a5,y_a5)+torch.mm(norm_G_a6,y_a6))
        z_mul=torch.mm(norm_G_a1,y_a1)*torch.mm(norm_G_a2,y_a2)*torch.mm(norm_G_a3,y_a3)*torch.mm(norm_G_a4,y_a4)*torch.mm(norm_G_a5,y_a5)*torch.mm(norm_G_a6,y_a6)
        
        
        if len(a1.size())<4:
            y_a1 = self.fc_a1(z).view(b_a1, c_a1)
            y_a1_add = self.fc_a1_add(z_add).view(b_a1, c_a1)
            y_a1_mul = self.fc_a1_mul(z_mul).view(b_a1, c_a1)
            y_a2 = self.fc_a2(z).view(b_a2, c_a2)
            y_a2_add = self.fc_a2_add(z_add).view(b_a2, c_a2)
            y_a2_mul = self.fc_a2_mul(z_mul).view(b_a2, c_a2)
            y_a3 = self.fc_a3(z).view(b_a3, c_a3)
            y_a3_add = self.fc_a3_add(z_add).view(b_a3, c_a3)
            y_a3_mul = self.fc_a3_mul(z_mul).view(b_a3, c_a3)
            y_a4 = self.fc_a4(z).view(b_a4, c_a4)
            y_a4_add = self.fc_a4_add(z_add).view(b_a4, c_a4)
            y_a4_mul = self.fc_a4_mul(z_mul).view(b_a4, c_a4)
            y_a5 = self.fc_a5(z).view(b_a5, c_a5)
            y_a5_add = self.fc_a5_add(z_add).view(b_a5, c_a5)
            y_a5_mul = self.fc_a5_mul(z_mul).view(b_a5, c_a5)
            y_a6 = self.fc_a6(z).view(b_a6, c_a6)
            y_a6_add = self.fc_a6_add(z_add).view(b_a6, c_a6)
            y_a6_mul = self.fc_a6_mul(z_mul).view(b_a6, c_a6)
        else:
            y_a1 = self.fc_a1(z).view(b_a1, c_a1, 1, 1)
            y_a1_add = self.fc_a1_add(z_add).view(b_a1, c_a1, 1, 1)
            y_a1_mul = self.fc_a1_mul(z_mul).view(b_a1, c_a1, 1, 1)
            y_a2 = self.fc_a2(z).view(b_a2, c_a2, 1, 1)
            y_a2_add = self.fc_a2_add(z_add).view(b_a2, c_a2, 1, 1)
            y_a2_mul = self.fc_a2_mul(z_mul).view(b_a2, c_a2, 1, 1)
            y_a3 = self.fc_a3(z).view(b_a3, c_a3, 1, 1)
            y_a3_add = self.fc_a3_add(z_add).view(b_a3, c_a3, 1, 1)
            y_a3_mul = self.fc_a3_mul(z_mul).view(b_a3, c_a3, 1, 1)
            y_a4 = self.fc_a4(z).view(b_a4, c_a4, 1, 1)
            y_a4_add = self.fc_a4_add(z_add).view(b_a4, c_a4, 1, 1)
            y_a4_mul = self.fc_a4_mul(z_mul).view(b_a4, c_a4, 1, 1)
            y_a5 = self.fc_a5(z).view(b_a5, c_a5, 1, 1)
            y_a5_add = self.fc_a5_add(z_add).view(b_a5, c_a5, 1, 1)
            y_a5_mul = self.fc_a5_mul(z_mul).view(b_a5, c_a5, 1, 1)
            y_a6 = self.fc_a6(z).view(b_a6, c_a6, 1, 1)
            y_a6_add = self.fc_a6_add(z_add).view(b_a6, c_a6, 1, 1)
            y_a6_mul = self.fc_a6_mul(z_mul).view(b_a6, c_a6, 1, 1)

        return torch.mul(((F.relu(y_a1)+F.relu(y_a1_add)+F.relu(y_a1_mul))).expand_as(a1),a1), \
               torch.mul(((F.relu(y_a2)+F.relu(y_a2_add)+F.relu(y_a2_mul))).expand_as(a2),a2), \
               torch.mul(((F.relu(y_a3)+F.relu(y_a3_add)+F.relu(y_a3_mul))).expand_as(a3),a3), \
               torch.mul(((F.relu(y_a4)+F.relu(y_a4_add)+F.relu(y_a4_mul))).expand_as(a4),a4), \
               torch.mul(((F.relu(y_a5)+F.relu(y_a5_add)+F.relu(y_a5_mul))).expand_as(a5),a5), \
               torch.mul(((F.relu(y_a6)+F.relu(y_a6_add)+F.relu(y_a6_mul))).expand_as(a6),a6)




def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        
        nn.BatchNorm2d(chann_out),   
        
        nn.ReLU()
    )
    return layer


def se_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        
        nn.BatchNorm2d(chann_out),
        SELayer(chann_out, 8), 
        nn.ReLU()
    )
    return layer


def attention_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        
        nn.BatchNorm2d(chann_out),
        SelfAttention(chann_out), 
        nn.ReLU()
    )
    return layer  


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_se_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ se_conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VggNet16(nn.Module):
    output_size = 512
    def __init__(self, n_classes):
        super(VggNet16, self).__init__()
        
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        
        
        self.layer6 = vgg_fc_layer(7*7*512, 256) 
        self.layer7 = vgg_fc_layer(256,64)

        
        self.layer8 = nn.Linear(64, n_classes)

    def forward(self, x, get_ha=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        vgg16_features = self.layer5(out4)
        out4 = vgg16_features.view(out4.size(0), -1)
        out5 = self.layer6(out4)
        out6=self.layer7(out5)
        out7 = self.layer8(out6)
        
        
        if get_ha:
            return out2,out3,out4,out5,out6,out7

        return out7

class SeVggNet16(nn.Module):
    output_size =512
    def __init__(self, n_classes):
        super(SeVggNet16, self).__init__()
        
        self.layer1 = vgg_se_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_se_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_se_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_se_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_se_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        
        
        self.layer6 = vgg_fc_layer(7*7*512, 256) 
        self.layer7 = vgg_fc_layer(256,64)

        
        self.layer8 = nn.Linear(64, n_classes)

    def forward(self, x, get_ha=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        vgg16_features = self.layer5(out4)
        out4 = vgg16_features.view(out4.size(0), -1)
        out5 = self.layer6(out4)
        out6=self.layer7(out5)
        out7 = self.layer8(out6)
        
        if get_ha:
            return out1,out2,out3,out4,out5,out6,out7

        return out7

class SeFusionVGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(SeFusionVGG16, self).__init__()
        
        self.g_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.g_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.g_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.g_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.g_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        
        self.g_layer6 = vgg_fc_layer(7*7*512, 768) 
        self.g_layer7 = nn.Linear(768,768)

        
        self.g_layer8 = nn.Linear(768, n_classes)

        
        self.a_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        
        self.a_layer6 = vgg_fc_layer(7*7*512, 768) 
        self.a_layer7 = nn.Linear(768,768)

        
        self.a_layer8 = nn.Linear(768, n_classes)

        
        self.fusion1 = SE_Fusion(64, 64, 4)
        self.fusion2 = SE_Fusion(128, 128, 4)
        self.fusion3 = SE_Fusion(256, 256, 4)
        self.fusion4 = SE_Fusion(512, 512, 4)
        self.fusion5 = SE_Fusion(512, 512, 4)
        self.fusion6 = SE_Fusion(256, 256, 4)
        self.fusion7 = SE_Fusion(64, 64, 4)
        self.fusion8 = SE_Fusion(n_classes, n_classes, 4)

    def forward(self, gyro, acce, get_ha=False):
        
        g_out1 = self.g_layer1(gyro)
        a_out1 = self.a_layer1(acce)
        g_out1, a_out1 = self.fusion1(g_out1, a_out1)

        g_out2 = self.g_layer2(g_out1)
        a_out2 = self.a_layer2(a_out1)
        g_out2, a_out2 = self.fusion2(g_out2, a_out2)

        g_out3 = self.g_layer3(g_out2)
        a_out3 = self.a_layer3(a_out2)
        g_out3, a_out3 = self.fusion3(g_out3, a_out3)

        g_out4 = self.g_layer4(g_out3)
        a_out4 = self.a_layer4(a_out3)
        g_out4, a_out4 = self.fusion4(g_out4, a_out4)

        g_out5 = self.g_layer5(g_out4)
        a_out5 = self.a_layer5(a_out4)
        g_out5, a_out5 = self.fusion5(g_out5, a_out5)

        
        g_out5_flat = g_out5.view(g_out5.size(0), -1)
        a_out5_flat = a_out5.view(a_out5.size(0), -1)

        g_out6 = self.g_layer6(g_out5_flat)
        a_out6 = self.a_layer6(a_out5_flat)
        g_out6, a_out6 = self.fusion6(g_out6, a_out6)

        g_out7 = self.g_layer7(g_out6)
        a_out7 = self.a_layer7(a_out6)
        g_out7, a_out7 = self.fusion7(g_out7, a_out7)

        g_out8 = self.g_layer8(g_out7)
        a_out8 = self.a_layer8(a_out7)

        if get_ha:
            return g_out1, a_out1, g_out2, a_out2, g_out3, a_out3, g_out4, a_out4, g_out5, a_out5, g_out6, a_out6, g_out7, a_out7, g_out8, a_out8

        return g_out8, a_out8

class SeFusionVGG16_Berkeley(nn.Module):
    def __init__(self, n_classes=1000):
        super(SeFusionVGG16_Berkeley, self).__init__()

        
        self.a1_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a1_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a1_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a1_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a1_layer7 = nn.Linear(300,300)
        
        self.a1_layer8 = nn.Linear(300, n_classes)

        
        self.a2_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a2_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a2_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a2_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a2_layer7 = nn.Linear(300,300)
        
        self.a2_layer8 = nn.Linear(300, n_classes)

        
        self.a3_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a3_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a3_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a3_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a3_layer7 = nn.Linear(300,300)
        
        self.a3_layer8 = nn.Linear(300, n_classes)

        
        self.a4_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a4_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a4_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a4_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a4_layer7 = nn.Linear(300,300)
        
        self.a4_layer8 = nn.Linear(300, n_classes)

        self.a5_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a5_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a5_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a5_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a5_layer7 = nn.Linear(300,300)
        
        self.a5_layer8 = nn.Linear(300, n_classes)

        self.a6_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a6_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a6_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a6_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a6_layer7 = nn.Linear(300,300)
        
        self.a6_layer8 = nn.Linear(300, n_classes)

        self.fusion1=SE_semantic_Fusion_Berkeley(64,64,64,64,64,64,12)
        self.fusion2=SE_semantic_Fusion_Berkeley(128,128,128,128,128,128,12)
        self.fusion3=SE_semantic_Fusion_Berkeley(256,256,256,256,256,256,12)
        self.fusion4=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion5=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion6=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)
        self.fusion7=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)

    def forward(self, a1, a2, a3, a4, a5, a6, get_ha=False):
        a1_out1 = self.a1_layer1(a1)
        a2_out1 = self.a2_layer1(a2)
        a3_out1 = self.a3_layer1(a3)
        a4_out1 = self.a4_layer1(a4)
        a5_out1 = self.a3_layer1(a5)
        a6_out1 = self.a4_layer1(a6)
        

        
        
        
        
        
        
        

        a1_out2 = self.a1_layer2(a1_out1)
        a2_out2 = self.a2_layer2(a2_out1)
        a3_out2 = self.a3_layer2(a3_out1)
        a4_out2 = self.a4_layer2(a4_out1)
        a5_out2 = self.a5_layer2(a5_out1)
        a6_out2 = self.a6_layer2(a6_out1)
        

        a1_out3 = self.a1_layer3(a1_out2)
        a2_out3 = self.a2_layer3(a2_out2)
        a3_out3 = self.a3_layer3(a3_out2)
        a4_out3 = self.a4_layer3(a4_out2)
        a5_out3 = self.a5_layer3(a5_out2)
        a6_out3 = self.a6_layer3(a6_out2)
        

        a1_out4 = self.a1_layer4(a1_out3)
        a2_out4 = self.a2_layer4(a2_out3)
        a3_out4 = self.a3_layer4(a3_out3)
        a4_out4 = self.a4_layer4(a4_out3)
        a5_out4 = self.a5_layer4(a5_out3)
        a6_out4 = self.a6_layer4(a6_out3)
        

        a1_out5 = self.a1_layer5(a1_out4)
        a2_out5 = self.a2_layer5(a2_out4)
        a3_out5 = self.a3_layer5(a3_out4)
        a4_out5 = self.a4_layer5(a4_out4)
        a5_out5 = self.a5_layer5(a5_out4)
        a6_out5 = self.a6_layer5(a6_out4)
        


        a1_out5_temp = a1_out5.view(a1_out5.size(0), -1)
        a2_out5_temp = a2_out5.view(a2_out5.size(0), -1)
        a3_out5_temp = a3_out5.view(a3_out5.size(0), -1)
        a4_out5_temp = a4_out5.view(a4_out5.size(0), -1)
        a5_out5_temp = a5_out5.view(a5_out5.size(0), -1)
        a6_out5_temp = a6_out5.view(a6_out5.size(0), -1)

        a1_out6 = self.a1_layer6(a1_out5_temp)
        a2_out6 = self.a2_layer6(a2_out5_temp)
        a3_out6 = self.a3_layer6(a3_out5_temp)
        a4_out6 = self.a4_layer6(a4_out5_temp)
        a5_out6 = self.a5_layer6(a5_out5_temp)
        a6_out6 = self.a6_layer6(a6_out5_temp)
        


        a1_out7 = self.a1_layer7(a1_out6)
        a2_out7 = self.a2_layer7(a2_out6)
        a3_out7 = self.a3_layer7(a3_out6)
        a4_out7 = self.a4_layer7(a4_out6)
        a5_out7 = self.a5_layer7(a5_out6)
        a6_out7 = self.a6_layer7(a6_out6)
        


        a1_out8 = self.a1_layer8(a1_out7)
        a2_out8 = self.a2_layer8(a2_out7)
        a3_out8 = self.a3_layer8(a3_out7)
        a4_out8 = self.a4_layer8(a4_out7)
        a5_out8 = self.a5_layer8(a5_out7)
        a6_out8 = self.a6_layer8(a6_out7)
        

        if get_ha:
            return a1_out1, a2_out1, a3_out1, a4_out1,a5_out1, a6_out1,\
            a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2,\
            a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3,\
            a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4,\
            a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5,\
            a1_out6, a2_out6, a3_out6, a4_out6, a5_out6, a6_out6,\
            a1_out7, a2_out7, a3_out7, a4_out7, a5_out7, a6_out7,\
            a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8

        return a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8

class SemanticFusionVGG16(nn.Module):
    def __init__(self, n_classes=1000, fusion_mode='full'):
        super(SemanticFusionVGG16, self).__init__()
        self.fusion_mode = fusion_mode
        
        self.g_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.g_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.g_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.g_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.g_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        
        self.g_layer6 = vgg_fc_layer(7*7*512, 768) 
        self.g_layer7 = nn.Linear(768,768)

        
        self.g_layer8 = nn.Linear(768, n_classes)

        
        self.a_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        
        self.a_layer6 = vgg_fc_layer(7*7*512, 768) 
        self.a_layer7 = nn.Linear(768,768)

        
        self.a_layer8 = nn.Linear(768, n_classes)

        
        self.fusion1 = SMMRF(64, mode=fusion_mode)
        self.fusion2 = SMMRF(128, mode=fusion_mode)
        self.fusion3 = SMMRF(256, mode=fusion_mode)
        self.fusion4 = SMMRF(512, mode=fusion_mode)
        self.fusion5 = SMMRF(512, mode=fusion_mode)
        self.fusion6 = SMMRF(768, mode=fusion_mode)
        self.fusion7 = SMMRF(768, mode=fusion_mode)

    def forward(self, gyro, acce, get_ha=False):
        
        g_out1 = self.g_layer1(gyro)
        a_out1 = self.a_layer1(acce)
        g_out1, a_out1 = self.fusion1(g_out1, a_out1)

        g_out2 = self.g_layer2(g_out1)
        a_out2 = self.a_layer2(a_out1)
        g_out2, a_out2 = self.fusion2(g_out2, a_out2)

        g_out3 = self.g_layer3(g_out2)
        a_out3 = self.a_layer3(a_out2)
        g_out3, a_out3 = self.fusion3(g_out3, a_out3)

        g_out4 = self.g_layer4(g_out3)
        a_out4 = self.a_layer4(a_out3)
        g_out4, a_out4 = self.fusion4(g_out4, a_out4)

        g_out5 = self.g_layer5(g_out4)
        a_out5 = self.a_layer5(a_out4)
        g_out5, a_out5 = self.fusion5(g_out5, a_out5)

        
        g_out5_flat = g_out5.view(g_out5.size(0), -1)
        a_out5_flat = a_out5.view(a_out5.size(0), -1)

        g_out6 = self.g_layer6(g_out5_flat)
        a_out6 = self.a_layer6(a_out5_flat)
        g_out6, a_out6 = self.fusion6(g_out6, a_out6)

        g_out7 = self.g_layer7(g_out6)
        a_out7 = self.a_layer7(a_out6)
        g_out7, a_out7 = self.fusion7(g_out7, a_out7)

        g_out8 = self.g_layer8(g_out7)
        a_out8 = self.a_layer8(a_out7)

        if get_ha:
            return g_out1, a_out1, g_out2, a_out2, g_out3, a_out3, g_out4, a_out4, g_out5, a_out5, g_out6, a_out6, g_out7, a_out7, g_out8, a_out8

        return g_out8, a_out8

class SemanticFusionVGG16_Berkeley(nn.Module):
    def __init__(self, n_classes=1000):
        super(SemanticFusionVGG16_Berkeley, self).__init__()

        
        self.a1_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a1_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a1_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a1_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a1_layer7 = nn.Linear(300,300)
        
        self.a1_layer8 = nn.Linear(300, n_classes)

        
        self.a2_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a2_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a2_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a2_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a2_layer7 = nn.Linear(300,300)
        
        self.a2_layer8 = nn.Linear(300, n_classes)

        
        self.a3_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a3_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a3_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a3_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a3_layer7 = nn.Linear(300,300)
        
        self.a3_layer8 = nn.Linear(300, n_classes)

        
        self.a4_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a4_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a4_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a4_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a4_layer7 = nn.Linear(300,300)
        
        self.a4_layer8 = nn.Linear(300, n_classes)

        self.a5_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a5_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a5_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a5_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a5_layer7 = nn.Linear(300,300)
        
        self.a5_layer8 = nn.Linear(300, n_classes)   

        self.a6_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a6_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a6_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        
        self.a6_layer6 = vgg_fc_layer(7*7*512, 300) 
        self.a6_layer7 = nn.Linear(300,300)
        
        self.a6_layer8 = nn.Linear(300, n_classes)       

        self.fusion1=SE_semantic_Fusion_Berkeley(64,64,64,64,64,64,12)
        self.fusion2=SE_semantic_Fusion_Berkeley(128,128,128,128,128,128,12)
        self.fusion3=SE_semantic_Fusion_Berkeley(256,256,256,256,256,256,12)
        self.fusion4=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion5=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion6=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)
        self.fusion7=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)

    def forward(self, a1, a2, a3, a4, a5, a6, get_ha=False):
        a1_out1 = self.a1_layer1(a1)
        a2_out1 = self.a2_layer1(a2)
        a3_out1 = self.a3_layer1(a3)
        a4_out1 = self.a4_layer1(a4)
        a5_out1 = self.a3_layer1(a5)
        a6_out1 = self.a4_layer1(a6)
        a1_out1, a2_out1, a3_out1, a4_out1, a5_out1, a6_out1=self.fusion1(a1_out1,a2_out1,a3_out1,a4_out1,a5_out1,a6_out1)

        
        
        
        
        
        
        

        a1_out2 = self.a1_layer2(a1_out1)
        a2_out2 = self.a2_layer2(a2_out1)
        a3_out2 = self.a3_layer2(a3_out1)
        a4_out2 = self.a4_layer2(a4_out1)
        a5_out2 = self.a5_layer2(a5_out1)
        a6_out2 = self.a6_layer2(a6_out1)
        a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2=self.fusion2(a1_out2,a2_out2,a3_out2,a4_out2,a5_out2,a6_out2)

        a1_out3 = self.a1_layer3(a1_out2)
        a2_out3 = self.a2_layer3(a2_out2)
        a3_out3 = self.a3_layer3(a3_out2)
        a4_out3 = self.a4_layer3(a4_out2)
        a5_out3 = self.a5_layer3(a5_out2)
        a6_out3 = self.a6_layer3(a6_out2)
        a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3=self.fusion3(a1_out3,a2_out3,a3_out3,a4_out3,a5_out3,a6_out3)

        a1_out4 = self.a1_layer4(a1_out3)
        a2_out4 = self.a2_layer4(a2_out3)
        a3_out4 = self.a3_layer4(a3_out3)
        a4_out4 = self.a4_layer4(a4_out3)
        a5_out4 = self.a5_layer4(a5_out3)
        a6_out4 = self.a6_layer4(a6_out3)
        a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4=self.fusion4(a1_out4,a2_out4,a3_out4,a4_out4,a5_out4,a6_out4)

        a1_out5 = self.a1_layer5(a1_out4)
        a2_out5 = self.a2_layer5(a2_out4)
        a3_out5 = self.a3_layer5(a3_out4)
        a4_out5 = self.a4_layer5(a4_out4)
        a5_out5 = self.a5_layer5(a5_out4)
        a6_out5 = self.a6_layer5(a6_out4)
        a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5=self.fusion5(a1_out5,a2_out5,a3_out5,a4_out5,a5_out5,a6_out5)


        a1_out5_temp = a1_out5.view(a1_out5.size(0), -1)
        a2_out5_temp = a2_out5.view(a2_out5.size(0), -1)
        a3_out5_temp = a3_out5.view(a3_out5.size(0), -1)
        a4_out5_temp = a4_out5.view(a4_out5.size(0), -1)
        a5_out5_temp = a5_out5.view(a5_out5.size(0), -1)
        a6_out5_temp = a6_out5.view(a6_out5.size(0), -1)

        a1_out6 = self.a1_layer6(a1_out5_temp)
        a2_out6 = self.a2_layer6(a2_out5_temp)
        a3_out6 = self.a3_layer6(a3_out5_temp)
        a4_out6 = self.a4_layer6(a4_out5_temp)
        a5_out6 = self.a5_layer6(a5_out5_temp)
        a6_out6 = self.a6_layer6(a6_out5_temp)
        a1_out6,a2_out6,a3_out6,a4_out6,a5_out6,a6_out6 = self.fusion6(a1_out6,a2_out6,a3_out6,a4_out6,a5_out6,a6_out6)


        a1_out7 = self.a1_layer7(a1_out6)
        a2_out7 = self.a2_layer7(a2_out6)
        a3_out7 = self.a3_layer7(a3_out6)
        a4_out7 = self.a4_layer7(a4_out6)
        a5_out7 = self.a5_layer7(a5_out6)
        a6_out7 = self.a6_layer7(a6_out6)
        a1_out7,a2_out7,a3_out7,a4_out7,a5_out7,a5_out7 = self.fusion7(a1_out7,a2_out7,a3_out7,a4_out7,a5_out7,a6_out7)


        a1_out8 = self.a1_layer8(a1_out7)
        a2_out8 = self.a2_layer8(a2_out7)
        a3_out8 = self.a3_layer8(a3_out7)
        a4_out8 = self.a4_layer8(a4_out7)
        a5_out8 = self.a5_layer8(a5_out7)
        a6_out8 = self.a6_layer8(a6_out7)
        

        if get_ha:
            return a1_out1, a2_out1, a3_out1, a4_out1,a5_out1, a6_out1,\
            a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2,\
            a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3,\
            a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4,\
            a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5,\
            a1_out6, a2_out6, a3_out6, a4_out6, a5_out6, a6_out6,\
            a1_out7, a2_out7, a3_out7, a4_out7, a5_out7, a6_out7,\
            a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8

        return a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8                                                             
