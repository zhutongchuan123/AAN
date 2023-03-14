import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

import sys
sys.path.append('global_module/')
from activation import mish, gelu, gelu_new, swish

#SA
class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

    # if __name__ == '__main__':
    #     input = torch.randn(50, 512, 7, 7)
    #     se = ShuffleAttention(channel=512, G=8)
    #     output = se(input)
    #     print(output.shape)

import torch.nn as nn

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)#16, 49, 128

        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)#16, 7, 7, 128
        # print('x_att_permute.shape:', x_att_permute.shape)
        # print('x_att_permute.shape:', x_att_permute.shape)
        # return
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


    # if __name__ == '__main__':
    #     x = torch.randn(1, 64, 32, 48)
    #     b, c, h, w = x.shape
    #     net = GAM_Attention(in_channels=c, out_channels=c)
    #     y = net(x)

#保留GAM的通道注意力，使用sa的空间注意力，mish激活函数
class GAM4(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8, rate=4):
        super().__init__()
        self.G=G
        self.channel=channel#128
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()

        # super(GAM_Attention, self).__init__()
        # print('channel // (2 * G) / rate:', channel // (2 * G) / rate)
        # print('channel // (2 * G):', channel // (2 * G))

        self.channel_attention = nn.Sequential(
            nn.Linear(channel // (2 * G), int(channel // (2 * G) / rate)),
            mish(),
            nn.Linear(int(channel // (2 * G) / rate), channel // (2 * G))
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channel // (2 * G), int(channel // (2 * G) / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(channel // (2 * G) / rate)),
            mish(),
            nn.Conv2d(int(channel // (2 * G) / rate), channel // (2 * G), kernel_size=7, padding=3),
            nn.BatchNorm2d(channel // (2 * G))
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        #print('x.shape:', x.shape)
        b, c, h, w = x.size() #
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w


        # #print('x.type:', x.type)
        # print('x.shape:', x.shape)
        # #channel_split
        # x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w,,,128, 8, 7, 7
        # print('x_0.shape:', x_0.shape)
        # print('x_1.shape:', x_1.shape)
        # return
        #x = self.channel_shuffle(x, 2)
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w,,,128, 8, 7, 7


        #channel attention
        x_permute = x_0.permute(0, 2, 3, 1).view(b*self.G, -1, c//(2*self.G)) #128*8,49,1
        # print('x_permute.shape:', x_permute.shape)
        # return
        # print('x_permute.shape:', x_permute.shape)
        # print('b*self.G:', b*self.G)
        # print('c//(2*self.G):', c//(2*self.G))
        x_att_permute = self.channel_attention(x_permute).view(b*self.G, h, w, c//(2*self.G))
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # print('x_channel_att[1]:', x_channel_att[1])
        # print('x_channel_att[1].shape:', x_channel_att[1].shape)
        # return
        x_channel = x_0 * self.sigmoid(x_channel_att)

        # x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        # x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        # x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        # #x_1 = x_1.permute(0, 2, 3, 1).view(b*self.G, -1, c//(2*self.G))
        # # x_spatial_att = self.spatial_attention(x_1).view(b*self.G, c//(2*self.G), h, w)
        # x_spatial_att = self.spatial_attention(x_1).sigmoid()
        # x_spatial = x_1 * x_spatial_att

        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        # print('x_spatial[1]:', x_spatial[1])
        # return
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out = out.contiguous().view(b,-1,h,w)
        # print('out.shape:', out.shape)
        # return


        return out

    # if __name__ == '__main__':
    #     input = torch.randn(50, 512, 7, 7)
    #     se = ShuffleAttention(channel=512, G=8)
    #     output = se(input)
    #     print(output.shape)

#自注意力
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out