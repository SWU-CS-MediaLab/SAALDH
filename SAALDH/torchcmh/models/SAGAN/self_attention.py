from torch import nn
import torch

# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim, activation=torch.relu()):
#         super(Self_Attn, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)  #
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x
#         return out, attention
#
# if __name__ == '__main__':
#     net = Self_Attn(64)
#     net = net.cuda()
#     x1 = torch.ones(64, 64)
#     x2 = torch.ones(64, 64)
#     x3 = torch.ones(64, 64)
#     x4 = torch.ones(64, 64)
#     x1 = x1.cuda()
#     x2 = x2.cuda()
#     x3 = x3.cuda()
#     x4 = x4.cuda()
#     out = net(x1, x2, x3, x4)
#     print(out.shape)
#     print(net.weight)
#     print(torch.mean(net.weight, dim=1))


class ImageSelfAttention(nn.Module):
    """ Self-attention module for CNN's feature map.
    Inspired by: Zhang et al., 2018 The self-attention mechanism in SAGAN.
    """

    def __init__(self, planes):
        super(ImageSelfAttention, self).__init__()
        inner = planes // 8
        self.conv_f = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_g = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_h = nn.Conv1d(planes, planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)
        sim_beta = torch.matmul(f.transpose(1, 2), g)
        beta = nn.functional.softmax(sim_beta, dim=1)
        o = torch.matmul(h, beta)
        return o