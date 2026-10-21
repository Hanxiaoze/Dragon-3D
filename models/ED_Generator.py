import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class ED_generator(nn.Module):

    def __init__(self, dim_tmp=128, drop_rate=0.2, stride=2, padding=2):
        super(ED_generator, self).__init__()
        self.dim_tmp = dim_tmp
        self.drop_rate = drop_rate
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Sequential(
            #     输入通道数   输出通道数
            nn.Conv3d(1, self.dim_tmp // 4, kernel_size=5, padding=self.padding),
            nn.LeakyReLU(),
            nn.Dropout3d(self.drop_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.dim_tmp // 4, self.dim_tmp // 2, kernel_size=4, stride=self.stride, padding=self.padding-1),
            nn.InstanceNorm3d(self.dim_tmp // 2),
            nn.LeakyReLU(),
            nn.Dropout3d(self.drop_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.dim_tmp // 2, self.dim_tmp, kernel_size=3, stride=self.stride, padding=self.padding-1),
            nn.InstanceNorm3d(self.dim_tmp),
            nn.LeakyReLU(),
            nn.Dropout3d(self.drop_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(self.dim_tmp, self.dim_tmp * 2, kernel_size=4),
            nn.InstanceNorm3d(self.dim_tmp * 2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(self.drop_rate)
        )
        self.fc1 = nn.Linear(self.dim_tmp * 2 * 4 * 4 * 4, self.dim_tmp)
        self.fc2 = nn.Linear(self.dim_tmp * 2 * 4 * 4 * 4, self.dim_tmp)
        self.fc3 = nn.Linear(self.dim_tmp, self.dim_tmp * 3 * 3 * 3)
        self.deconv1 = nn.Sequential(
                            #     输入通道数   输出通道数    常用于 上采样（把小的体素网格放大）
            nn.ConvTranspose3d(self.dim_tmp, self.dim_tmp // 4, kernel_size=5, output_padding=self.padding-1, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp // 4, self.dim_tmp // 8, kernel_size=5, output_padding=self.padding-1, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp // 8, self.dim_tmp // 16, kernel_size=5, output_padding=self.padding-1, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp // 16, 1, kernel_size=6, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
    
    def encode(self, grid):
        e1 = self.conv1(grid)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        return self.fc1(e4.view(e4.size()[0], -1)), self.fc2(e4.view(e4.size()[0], -1))
        
    def sampling(self, z_mean, z_logvar):     
        # z_logvar 是模型输出的潜在变量z的 对数方差 log(σ²)   
        """
        如果我们直接 z ~ N(μ, σ²) 采样，梯度是没法回传的（随机采样不可导）。
        所以才用 重参数化技巧：
        把随机性放到 ε（标准正态噪声）里
        均值和标准差部分仍然是可微的函数，保证 梯度能反传到编码器
        """
        std = z_logvar.mul(0.5).exp_()  # 计算标准差        
        eps = Variable(std.data.new(std.size()).normal_())  # 生成标准正态分布的噪声 eps ~ N(0, I)
        # 将噪声乘以标准差并加上均值，得到采样结果
        return eps.mul(std).add_(z_mean)
    
    def decode(self, z):
        d1 = self.deconv1( self.fc3(z).view( z.size()[0], self.dim_tmp, 3, 3, 3 ) )
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        d4 = self.deconv4(d3)
        return d4
    
    def forward(self, grid):
        z_mean, z_logvar = self.encode(grid)
        z = self.sampling(z_mean, z_logvar)
        output = self.decode(z)
        return z_mean, z_logvar, output  
    

# by ZZX @ JLU

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groupnorm_safe(channels):
    # 选择一个可被 channels 整除的 group 数，fallback 到 1
    if channels % 8 == 0:
        groups = 8
    elif channels % 4 == 0:
        groups = 4
    elif channels % 2 == 0:
        groups = 2
    else:
        groups = 1
    return nn.GroupNorm(groups, channels)


def conv_block(in_ch, out_ch, stride=1, bias=False):
    """
    基础 conv -> GN -> LeakyReLU 块
    stride 可用于下采样
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias),
        _groupnorm_safe(out_ch),
        nn.LeakyReLU(inplace=True)
    )


class UNet3D_VAE_Aligned(nn.Module):
    """
    完整的通道对齐的 3D U-Net VAE（可变输入尺寸友好）。
    - dim_tmp: 基础大通道数 (建议 128 或 256)
    - latent_dim: 潜变量维度
    - pool_size: 在送入 FC 前将最深特征池化到 (pool_size^3)
    """
    def __init__(self, dim_tmp=128, latent_dim=128, pool_size=4, drop_rate=0.2):
        super().__init__()
        assert dim_tmp % 8 == 0, "dim_tmp 要能被 8 整除，便于通道分配。"

        # channel splits (严格对齐)
        self.c1 = dim_tmp // 8   # enc1 out
        self.c2 = dim_tmp // 4   # enc2 out
        self.c3 = dim_tmp // 2   # enc3 out
        self.c4 = dim_tmp        # enc4 out (deep)

        self.pool_size = pool_size
        self.latent_dim = latent_dim

        # ----- Encoder ----- #
        self.enc1 = conv_block(1, self.c1, stride=1)           # keep spatial
        self.enc2 = conv_block(self.c1, self.c2, stride=2)     # down x2
        self.enc3 = conv_block(self.c2, self.c3, stride=2)     # down x2
        self.enc4 = conv_block(self.c3, self.c4, stride=2)     # down x2

        # ----- Adaptive pool -> latent FC ----- #
        self.adaptive_pool = nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size))
        hidden = self.c4 * (pool_size ** 3)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden)

        # ----- Decoder convs (after concat) ----- #
        # we will interpolate projected h to e3.shape, then cat: (c4 + c3) -> produce c3
        self.up_conv1 = conv_block(self.c4 + self.c3, self.c3)
        # then (c3 + c2) -> c2
        self.up_conv2 = conv_block(self.c3 + self.c2, self.c2)
        # then (c2 + c1) -> c1
        self.up_conv3 = conv_block(self.c2 + self.c1, self.c1)

        self.final_conv = nn.Conv3d(self.c1, 1, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()  # 输出 [0,1]，可用 MSE 或 BCE 作为重建损失

        self.drop = nn.Dropout3d(p=drop_rate) if drop_rate > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------
    # Encoder / VAE functions
    # -------------------------
    def encode(self, x):
        """
        返回 mu, logvar, 以及用于 skip-connections 的 encoder 特征 (e1,e2,e3,e4)
        """
        # x: [B,1,D,H,W]
        e1 = self.enc1(x)             # [B, c1, D, H, W]
        e2 = self.enc2(e1)            # [B, c2, D/2, H/2, W/2]
        e3 = self.enc3(e2)            # [B, c3, D/4, H/4, W/4]
        e4 = self.enc4(e3)            # [B, c4, D/8, H/8, W/8]

        p = self.adaptive_pool(e4)    # [B, c4, pool, pool, pool]
        flat = p.view(p.size(0), -1)  # [B, hidden]
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, (e1, e2, e3, e4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        """
        z -> fc_decode -> small 3D grid -> 上采样到 enc3 尺寸并 concat e3 -> up_conv1 -> ...
        """
        e1, e2, e3, e4 = skips
        B = z.size(0)
        # project back to (B, c4, pool, pool, pool)
        h = self.fc_decode(z).view(B, self.c4, self.pool_size, self.pool_size, self.pool_size)

        # up to enc3 size, concat e3
        h = F.interpolate(h, size=e3.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, e3], dim=1)     # channels: c4 + c3
        h = self.up_conv1(h)
        h = self.drop(h)

        # up to enc2
        h = F.interpolate(h, size=e2.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, e2], dim=1)     # channels: c3 + c2
        h = self.up_conv2(h)
        h = self.drop(h)

        # up to enc1
        h = F.interpolate(h, size=e1.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, e1], dim=1)     # channels: c2 + c1
        h = self.up_conv3(h)
        h = self.drop(h)

        out = self.final_conv(h)
        out = self.output_act(out)
        return out

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return mu, logvar, recon