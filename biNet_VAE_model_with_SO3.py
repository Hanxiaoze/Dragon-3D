"""
Training overview:

Phase 0: Pretrain single-pocket VAE
    - Train PocketEncoder E_p and LatentDecoder D_l as a beta-VAE mapping Pocket -> Ligand:
        p -> E_p(p) = (mu_p, logvar_p) -> z_p (reparameterize) -> D_l(z_p) ≈ L
    - Loss = recon(MSE between D_l(z_p) and L) + beta * KL(E_p)
    - Save E_p, D_l

Phase 1: Train LigandEncoder E_l to invert D_l (freeze D_l)
    - For real ligands L:
        z_l_mu, z_l_logvar = E_l(L)
        L_rec = D_l(z_l_mu)   # optionally sample
        Loss = MSE(L_rec, L) + beta_l * KL(z_l)
    - After converge, E_l approximates inverse of D_l.
    - Save E_l

Phase 2: Train BiNet:
    - Create BiNet with encoders E_p (initialized from pretrained), fusion, decoder D_l (frozen initially), E_l (frozen initially)
    - For each iteration, sample (P_a, L_a) from dataset and also sample random P_b (could be same batch)
    - Forward: outputs = BiNet(P_a, P_b) -> L_star
    - Re-encode L_star: mu_lstar, logvar_lstar = E_l(L_star)  (E_l frozen)
    - Loss components:
        * L_match = match_loss(mu_p_a, mu_p_b, mu_lstar)   # ensure L* matches both pockets
        * L_kl = KL(mu_p_a, logvar_p_a) + KL(mu_p_b, logvar_p_b)
        * L_recon = match_loss(E_l(L_a), E_l(L_b), E_l(L_star))
        * Total: Loss = lambda_match * L_match + beta*(L_kl) + lambda_recon * L_recon
    - Training schedule:
        1) Stage A: freeze E_p, D_l and E_l; train fusion only

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    # returns per-batch averaged KL (nats)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = torch.mean(kld)
    return kl

def vae_recon_loss(pred, target):
    # target: real ligand ED if available (pseudo-label), else omitted
    return F.mse_loss(pred, target, reduction='sum')/target.shape[0]

# -------------------------
# Helper functions
# -------------------------

def cube_symmetrize(weight: torch.Tensor) -> torch.Tensor:
    """
    对卷积核在立方体的 90° 旋转群上做平均（近似 O_h 的离散旋转对称）。
    输入 weight 形状: (out_ch, in_ch, kx, ky, kz)
    返回与 weight 同形状的对称化核。
    """
    ws = []
    # xy 平面旋转
    for k in range(4):
        ws.append(torch.rot90(weight, k, dims=(2, 3)))
    # xz 平面旋转
    for k in range(4):
        ws.append(torch.rot90(weight, k, dims=(2, 4)))
    # yz 平面旋转
    for k in range(4):
        ws.append(torch.rot90(weight, k, dims=(3, 4)))

    w_stack = torch.stack(ws, dim=0)
    return w_stack.mean(dim=0)

def symmetrize_bias(bias: torch.Tensor) -> torch.Tensor:
    if bias is None:
        return None
    return bias.mean().expand_as(bias)

# ====================
# 对称卷积
# ====================
class CubeSymConv3d(nn.Conv3d):
    def forward(self, x):
        weight_sym = cube_symmetrize(self.weight).clone()
        bias_sym = symmetrize_bias(self.bias).clone() if self.bias is not None else None
        return F.conv3d(x, weight_sym, bias_sym, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

class CubeSymConvTranspose3d(nn.ConvTranspose3d):
    def forward(self, x):
        weight_sym = cube_symmetrize(self.weight).clone()
        bias_sym = symmetrize_bias(self.bias).clone() if self.bias is not None else None
        return F.conv_transpose3d(x, weight_sym, bias_sym, stride=self.stride, padding=self.padding,
                                  output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)

# -------------------------
# Pocket Encoder
# -------------------------
class PocketEncoder(nn.Module):
    def __init__(self, dim_tmp=128, drop_rate=0.2, stride=2, padding=2):
        super().__init__()
        self.dim_tmp = dim_tmp
        self.drop_rate = drop_rate
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Sequential(
            CubeSymConv3d(1, dim_tmp // 4, kernel_size=5, padding=padding),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_rate)
        )
        self.conv2 = nn.Sequential(
            CubeSymConv3d(dim_tmp // 4, dim_tmp // 2, kernel_size=4, stride=stride, padding=padding-1),
            nn.InstanceNorm3d(dim_tmp // 2),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_rate)
        )
        self.conv3 = nn.Sequential(
            CubeSymConv3d(dim_tmp // 2, dim_tmp, kernel_size=3, stride=stride, padding=padding-1),
            nn.InstanceNorm3d(dim_tmp),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_rate)
        )
        self.conv4 = nn.Sequential(
            CubeSymConv3d(dim_tmp, dim_tmp*2, kernel_size=4),
            nn.InstanceNorm3d(dim_tmp*2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(drop_rate)
        )

        self.fc1 = nn.Linear(dim_tmp*2*4*4*4, dim_tmp)
        self.fc2 = nn.Linear(dim_tmp*2*4*4*4, dim_tmp)
        self.fc3 = nn.Linear(dim_tmp, dim_tmp*3*3*3)

    def forward(self, grid):
        e1 = self.conv1(grid)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        flat = e4.view(e4.size(0), -1)
        z_mean, z_logvar = self.fc1(flat), self.fc2(flat)
        return z_mean, z_logvar

# -------------------------
# Ligand Encoder
# -------------------------
class LigandEncoder(nn.Module):
    def __init__(self, dim_tmp=128, drop_rate=0.2, stride=2, padding=2):
        super().__init__()
        self.dim_tmp = dim_tmp
        self.drop_rate = drop_rate
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Sequential(
            CubeSymConv3d(1, dim_tmp // 4, kernel_size=5, padding=padding),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_rate)
        )
        self.conv2 = nn.Sequential(
            CubeSymConv3d(dim_tmp // 4, dim_tmp // 2, kernel_size=4, stride=stride, padding=padding-1),
            nn.InstanceNorm3d(dim_tmp // 2),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_rate)
        )
        self.conv3 = nn.Sequential(
            CubeSymConv3d(dim_tmp // 2, dim_tmp, kernel_size=3, stride=stride, padding=padding-1),
            nn.InstanceNorm3d(dim_tmp),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_rate)
        )
        self.conv4 = nn.Sequential(
            CubeSymConv3d(dim_tmp, dim_tmp*2, kernel_size=4),
            nn.InstanceNorm3d(dim_tmp*2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(drop_rate)
        )

        self.fc1 = nn.Linear(dim_tmp*2*4*4*4, dim_tmp)
        self.fc2 = nn.Linear(dim_tmp*2*4*4*4, dim_tmp)
        self.fc3 = nn.Linear(dim_tmp, dim_tmp*3*3*3)

    def forward(self, grid):
        e1 = self.conv1(grid)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        flat = e4.view(e4.size(0), -1)
        z_mean, z_logvar = self.fc1(flat), self.fc2(flat)
        return z_mean, z_logvar

# -------------------------
# Latent Decoder
# -------------------------
class LatentDecoder(nn.Module):
    def __init__(self, dim_tmp=128, drop_rate=0.2, stride=2, padding=2):
        super().__init__()
        self.dim_tmp = dim_tmp
        self.stride = stride
        self.padding = padding

        self.fc3 = nn.Linear(dim_tmp, dim_tmp*3*3*3)

        self.deconv1 = nn.Sequential(
            CubeSymConvTranspose3d(dim_tmp, dim_tmp//4, kernel_size=5, stride=stride, padding=padding, output_padding=padding-1),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            CubeSymConvTranspose3d(dim_tmp//4, dim_tmp//8, kernel_size=5, stride=stride, padding=padding, output_padding=padding-1),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            CubeSymConvTranspose3d(dim_tmp//8, dim_tmp//16, kernel_size=5, stride=stride, padding=padding, output_padding=padding-1),
            nn.LeakyReLU()
        )
        self.deconv4 = nn.Sequential(
            CubeSymConvTranspose3d(dim_tmp//16, 1, kernel_size=6, stride=stride, padding=padding),
            nn.LeakyReLU()
        )

    def forward(self, z):
        d1 = self.deconv1(self.fc3(z).view(z.size(0), self.dim_tmp, 3, 3, 3))
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        d4 = self.deconv4(d3)
        return d4

# -------------------------
# Fusion Module
# -------------------------
class FusionModule(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=1)
        return self.net(z)

# -------------------------
# BiNet (two pockets -> ligand)
# -------------------------
class BiNet(nn.Module):
    def __init__(self, pocket_encoder: PocketEncoder, fusion: FusionModule, ligand_decoder: LatentDecoder, share_encoder=True):
        super().__init__()
        self.encA = pocket_encoder
        if share_encoder:
            self.encB = pocket_encoder
        else:
            import copy
            self.encB = copy.deepcopy(pocket_encoder)
        self.fusion = fusion
        self.decoder = ligand_decoder

    def forward(self, p1, p2, sample_from_mu=True):
        mu1, logvar1 = self.encA(p1)
        mu2, logvar2 = self.encB(p2)
        z1, z2 = mu1, mu2
        z_fused = self.fusion(z1, z2)
        L_star = self.decoder(z_fused)
        return {
            'mu1': mu1, 'logvar1': logvar1,
            'mu2': mu2, 'logvar2': logvar2,
            'z_fused': z_fused, 'L_star': L_star
        }

# -------------------------
# Match loss
# -------------------------
def match_loss(z_p1, z_p2, z_l_star):
    return F.mse_loss(z_l_star, z_p1, reduction='sum')/z_p1.shape[0] + \
           F.mse_loss(z_l_star, z_p2, reduction='sum')/z_p2.shape[0]

def recon_loss(z_la, z_lb, z_l_star):
    return F.mse_loss(z_l_star, z_la, reduction='sum')/z_la.shape[0] + \
           F.mse_loss(z_l_star, z_lb, reduction='sum')/z_lb.shape[0]

