import torch
import pandas as pd

# -------------------- 自定义包装类 --------------------

# 自定义 Tensor 的 __repr__ (Torch)
original_tensor_repr = torch.Tensor.__repr__
def custom_tensor_repr(self):
    return f'{{Tensor_shape: {tuple(self.shape)}}} {original_tensor_repr(self)}'
torch.Tensor.__repr__ = custom_tensor_repr

# 自定义 DataFrame 的 __repr__ (Pandas)
original_dataframe_repr = pd.DataFrame.__repr__
def custom_dataframe_repr(self):
    return f'{{DataFrame_shape: {self.shape}}} {original_dataframe_repr(self)}'
pd.DataFrame.__repr__ = custom_dataframe_repr

import os
import sys
import time
import math
import glob
import json
import random
import logging
import argparse
import numpy as np
from typing import Tuple, List

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 你的模型：需确保该路径存在，并且 ED_generator 的 forward 返回 (mu, logvar, recon)
from models.ED_Generator import ED_generator, UNet3D_VAE_Aligned
from biNet_VAE_model_with_SO3 import *
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):    
    random.seed(seed)    
    np.random.seed(seed)    
    th.manual_seed(seed)    
    th.cuda.manual_seed_all(seed)
    # 关闭PyTorch的cudnn基准测试模式
    th.backends.cudnn.benchmark = False
    # 开启PyTorch的cudnn确定性模式
    th.backends.cudnn.deterministic = True


def setup_logger(save_dir: str) -> logging.Logger:    
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")

    # 获取一个日志记录器对象
    logger = logging.getLogger("Pocket2ED-VAE")
    # 设置日志级别为 INFO
    logger.setLevel(logging.INFO)
    # 清空日志记录器中的所有处理器
    logger.handlers.clear()

    # 设置日志格式
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 创建一个文件处理器，用于将日志  写入文件
    fh = logging.FileHandler(log_path, encoding="utf-8")
    # 设置文件处理器的日志级别为 INFO
    fh.setLevel(logging.INFO)
    # 设置文件处理器的日志格式
    fh.setFormatter(fmt)
    # 将文件处理器添加到日志记录器中
    logger.addHandler(fh)

    # 创建一个流处理器，用于将日志  输出到控制台
    sh = logging.StreamHandler(sys.stdout)
    # 设置流处理器的日志级别为 INFO
    sh.setLevel(logging.INFO)
    # 设置流处理器的日志格式
    sh.setFormatter(fmt)
    # 将流处理器添加到日志记录器中
    logger.addHandler(sh)

    return logger


def count_parameters(model: th.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Dataset
# ----------------------------
class PocketLigandEDDataset(Dataset):
    """
    读取一份 CSV/文本索引文件，每行两列(用,分隔):
        pocket_path,ligand_path

    每个 .npy 形状为 (48*48*48, 1)  或 (1,48,48,48)
    返回张量形状为 (1,48,48,48)
    """
    def __init__(self, index_file: str, dtype=th.float32):
        super().__init__()
        assert os.path.isfile(index_file), f"Index file not found: {index_file}"
        self.pairs = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                assert len(parts) == 2, f"Bad line in {index_file}: {line}"
                self.pairs.append((parts[0], parts[1]))
        self.dtype = dtype

    def __len__(self):
        return len(self.pairs)

    def _load_grid(self, path: str) -> th.Tensor:
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 2:              # (48*48*48, 1)      
            arr = arr.squeeze()        # (48*48*48,  )                  
            arr = arr.reshape((48, 48, 48))
            arr = arr[None, ...]            # (1,48,48,48)
            # print(f"Loaded grid shape: {arr.shape}")
        assert arr.ndim == 4, f"Loaded grid shape: {arr.shape} Grid must be 4D (C,D,H,W). Got {arr.shape} from {path}"
        return th.tensor(arr, dtype=self.dtype)

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor]:
        p_path, l_path = self.pairs[idx]
        pocket = self._load_grid(p_path)    # (1,48,48,48)
        ligand = self._load_grid(l_path)    # (1,48,48,48)
        return pocket, ligand


# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_ckpt(save_dir, epoch, model, optimizer, best_val, tag):
    path = os.path.join(save_dir, f"checkpoint_{tag}.pt")
    th.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val": best_val
    }, path)
    return path


def load_ckpt(path, model, optimizer=None, map_location="cuda"):
    ckpt = th.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

def statistic_of_model_params_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name}: {p.numel():,}")


# 保存 feature maps 的容器
feature_maps = {}

def get_feature_once(model, inputs, device, target_layer_name=""):
    """一次性 hook：forward 一次，提取指定层/输出的 feature map"""
    features = {}
    handles = []

    def hook(module, input, output):
        if isinstance(output, (tuple, list)):
            features[target_layer_name] = output[2].detach()
        else:
            features[target_layer_name] = output.detach()

    if target_layer_name == "__output__":
        h = model.register_forward_hook(hook)
        handles.append(h)
    else:
        for name, layer in model.named_modules():
            if name == target_layer_name:
                h = layer.register_forward_hook(hook)
                handles.append(h)

    model.eval()
    with th.no_grad():
        _ = model(inputs.to(device))

    for h in handles:
        h.remove()

    return features.get(target_layer_name, None)



# reparam trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def train_Ep_and_Dl(
    train_loader,
    val_loader,
    pocket_encoder,
    ligand_decoder,
    device="cuda",
    lr=1e-4,
    beta=1.0,
    epochs=1000,
    save_path="./checkpoints/"
):
    # === 优化器 ===
    params = list(pocket_encoder.parameters()) + list(ligand_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay = 0.001)

    # === AMP 的梯度缩放器 ===
    scaler = GradScaler()

    for epoch in range(epochs):
        pocket_encoder.train()
        ligand_decoder.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        pbar = tqdm(train_loader, desc="Train", leave=False)

        for pocket, ligand in pbar:
            pocket, ligand = pocket.to(device), ligand.to(device)

            optimizer.zero_grad()
            # ✅ 混合精度前向 & 反向
            with autocast():
                mu, logvar = pocket_encoder(pocket)
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                #     mu, logvar = pocket_encoder(pocket)
                # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                
                # now proceed
                z = reparameterize(mu, logvar)
                ligand_pred = ligand_decoder(z)

                # --- Loss ---
                recon = vae_recon_loss(ligand_pred, ligand)
                kld = kl_divergence(mu, logvar)
                loss = recon + beta * kld

            
            # ✅ 缩放反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kld.item()
            pbar.set_postfix(loss=f"{loss.item()}" if pbar.n else "0.0000")

        # --- 验证 ---
        pocket_encoder.eval()
        ligand_decoder.eval()
        with torch.no_grad():
            val_loss, val_recon, val_kl = 0, 0, 0
            for pocket, ligand in val_loader:
                pocket, ligand = pocket.to(device), ligand.to(device)
                # ✅ 验证也用 autocast（不需要 backward）
                with autocast():
                    mu, logvar = pocket_encoder(pocket)
                    z = mu
                    ligand_pred = ligand_decoder(z)
                    recon = vae_recon_loss(ligand_pred, ligand)
                    kld = kl_divergence(mu, logvar)
                    loss = recon + beta * kld

                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kld.item()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss {total_loss/len(train_loader):.4f} "
              f"(Recon {total_recon/len(train_loader):.4f}, KL {total_kl/len(train_loader):.4f}) | "
              f"Val Loss {val_loss/len(val_loader):.4f}")

        # === 保存模型 ===
        if epoch % 10 == 0:
            torch.save(pocket_encoder.state_dict(), f"{save_path}/E_p_epoch{epoch+1}.pt")
            torch.save(ligand_decoder.state_dict(), f"{save_path}/D_l_epoch{epoch+1}.pt")


def train_El_with_Dl_freeze(
    train_loader,    
    val_loader,
    Dl_ckpt_path,    # Latent Decoder 已训练权重路径
    epochs=1000,
    lr=1e-4,
    beta=1e-4,
    device="cuda",
    save_path="./"):

    # --- 加载模型 ---
    E_l = LigandEncoder().to(device)    
    D_l = LatentDecoder().to(device)

    # 加载预训练权重    
    D_l.load_state_dict(torch.load(Dl_ckpt_path, map_location=device))

    # 冻结 D_l    
    for p in D_l.parameters():
        p.requires_grad = False

    # 只训练 E_l
    optimizer = torch.optim.AdamW(E_l.parameters(), lr=lr, weight_decay=0.001)

    # 混合精度的梯度缩放器
    scaler = GradScaler()

    # --- Training ---
    for epoch in range(epochs):
        E_l.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for _ , batch in pbar:
            # batch: ligand_ED
            ligand = batch.to(device)

            with autocast():  # 开启混合精度
                # forward pass
                mu, logvar = E_l(ligand)
                z = reparameterize(mu, logvar)
                recon = D_l(z)

                # loss
                recon_loss = vae_recon_loss(recon, ligand)
                kl_loss = kl_divergence(mu, logvar)
                loss = recon_loss + beta * kl_loss

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()   # 用Scaler来backward
            scaler.step(optimizer)          # 用Scaler来step
            scaler.update()                 # 更新缩放因子            

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}"
            })

        # --- 验证 ---
        E_l.eval()
        D_l.eval()
        with torch.no_grad():
            val_loss, val_recon, val_kl = 0, 0, 0
            for _ , ligand in val_loader:
                ligand = ligand.to(device)
                with autocast():  # 验证也可以加AMP
                    mu, logvar = E_l(ligand)
                    z = mu
                    ligand_pred = D_l(z)
                    recon = vae_recon_loss(ligand_pred, ligand)
                    kld = kl_divergence(mu, logvar)
                    loss = recon + beta * kld
                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kld.item()

        print(f"Epoch {epoch+1}: "
              f"train_Loss={total_loss/len(train_loader):.4f}, "              
              f"train_Recon={total_recon/len(train_loader):.4f}, "
              f"train_KL={total_kl/len(train_loader):.4f}, \n"
              f"Epoch {epoch+1}: "
              f"val_Loss={val_loss/len(val_loader):.4f}, "
              f"val_Recon={val_recon/len(val_loader):.4f}, "
              f"val_KL={val_kl/len(val_loader):.4f}"
              )

        # 保存 checkpoint
        if epoch % 10 == 0:
            torch.save(E_l.state_dict(), f"{save_path}/E_l_epoch{epoch+1}.pt")


def train_Fusion_with_Ep_Dl_El_freeze(
    train_loader,    
    val_loader,
    Ep_ckpt_path,
    Dl_ckpt_path,    # Latent Decoder 已训练权重路径
    El_ckpt_path,
    epochs=1000,
    lr=1e-4,
    beta=1e-4,
    device="cuda",
    save_path="./"):

    # --- 加载模型 ---
    E_p = PocketEncoder().to(device)   
    D_l = LatentDecoder().to(device)
    E_l = LigandEncoder().to(device)    
    fusion = FusionModule().to(device)   
    

    # 加载预训练权重    
    E_p.load_state_dict(torch.load(Ep_ckpt_path, map_location=device))
    D_l.load_state_dict(torch.load(Dl_ckpt_path, map_location=device))
    E_l.load_state_dict(torch.load(El_ckpt_path, map_location=device))

    # 冻结 E_p    
    for p in E_p.parameters():
        p.requires_grad = False

    # 冻结 D_l    
    for p in D_l.parameters():
        p.requires_grad = False

    # 冻结 E_l    
    for p in E_l.parameters():
        p.requires_grad = False

    # 只训练 fusion
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=0.001)
    # 混合精度的梯度缩放器
    scaler = GradScaler()

    # --- Training ---
    for epoch in range(epochs):
        fusion.train()
        total_loss, total_match, total_recon, total_kl = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for p_a , l_a, p_b , l_b in pbar:
            
            pocket_a = p_a.to(device)
            ligand_a = l_a.to(device)
            pocket_b = p_b.to(device)
            ligand_b = l_b.to(device)

            with autocast():  # 开启混合精度
                # forward pass
                mu_pa, logvar_pa = E_p(pocket_a)
                mu_pb, logvar_pb = E_p(pocket_b)

                mu_la, logvar_la = E_l(ligand_a)
                mu_lb, logvar_lb = E_l(ligand_b)

                z_fuse = fusion(mu_pa, mu_pb)
                l_star = D_l(z_fuse)
                mu_lstar, logvar_lstar = E_l(l_star)

                # loss
                match_los = match_loss(mu_pa, mu_pb, mu_lstar)
                recon_los = recon_loss(mu_la, mu_lb, mu_lstar)            
                kl_los = kl_divergence(mu_lstar, logvar_lstar)
                loss = match_los + recon_los + beta*kl_los

            # backward
            optimizer.zero_grad()            
            scaler.scale(loss).backward()   # 用Scaler来backward
            scaler.step(optimizer)          # 用Scaler来step
            scaler.update()                 # 更新缩放因子

            total_loss += loss.item()
            total_match += match_los.item()
            total_recon += recon_los.item()
            total_kl += kl_los.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "match": f"{match_los.item():.4f}",
                "recon": f"{recon_los.item():.4f}",
                "kl": f"{kl_los.item():.4f}"
            })

        # --- 验证 ---
        E_p.eval()
        E_l.eval()
        D_l.eval()
        fusion.eval()

        with torch.no_grad():
            val_loss, val_match, val_recon, val_kl = 0, 0, 0, 0
            for p_a , l_a, p_b , l_b in val_loader:
                pocket_a = p_a.to(device)
                ligand_a = l_a.to(device)
                pocket_b = p_b.to(device)
                ligand_b = l_b.to(device)

                with autocast():  # 验证也可以加AMP
                    # forward pass
                    mu_pa, logvar_pa = E_p(pocket_a)
                    mu_pb, logvar_pb = E_p(pocket_b)

                    mu_la, logvar_la = E_l(ligand_a)
                    mu_lb, logvar_lb = E_l(ligand_b)

                    z_fuse = fusion(mu_pa, mu_pb)
                    l_star = D_l(z_fuse)
                    mu_lstar, logvar_lstar = E_l(l_star)

                    # loss
                    match = match_loss(mu_pa, mu_pb, mu_lstar)
                    recon = recon_loss(mu_la, mu_lb, mu_lstar)            
                    kl = kl_divergence(mu_lstar, logvar_lstar)
                    loss = match + recon + beta*kl
               
                val_loss += loss.item()
                val_match += match.item()
                val_recon += recon.item()
                val_kl += kl.item()

        print(f"Epoch {epoch+1}: "
              f"train_Loss={total_loss/len(train_loader):.4f}, "      
              f"train_Match={total_match/len(train_loader):.4f}, "        
              f"train_Recon={total_recon/len(train_loader):.4f}, "
              f"train_KL={total_kl/len(train_loader):.4f}, \n"
              f"Epoch {epoch+1}: "
              f"val_Loss={val_loss/len(val_loader):.4f}, "
              f"val_Match={val_match/len(val_loader):.4f}, "
              f"val_Recon={val_recon/len(val_loader):.4f}, "
              f"val_KL={val_kl/len(val_loader):.4f}"
              )

        # 保存 checkpoint
        if epoch % 10 == 0:
            torch.save(fusion.state_dict(), f"{save_path}/Fusion_epoch{epoch+1}.pt")

def infer_single_VAE(
    infer_loader,
    Ep_ckpt_path,
    Dl_ckpt_path,
    device="cuda",
    sample_times=10,
    save_path="./infer_sig_vae/"
):
    # --- 加载模型 ---
    E_p = PocketEncoder().to(device)   
    D_l = LatentDecoder().to(device)    

    # 加载预训练权重    
    E_p.load_state_dict(torch.load(Ep_ckpt_path, map_location=device))
    D_l.load_state_dict(torch.load(Dl_ckpt_path, map_location=device))

    # --- 推理 ---
    E_p.eval()
    D_l.eval()
    all_results = []  # 存放整个 dataset 的结果
    with torch.no_grad():
        for pocket, _ in infer_loader:  # pocket shape: [B, ...]
            pocket = pocket.to(device)
            mu, logvar = E_p(pocket)   # [B, latent_dim], [B, latent_dim]

            # 多次采样，结果堆叠到一个维度
            ligands = []
            for _ in range(sample_times):
                z = reparameterize(mu, logvar)  # [B, latent_dim]
                ligands.append(D_l(z))          # [B, C, H, W, D] or [B, ...]

            # 堆叠：得到 [sample_times, B, ...]
            ligands = torch.stack(ligands, dim=0)
            all_results.append(ligands)

    # 拼成完整 batch: [sample_times, N, ...]
    all_results = torch.cat(all_results, dim=1)
    all_results = all_results.permute(1, 0, *range(2, all_results.ndim))  # [N, sample_times, ...]
    all_results = all_results.detach().to('cpu').numpy().squeeze()
    np.save(f'{save_path}/infer_sigle_VAE_ligED.npy', all_results)
    print(f'Done single VAE infer ...\nresult shape: {all_results.shape}')
    
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import pyvista as pv
from matplotlib.colors import Normalize
from matplotlib import cm

def load_npy_as_voxelgrid(volume, threshold=0.0):
    """
    返回 PolyData 和对应的值，不直接生成 RGBA
    """
    mask = volume > threshold
    points = np.argwhere(mask)[:, [2,1,0]]  # 调整为 (x,y,z)
    values = volume[mask]

    if len(points) == 0:
        return None, None

    cloud = pv.PolyData(points)
    cloud["values"] = values  # 用 values 做标量
    return cloud, values

def volume_render_all_channels_voxel(fmap, channels=-1, threshold=0.0, voxel_size=0.6, cmap_name="coolwarm"):
    """
    fmap: torch tensor [B,C,D,H,W] 或 [C,D,H,W] 或 numpy array [C,D,H,W]
    channels: -1 表示全部 channel，否则指定 channel 索引列表或单个 int
    """
    if hasattr(fmap, "numpy"):
        fmap = fmap.cpu().numpy()

    if fmap.ndim == 5:  # [B,C,D,H,W]
        fmap = np.squeeze(fmap[0])
    elif fmap.ndim == 4:  # [C,D,H,W]
        pass
    elif fmap.ndim == 3:  # [D,H,W]
        fmap = fmap[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported fmap shape: {fmap.shape}")

    C, D, H, W = fmap.shape

    if channels == -1:
        channels_to_plot = list(range(C))
    elif isinstance(channels, int):
        channels_to_plot = [channels]
    else:
        channels_to_plot = channels

    ncols = math.ceil(math.sqrt(len(channels_to_plot)))
    nrows = math.ceil(len(channels_to_plot) / ncols)

    pl = pv.Plotter(shape=(nrows, ncols), title="Feature Maps (Voxel)")

    for idx, c in enumerate(channels_to_plot):
        row, col = divmod(idx, ncols)
        pl.subplot(row, col)

        volume = fmap[c].astype(np.float32)
        cloud, values = load_npy_as_voxelgrid(volume, threshold)
        if cloud is not None:
            cube = pv.Cube(center=(0.5,0.5,0.5),
                        x_length=voxel_size,
                        y_length=voxel_size,
                        z_length=voxel_size)
            glyph_mesh = cloud.glyph(scale=False, geom=cube)

            # 直接用 values + colormap 显示
            vmin, vmax = np.percentile(values, [1, 99])
            pl.add_mesh(glyph_mesh,
                        scalars="values",
                        cmap=cmap_name,
                        clim=[vmin, vmax],
                        opacity=0.3)

            pl.add_scalar_bar(title=f"Channel {c}", n_labels=5, vertical=True)

        pl.add_text(f"Channel {c}", font_size=10)

    pl.show()





def infer_biNet_VAE(
    infer_biNet_loader, 
    Ep_ckpt_path,
    Dl_ckpt_path,    # Latent Decoder 已训练权重路径
    El_ckpt_path,
    fusion_ckpt_path,
    device="cuda",
    sample_times=10,
    save_path="./"):

    # --- 加载模型 ---
    E_p = PocketEncoder().to(device)   
    D_l = LatentDecoder().to(device)
    E_l = LigandEncoder().to(device)    
    fusion = FusionModule().to(device)    

    total_params = (
        count_trainable_params(E_p) +
        count_trainable_params(D_l) +
        count_trainable_params(E_l) +
        count_trainable_params(fusion)
        )

    # print(f"Total trainable parameters: {total_params:,}")  

    # 加载预训练权重    
    E_p.load_state_dict(torch.load(Ep_ckpt_path, map_location=device))
    D_l.load_state_dict(torch.load(Dl_ckpt_path, map_location=device))
    E_l.load_state_dict(torch.load(El_ckpt_path, map_location=device))
    fusion.load_state_dict(torch.load(fusion_ckpt_path, map_location=device))

    # --- 推理 ---
    E_p.eval()
    E_l.eval()
    D_l.eval()
    fusion.eval()

    all_dual_targ_results = []  # 存放整个 dataset 的结果
    all_A_targ_results = []
    all_B_targ_results = []
    all_all_3_results = []
    all_all_5_results = []

    # ========== 3. 注册 hook（就在这里） ==========
    feature_maps = {}

    def save_activation(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook

    E_p.conv1[0].register_forward_hook(save_activation('E_p_conv1'))
    E_p.conv2[0].register_forward_hook(save_activation('E_p_conv2'))
    E_p.conv3[0].register_forward_hook(save_activation('E_p_conv3'))

    D_l.deconv1[0].register_forward_hook(save_activation('D_l_deconv1'))
    D_l.deconv3[0].register_forward_hook(save_activation('D_l_deconv3'))

    with torch.no_grad():        
        for p_a , p_b in infer_biNet_loader:
            pocket_a = p_a.to(device)            
            pocket_b = p_b.to(device)            

            # forward pass
            mu_pa, logvar_pa = E_p(pocket_a)
            mu_pb, logvar_pb = E_p(pocket_b)            

            z_fuse = fusion(mu_pa, mu_pb)
            l_star = D_l(z_fuse)
            mu_lstar, logvar_lstar = E_l(l_star)            
            # 多次采样，结果堆叠到一个维度
            dual_targ_ligands = []
            A_ligands = []
            B_ligands = []
            all_3 = []
            all_5 = []
            for _ in range(sample_times):
                z = reparameterize(mu_lstar, logvar_lstar)  # [B, latent_dim]
                dual_targ_ligands.append(D_l(z))            # [B, C, H, W, D] or [B, ...]
                all_3.append(D_l(z))                

                # 记录单靶点生成结果
                z_a = reparameterize(mu_pa, logvar_pa)
                # print(f'z_a std: {torch.exp(0.5 * logvar_pa)}')
                l_a = D_l(z_a)
                A_ligands.append(l_a)
                all_3.append(l_a)

                z_b = reparameterize(mu_pb, logvar_pb)
                l_b = D_l(z_b)
                B_ligands.append(l_b)
                all_3.append(l_b)

                all_5.extend([pocket_a, pocket_b, l_a, l_b, D_l(z)])


            # 堆叠：得到 [sample_times, B, ...]
            dual_targ_ligands = torch.stack(dual_targ_ligands, dim=0)
            all_dual_targ_results.append(dual_targ_ligands)

            A_ligands = torch.stack(A_ligands, dim=0)
            all_A_targ_results.append(A_ligands)
            B_ligands = torch.stack(B_ligands, dim=0)
            all_B_targ_results.append(B_ligands)
            all_3 = torch.stack(all_3, dim=0)
            all_all_3_results.append(all_3)  # [sample_times*3, B, ...]
            all_5 = torch.stack(all_5, dim=0)
            all_all_5_results.append(all_5)  # [sample_times*5, B, ...]

    # ========== 5. 现在你就可以用 feature_maps ==========
    volume_render_all_channels_voxel(feature_maps['E_p_conv1'])
    volume_render_all_channels_voxel(feature_maps['E_p_conv2'])
    volume_render_all_channels_voxel(feature_maps['E_p_conv3'])
    volume_render_all_channels_voxel(feature_maps['D_l_deconv1'])
    volume_render_all_channels_voxel(feature_maps['D_l_deconv3'])
    

    # 拼成完整 batch: [sample_times, N, ...]
    all_dual_targ_results = torch.cat(all_dual_targ_results, dim=1)
    all_dual_targ_results = all_dual_targ_results.permute(1, 0, *range(2, all_dual_targ_results.ndim))  # [N, sample_times, ...]
    all_dual_targ_results = all_dual_targ_results.detach().to('cpu').numpy().squeeze()
    np.save(f'{save_path}/infer_biNet_VAE_ligED.npy', all_dual_targ_results)
    print(f'Done biNet VAE infer ...\nresult shape: {all_dual_targ_results.shape}')

    all_A_targ_results = torch.cat(all_A_targ_results, dim=1)
    all_A_targ_results = all_A_targ_results.permute(1, 0, *range(2, all_A_targ_results.ndim))
    all_A_targ_results = all_A_targ_results.detach().to('cpu').numpy().squeeze()
    np.save(f'{save_path}/infer_sigA_VAE_ligED.npy', all_A_targ_results)
    print(f'Done sigA VAE infer ...\nresult shape: {all_A_targ_results.shape}')

    all_B_targ_results = torch.cat(all_B_targ_results, dim=1)
    all_B_targ_results = all_B_targ_results.permute(1, 0, *range(2, all_B_targ_results.ndim))
    all_B_targ_results = all_B_targ_results.detach().to('cpu').numpy().squeeze()
    np.save(f'{save_path}/infer_sigB_VAE_ligED.npy', all_B_targ_results)
    print(f'Done sigB VAE infer ...\nresult shape: {all_B_targ_results.shape}')

   
    all_all_3_results = torch.cat(all_all_3_results, dim=0)
    all_all_3_results = all_all_3_results.permute(1, 0, *range(2, all_all_3_results.ndim))
    all_all_3_results = all_all_3_results.detach().to('cpu').numpy().squeeze()
    np.save(f'{save_path}/infer_biNet_dualAB_VAE_ligED.npy', all_all_3_results)
    print(f'Done biNet dualAB VAE infer ...\nresult shape: {all_all_3_results.shape}')

       
    all_all_5_results = torch.cat(all_all_5_results, dim=0)
    all_all_5_results = all_all_5_results.permute(1, 0, *range(2, all_all_5_results.ndim))
    all_all_5_results = all_all_5_results.detach().to('cpu').numpy().squeeze()
    np.save(f'{save_path}/infer_biNet_pocA_pocB_ligA_ligB_dualAB_VAE_ligED.npy', all_all_5_results)
    print(f'Done biNet pocA_pocB_ligA_ligB_dualAB VAE infer ...\nresult shape: {all_all_5_results.shape}')

    return all_dual_targ_results




def collate_pairs(batch):
    # batch 里本来是一堆 (p, l)
    ps, ls = zip(*batch)
    n = len(ps)
    # 随机打乱索引作为 pairing
    idx_perm = torch.randperm(n)
    p_a = torch.stack(ps)
    l_a = torch.stack(ls)
    p_b = torch.stack([ps[j] for j in idx_perm])
    l_b = torch.stack([ls[j] for j in idx_perm])
    return p_a, l_a, p_b, l_b

# ----------------------------
# Main
# ----------------------------
from torch.utils.tensorboard import SummaryWriter
from skimage import measure

def main():
    parser = argparse.ArgumentParser("Train Pocket→Ligand ED VAE")
    parser.add_argument("--index_file", type=str, default="./new_pocED_2_ligED_train_index.csv",
                        help="训练/验证数据对的索引文件（CSV/每行 pocket,ligand）")
    parser.add_argument("--val_index_file", type=str, default="",
                        help="可选：单独的验证集索引文件。若不提供，将从训练集划分。")
    parser.add_argument("--save_dir", type=str, default="../pocED_2_ligED_zzx_train_biNet",
                        help="模型与日志保存目录")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="MSE 权重")
    parser.add_argument("--beta", type=float, default=1e-4, help="KL 权重")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="若未提供验证集文件，从训练集中划分比例")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=30, help="每 N 个 epoch 保存一次 checkpoint")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 路径恢复训练")
    parser.add_argument("--no_amp", default=False, help="禁用混合精度")    
    
    parser.add_argument("--dump_feature", type=list, 
                        # default=["conv1.0", "deconv4.0", "__output__"], 
                        default=["enc1.0", "up_conv3.0", "__output__"], 
                        help="指定某层的名字（model.named_modules()里的name），提取一次 feature map")
    parser.add_argument("--dump_epoch", type=int, default=5, 
                        help="在哪个 epoch 提取 feature map（默认-1表示不提取）")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # TensorBoard
    tb_dir = os.path.join(args.save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    # 数据集
    full_dataset = PocketLigandEDDataset(args.index_file)
    if args.val_index_file and os.path.isfile(args.val_index_file):
        train_dataset = full_dataset
        val_dataset = PocketLigandEDDataset(args.val_index_file)
    else:
        val_len = max(1, int(len(full_dataset) * args.val_ratio))
        train_len = len(full_dataset) - val_len
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader_sig_VAE = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    val_loader_sig_VAE = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    pocket_encoder = PocketEncoder().to(device)
    ligand_decoder = LatentDecoder().to(device)

    # train_Ep_and_Dl(train_loader_sig_VAE, val_loader_sig_VAE, pocket_encoder, ligand_decoder, device="cuda",
    #     lr=1e-4, beta=0.0001, epochs=1000, save_path=f"{args.save_dir}")

    # train_El_with_Dl_freeze(train_loader_sig_VAE, val_loader_sig_VAE, f"{args.save_dir}/D_l_epoch991.pt",
    #                         1000, 1e-4, 1e-4, "cuda", f"{args.save_dir}")
    
    # train_loader_dual_VAE = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0), collate_fn=collate_pairs
    # )
    # val_loader_dual_VAE = DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0), collate_fn=collate_pairs
    # )

    # train_Fusion_with_Ep_Dl_El_freeze(train_loader_dual_VAE, val_loader_dual_VAE, f"{args.save_dir}/E_p_epoch991.pt",
    #                                   f"{args.save_dir}/D_l_epoch991.pt", f"{args.save_dir}/E_l_epoch341.pt",
    #                                   1000, 1e-4, 1e-4, "cuda", f"{args.save_dir}")
    
    # sig_infer_dataset = PocketLigandEDDataset('./sig_infer_VAE.csv')
    # sig_infer_VAE_loader = DataLoader(
    #     sig_infer_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    # )
    # # infer_single_VAE(sig_infer_VAE_loader, f"{args.save_dir}/E_p_epoch361.pt", f"{args.save_dir}/D_l_epoch361.pt",
    # #                  "cuda", 3, f"../pocED_2_ligED_zzx_train_biNet_infer")
    
    # biNet_infer_dataset = PocketLigandEDDataset('./sample_biNet_infer_for_simi_comput.csv')
    biNet_infer_dataset = PocketLigandEDDataset('cluster_analysis_validate_pockED/cluster_centers_pairs.csv')
    infer_biNet_VAE_loader = DataLoader(
        biNet_infer_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
    infer_biNet_VAE(infer_biNet_VAE_loader, f"{args.save_dir}/E_p_epoch991.pt",
                    f"{args.save_dir}/D_l_epoch991.pt", f"{args.save_dir}/E_l_epoch341.pt",
                    f"{args.save_dir}/Fusion_epoch301.pt","cuda", 1, save_path=f"./cluster_analysis_validate_pockED" )
    




# tensorboard --logdir=../pocED_2_ligED_zzx_train/tensorboard
if __name__ == "__main__":
    main()
