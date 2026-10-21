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
from models.ED_Generator import ED_generator

import warnings
warnings.filterwarnings("ignore")


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
                # parts = [p.strip() for p in line.split(",")]
                # assert len(parts) == 2, f"Bad line in {index_file}: {line}"
                # self.pairs.append((parts[0], parts[1]))
                self.pairs.append(line)
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
        # p_path, l_path = self.pairs[idx]
        # pocket = self._load_grid(p_path)    # (1,48,48,48)
        # ligand = self._load_grid(l_path)    # (1,48,48,48)

        p_path= self.pairs[idx]
        pocket = self._load_grid(p_path)    # (1,48,48,48)
       
        return pocket



@th.no_grad()
def model_infer(model, loader, device):
    model.eval()
    result = []

    pbar = tqdm(loader, desc="Infering ... ", leave=False)
    for pocket in pbar:
        pocket = pocket.to(device, non_blocking=True).float()        

        outputs = model(pocket)
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
            mu, logvar, x_rec = outputs[0], outputs[1], outputs[2]
            result.append(x_rec)
        else:
            raise RuntimeError("ED_generator.forward must return (mu, logvar, recon).")
    
    return result


# ----------------------------
# Checkpoint helpers
# ----------------------------    

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


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser("Train Pocket→Ligand ED VAE")
    parser.add_argument("--index_file", type=str, default="./pocED_2_ligED_infer_index.csv",
                        help="训练/验证数据对的索引文件（CSV/每行 pocket,ligand）")
    parser.add_argument("--val_index_file", type=str, default="",
                        help="可选：单独的验证集索引文件。若不提供，将从训练集划分。")
    parser.add_argument("--save_dir", type=str, default="../pocED_2_ligED_zzx_infer",
                        help="模型与日志保存目录")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="MSE 权重")
    parser.add_argument("--beta", type=float, default=1e-4, help="KL 权重")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="若未提供验证集文件，从训练集中划分比例")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=30, help="每 N 个 epoch 保存一次 checkpoint")
    parser.add_argument("--resume", type=str,
                        #  default="../pocED_2_ligED_zzx_train/checkpoint_best.pt",
                         default="../pocED_2_ligED_zzx_train/checkpoint_last.pt",
                        #  default='./Pocket2ED.pt',
                         help="从 checkpoint 路径恢复训练")
    parser.add_argument("--no_amp", default=True, help="禁用混合精度")    
    
    parser.add_argument("--dump_feature", type=list, default=["conv1.0", "deconv4.0", "__output__"], 
                        help="指定某层的名字（model.named_modules()里的name），提取一次 feature map")
    parser.add_argument("--dump_epoch", type=int, default=5, 
                        help="在哪个 epoch 提取 feature map（默认-1表示不提取）")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")


    # 数据集
    full_dataset = PocketLigandEDDataset(args.index_file)

    val_loader = DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )

    # 模型加载
    model = ED_generator().to(device)  # 预测 ED 生成    

    ckpt = torch.load(args.resume, map_location=device)
    print(ckpt.keys())   # 打印一下看看
    # model.load_state_dict(ckpt)
    model.load_state_dict(ckpt["model_state"])
    
    # statistic_of_model_params_num(model)
    # logger.info(f"Model params: {count_parameters(model):,}")   

    # 打印模型各层的名字
    for name, layer in model.named_modules():
        print(name, layer.__class__.__name__) 


    # TensorBoard: 添加模型结构 (取一批数据作为 dummy_input)
    try:
        dummy_val_pocket = next(iter(val_loader))
        dummy_val_pocket = dummy_val_pocket.to(device)        
        dummy_val_pocket_path = os.path.join(args.save_dir, f"dummy_infer_pocket.npy")        
        np.save(dummy_val_pocket_path, dummy_val_pocket[:].cpu().numpy())
        logger.info(f"✅Dummy_infer_pocket saved: {dummy_val_pocket_path}")
       

    except Exception as e:
        logger.warning(f"Could not add model graph to TensorBoard: {e}") 
       
    results = model_infer(model, val_loader, device)
    print('\n\n')
    print(f"{len(results)} batches results get!")
    print(f"Result shape is: {results[0].shape}")


    for i in range(0, len(args.dump_feature)):
        logger.info(f"🔍 Dump feature from layer: {args.dump_feature[i]}")
        
        fmap_val = get_feature_once(model, dummy_val_pocket, device, target_layer_name=args.dump_feature[i])
        if fmap_val is not None:
            # 保存成 .npy 文件（可用 numpy/pyvista 可视化）
            out_path = os.path.join(args.save_dir, f"feature_val_{args.dump_feature[i]}_infer.npy")
            np.save(out_path, fmap_val[:].cpu().numpy())  # 取第一个样本
            logger.info(f"✅ Feature val map saved: {out_path}")
        else:
            logger.warning(f"❌ Could not find layer {args.dump_feature[i]}")



# tensorboard --logdir=../pocED_2_ligED_zzx_train/tensorboard
if __name__ == "__main__":
    main()
