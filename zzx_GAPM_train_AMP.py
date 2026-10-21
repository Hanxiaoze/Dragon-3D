# train_torsion_EDlabels.py
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

import os, sys
import torch as th
import torch.nn as nn
import torch.optim as optim


from rdkit.Chem.rdMolTransforms import GetDihedralDeg
import numpy as np
from tqdm import tqdm

from utils.Utils import SetSeed, GetConfigs

import copy
import warnings
warnings.filterwarnings("ignore")
from math import pi
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from torch_geometric import loader
from torch_geometric.data import Data, DataLoader
from utils.Utils import *
from models.Mol_Generator import TorsionAnglePredictionModel
from utils.EDExtract import Voxelize, Fcalc
from scipy.spatial import KDTree
import pickle
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from torch.utils.data import random_split


# ===== 全局voxpoints & KDTree =====
GLOBAL_VOXPOINTS = None
GLOBAL_KDTREE = None

def init_global_voxpoints(center=(24.0,24.0,24.0), GridSize=12, SpacingCutoff=0.5):
    """
    初始化全局voxpoints和KDTree (只调用一次)
    """
    global GLOBAL_VOXPOINTS, GLOBAL_KDTREE    
    GLOBAL_VOXPOINTS = Voxelize((24.0,24.0,24.0), 12, 0.5)
    GLOBAL_KDTREE = KDTree(GLOBAL_VOXPOINTS)
    print(f"[INFO] Global voxpoints initialized: {GLOBAL_VOXPOINTS.shape}, KDTree ready.")


def remove_marker_atoms_with_map(mol_l):
    """
    移除 R/H 标记原子，并返回:
        - mol_new: 去掉标记后的新分子 (失败时返回 None)
        - old2new: 从 mol_l 索引到 mol_new 索引的映射字典
    """
    try:
        rw_mol = Chem.RWMol(mol_l)
        atoms_to_remove = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetSymbol() in ["R", "H"]]

        keep_idxs = [a.GetIdx() for a in mol_l.GetAtoms() if a.GetIdx() not in atoms_to_remove]

        for idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.RemoveAtom(idx)

        mol_new = rw_mol.GetMol()
        # Chem.SanitizeMol(mol_new)  # 可能报错

        old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_idxs)}
        return mol_new, old2new
    except Exception as e:
        print(f"[WARN] remove_marker_atoms failed: {e}")
        return None, {}

#=========222====================
class Mol2GraphTorsionAngle:
    def __init__(self, mol, c):
        self.mol = mol
        self.c = c   # 中心的坐标

    def _onehotencoding(self, lst, i):
        return list(map(lambda x: 1 if x == i else 0, lst))
        
    def _atom_featurization(self):
        atomic_num, atom_vector = [], []
        conformer = self.mol.GetConformer()
        positions = conformer.GetPositions()
        for idx,a in enumerate(self.mol.GetAtoms()):
            atomtype_encoding = self._onehotencoding(["C", "N", "O", "S", "F", "R"], a.GetSymbol())
            atomdegree_encoding = self._onehotencoding([1, 2, 3, 4], a.GetDegree())
            atomhybrid_encoding = self._onehotencoding(["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "OTHER"], str(a.GetHybridization()))
            atomcharge_encoding = self._onehotencoding([-3, -2, -1, 0, 1, 2, 3], a.GetFormalCharge())
            atomhydrogen_encoding = self._onehotencoding([0, 1, 2, 3, 4], a.GetTotalNumHs())
            atom_is_in_aromatic = [int(a.GetIsAromatic())]
            atom_is_in_ring = [int(a.IsInRing())]
            atom_coord = conformer.GetAtomPosition(idx)
            if self.c is not None:
                center = self.c
                atom_position = list((np.array([atom_coord.x-center[0],atom_coord.y-center[1],atom_coord.z-center[2]])+np.array([12,12,12]))/24)
                atom_vector.append(atomtype_encoding + atomdegree_encoding + atomhybrid_encoding + \
                               atomcharge_encoding + atomhydrogen_encoding + atom_is_in_aromatic + atom_is_in_ring + atom_position)
            else:
                atom_vector.append(atomtype_encoding + atomdegree_encoding + atomhybrid_encoding + \
                                   atomcharge_encoding + atomhydrogen_encoding + atom_is_in_aromatic + atom_is_in_ring)

        atom_coords = th.from_numpy(positions)
        atom_vector = th.tensor(atom_vector, dtype=th.float)
        return atom_coords, atom_vector   

    def _bond_featurization(self):
        bond_vector = []
        for b in self.mol.GetBonds():
            b1 = [b.GetBeginAtomIdx()]
            b2 = [b.GetEndAtomIdx()]
            bondtype_encoding = self._onehotencoding([1.0, 2.0, 3.0, 1.5, 0.0], b.GetBondTypeAsDouble())
            bond_is_in_ring = [int(b.IsInRing())]
            bond_is_conjugated = [int(b.GetIsConjugated())]
            bond_vector.append(b1 + b2 + bondtype_encoding + bond_is_in_ring + bond_is_conjugated)
            bond_vector.append(b2 + b1 + bondtype_encoding + bond_is_in_ring + bond_is_conjugated)
        return th.tensor(bond_vector, dtype = th.long)

    def _create_graph(self):
        bond_vector = self._bond_featurization()
        atom_coords, atom_vector = self._atom_featurization()
        edge_index, edge_attr = bond_vector[:, :2], bond_vector[:, 2:]
        edge_index = edge_index.permute(1,0)
        atom_vector = atom_vector.float()
        edge_attr = edge_attr.float()
        graph = Data(x = atom_vector, edge_index = edge_index, edge_attr = edge_attr,  pos = atom_coords)
        return graph


def get_rotatable_bond(mol):
    for bond in mol.GetBonds():
        if not bond.IsInRing() and bond.GetBondTypeAsDouble() == 1.0:
            return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    return None, None

def calc_torsion_angle(core_l, frag_l, tol=2e-3):
    """
    计算 frag 中 core 和新增片段之间的二面角
    返回: angle (float, 单位: rad)
    """
    # 去掉指示原子 R
    core, core_old2new = remove_marker_atoms_with_map(core_l)
    frag, frag_old2new = remove_marker_atoms_with_map(frag_l)

    conf_core = core.GetConformer()
    conf_frag = frag.GetConformer()

    # ----------- 找到 a1 (core_l 中 R 的唯一邻居) -----------
    R_core = [a for a in core_l.GetAtoms() if a.GetSymbol() == "R"][0]
    R_id = R_core.GetIdx()
    a1 = R_core.GetNeighbors()[0].GetIdx()
    a1 = core_old2new[a1]
    try:
        a1_coord = np.array(conf_core.GetAtomPosition(a1))
    except Exception as e:
        print(e)

    # ----------- 找到 frag 中对应的 a1_f -----------
    a1_f = None
    for atom in frag.GetAtoms():
        pos = np.array(conf_frag.GetAtomPosition(atom.GetIdx()))
        if np.linalg.norm(pos - a1_coord) < tol:
            a1_f = atom.GetIdx()
            break
    if a1_f is None:
        raise ValueError("在 frag 中找不到和 core 中 a1 坐标匹配的 a1_f 原子")

    # ----------- 找到 a2_f -----------
    core_coords = {tuple(np.round(np.array(conf_core.GetAtomPosition(a)), 3))
               for a in range(core.GetNumAtoms())}
    a2_f = None

    l = []
    for nei in frag.GetAtomWithIdx(a1_f).GetNeighbors():       
        pos = tuple(np.round(np.array(conf_frag.GetAtomPosition(nei.GetIdx())), 3))
        if pos not in core_coords:  # 不属于 core，就是新增片段
            l.append(nei.GetIdx())
    if len(l) == 1:
        a2_f = l[0]
    else:
        raise ValueError(f"在 frag 中找到新增片段的锚点原子 a2_f 为 {len(l)} 个")

    # ----------- 找到二面角四个原子 -----------
    x = [n.GetIdx() for n in frag.GetAtomWithIdx(a1_f).GetNeighbors() if n.GetIdx() != a2_f][0]
    y = [n.GetIdx() for n in frag.GetAtomWithIdx(a2_f).GetNeighbors() if n.GetIdx() != a1_f][0]

    # ----------- 计算角度 -----------
    alpha_mean = CalcAlpha(frag, a1_f, a2_f)  
    # alpha_ori = GetDihedralRad(conf_frag, x, a1_f, a2_f, y)  

    # angle = alpha_ori - alpha_mean
    return alpha_mean


def angle_to_bin(angle, num_bins=36):
    """把角度离散化到分类标签"""
    deg = np.rad2deg(angle) % 360
    bin_size = 360 / num_bins
    return int(deg // bin_size)


#========111  推理用，不含角度信息====================
class DatasetTorsionAngle(Dataset):
    def __init__(self, frags, core, center, ED):
        super(DatasetTorsionAngle, self).__init__()
        self.frags = frags
        self.core = core
        self.center = center
        self.ED = ED

    def __len__(self):
        return len(self.frags)

    def __getitem__(self, idx):
        frag2graph = Mol2GraphTorsionAngle(self.frags[idx],None)
        core2graph = Mol2GraphTorsionAngle(self.core,self.center)
        frag_graph = frag2graph._create_graph()  # 调用 _create_graph() 方法，创建图
        core_graph = core2graph._create_graph()
        temp = [core_graph, frag_graph, self.ED]
        return temp

    def get(self, idx):
        pass

    def len(self, idx):
        pass

#=========111用于训练，含有角度信息====================
class DatasetTorsionAngle_Labels(Dataset):
    """
    输入: sdf 轨迹文件夹，每条轨迹 n 个分子 → n-1 个训练样本
    输出: (core_graph, frag_graph, ED)
    PyG图和ED分别缓存到磁盘，避免重复计算
    """
    def __init__(self, sdf_trajs_dir, cache_dir="../cache_TAPM_dataset", num_bins=36):
        super().__init__()
        self.samples = []  
        self.num_bins = num_bins
        self.ED_center = (24.0, 24.0, 24.0)
        self.cache_dir = cache_dir

        # 子目录
        self.core_dir = os.path.join(cache_dir, "core_graphs")
        self.frag_dir = os.path.join(cache_dir, "frag_graphs")
        self.ED_dir = os.path.join(cache_dir, "EDs")
        os.makedirs(self.core_dir, exist_ok=True)
        os.makedirs(self.frag_dir, exist_ok=True)
        os.makedirs(self.ED_dir, exist_ok=True)

        # 缓存索引文件
        self.samples_file = os.path.join(cache_dir, "samples.pkl")

        if os.path.exists(self.samples_file):
            print(f"[INFO] 从缓存加载索引: {self.samples_file}")
            with open(self.samples_file, "rb") as f:
                self.samples = pickle.load(f)
        else:
            print(f"[INFO] 未找到索引缓存，开始生成...")
            sdf_files = [os.path.join(sdf_trajs_dir, f) for f in os.listdir(sdf_trajs_dir) if f.endswith(".sdf")]
            for sdf_file in sdf_files:
                sdf_name = os.path.splitext(os.path.basename(sdf_file))[0]
                suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
                traj_mols = [m for m in suppl if m is not None]

                for i in range(len(traj_mols) - 1):
                    core, frag = traj_mols[i], traj_mols[i+1]
                    try:
                        # torsion 角度
                        angle = calc_torsion_angle(core, frag)
                        label = angle_to_bin(angle, num_bins)

                        # 图文件名
                        core_path = os.path.join(self.core_dir, f"{sdf_name}_{i}_core.pt")
                        frag_path = os.path.join(self.frag_dir, f"{sdf_name}_{i}_frag.pt")
                        ED_path = os.path.join(self.ED_dir, f"{sdf_name}_{i}_ED.npy")

                        # === 生成 PyG 图并保存 ===
                        if not os.path.exists(core_path):                             
                            core_graph = Mol2GraphTorsionAngle(core, self.ED_center)._create_graph()
                            core_graph.y = th.tensor(label, dtype=th.long)
                            th.save(core_graph, core_path)

                        if not os.path.exists(frag_path):
                            frag_graph = Mol2GraphTorsionAngle(frag, None)._create_graph()
                            th.save(frag_graph, frag_path)

                        # === 保存 ED ===
                        if not os.path.exists(ED_path):
                            # 假设 ED 文件原始路径
                            ED_file = '../cache_ed/' + sdf_file.split('/')[-1] + '_' + sdf_file.split('_')[-1].split('.')[0] + '_0_rho.npy'
                            if not os.path.exists(ED_file):
                                print(f"[WARN] 缺失 ED 文件，跳过: {ED_file}")
                                continue
                            np.save(ED_path, np.load(ED_file))

                        # 保存索引
                        self.samples.append((core_path, frag_path, ED_path))

                    except Exception as e:
                        print(f"[WARN] 样本跳过: {e} (file={sdf_file}, step={i})")
                        continue

            # 保存索引
            with open(self.samples_file, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"[INFO] 缓存完成, 样本数={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        core_path, frag_path, ED_path = self.samples[idx]
        try:
            core_graph = th.load(core_path)
            frag_graph = th.load(frag_path)
            ED = th.tensor(np.load(ED_path), dtype=th.float32).reshape(-1, 48, 48, 48)
            return core_graph, frag_graph, ED
        except Exception as e:
            print(f"[WARN] 加载失败 idx={idx}: {e}")
            return None


def collate_fn(batch):
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    core_graphs, frag_graphs, EDs = zip(*batch)
    return core_graphs, frag_graphs, EDs


def statistic_of_model_params_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name}: {p.numel():,}")


from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir ./runs --host 0.0.0.0 --port 6006 --load_fast=false

from torch.cuda.amp import autocast, GradScaler
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

def main():
    device = th.device("cuda")
    batch_size = 64
    lr = 1e-5
    epochs = 2500
    output_dir = '../TAPM_zzx_train'
    os.makedirs(output_dir, exist_ok=True)    
    log_file = os.path.join(output_dir, "TAPM_log.txt")
    SetSeed(42)

    # 数据集和 DataLoader
    sdf_trajs_dir = '../fragment_extension_dataset/translate_and_splited_sdf_subset1'
    full_dataset = DatasetTorsionAngle_Labels(sdf_trajs_dir)

    val_ratio = 0.2
    val_len = max(1, int(len(full_dataset) * val_ratio))
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=32, pin_memory=True, prefetch_factor=2, 
                              persistent_workers=True, collate_fn=collate_fn)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=32, pin_memory=True, prefetch_factor=2, 
                              persistent_workers=True, collate_fn=collate_fn)

    # 模型、优化器、损失
    model = TorsionAnglePredictionModel().to(device)
    statistic_of_model_params_num(model)
    # model = th.compile(model, backend="aot_eager")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
    # 文件日志
    with open(log_file, "w") as f:
        f.write("Epoch\tLoss\tAcc@1\tAcc@3\tAcc@5\tval_Loss\tval_Acc@1\tval_Acc@3\tval_Acc@5\tLR\n")

    # AMP GradScaler
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc, total_acc_3, total_acc_5 = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            if batch is None or (isinstance(batch, (list, tuple)) and any(b is None for b in batch)):
                continue

            core_graphs, frag_graphs, EDs = batch
            core_batch = core_graphs.to(device)
            frag_batch = frag_graphs.to(device)
            ED_batch = EDs.to(device)

            optimizer.zero_grad()
            
            # === 自动混合精度前向计算 ===
            with autocast():
                out = model(frag_batch.edge_index, frag_batch.x, frag_batch.pos, frag_batch.edge_attr, frag_batch.batch,
                            core_batch.edge_index, core_batch.x, core_batch.pos, core_batch.edge_attr, core_batch.batch,
                            ED_batch.float())
                loss = criterion(out, core_batch.y)

            # === AMP 反向 + step ===
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 先反缩放，再裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Top-1
            pred = out.argmax(dim=1)
            correct = (pred == core_batch.y).sum().item()
            acc = correct / core_batch.y.size(0)

            # Top-3 / Top-5
            topk_vals, topk_idx = torch.topk(out, k=5, dim=1)
            correct_top3 = (topk_idx[:, :3] == core_batch.y.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 = (topk_idx[:, :5] == core_batch.y.unsqueeze(1)).any(dim=1).sum().item()
            acc_top3 = correct_top3 / out.size(0)
            acc_top5 = correct_top5 / out.size(0)

            total_acc += acc
            total_acc_3 += acc_top3
            total_acc_5 += acc_top5

            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Acc@1": f"{acc*100:.2f}%",
                "Acc@3": f"{acc_top3*100:.2f}%",
                "Acc@5": f"{acc_top5*100:.2f}%"
            })

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = total_acc / len(train_loader)
        epoch_acc_3 = total_acc_3 / len(train_loader)
        epoch_acc_5 = total_acc_5 / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc@1={epoch_acc*100:.2f}%, Acc@3={epoch_acc_3*100:.2f}%, Acc@5={epoch_acc_5*100:.2f}%, lr={lr_now:.6f}\n")

        # TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/Top1", epoch_acc, epoch)
        writer.add_scalar("Accuracy/Top3", epoch_acc_3, epoch)
        writer.add_scalar("Accuracy/Top5", epoch_acc_5, epoch)
        writer.add_scalar("LR", lr_now, epoch)
        
        scheduler.step(epoch_loss)


        model.eval()
        val_total_loss = 0
        val_total_acc, val_total_acc_3, val_total_acc_5 = 0, 0, 0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            if batch is None or (isinstance(batch, (list, tuple)) and any(b is None for b in batch)):
                continue

            core_graphs, frag_graphs, EDs = batch
            core_batch = core_graphs.to(device)
            frag_batch = frag_graphs.to(device)
            ED_batch = EDs.to(device)            
            
            # === 自动混合精度前向计算 ===
            with autocast():
                out = model(frag_batch.edge_index, frag_batch.x, frag_batch.pos, frag_batch.edge_attr, frag_batch.batch,
                            core_batch.edge_index, core_batch.x, core_batch.pos, core_batch.edge_attr, core_batch.batch,
                            ED_batch.float())
                loss = criterion(out, core_batch.y)

            val_total_loss += loss.item()

            # Top-1
            pred = out.argmax(dim=1)
            correct = (pred == core_batch.y).sum().item()
            acc = correct / core_batch.y.size(0)

            # Top-3 / Top-5
            topk_vals, topk_idx = torch.topk(out, k=5, dim=1)
            correct_top3 = (topk_idx[:, :3] == core_batch.y.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 = (topk_idx[:, :5] == core_batch.y.unsqueeze(1)).any(dim=1).sum().item()
            acc_top3 = correct_top3 / out.size(0)
            acc_top5 = correct_top5 / out.size(0)

            val_total_acc += acc
            val_total_acc_3 += acc_top3
            val_total_acc_5 += acc_top5

            pbar.set_postfix({
                "Batch val Loss": f"{loss.item():.4f}",
                "val_Acc@1": f"{acc*100:.2f}%",
                "val_Acc@3": f"{acc_top3*100:.2f}%",
                "val_Acc@5": f"{acc_top5*100:.2f}%"
            })

        val_epoch_loss = val_total_loss / len(val_loader)
        val_epoch_acc = val_total_acc / len(val_loader)
        val_epoch_acc_3 = val_total_acc_3 / len(val_loader)
        val_epoch_acc_5 = val_total_acc_5 / len(val_loader)
        

        print(f"Epoch {epoch+1}: val_Loss={val_epoch_loss:.4f}, val_Acc@1={val_epoch_acc*100:.2f}%, val_Acc@3={val_epoch_acc_3*100:.2f}%, val_Acc@5={val_epoch_acc_5*100:.2f}%\n")

        # TensorBoard
        writer.add_scalar("val_Loss/train", val_epoch_loss, epoch)
        writer.add_scalar("val_Accuracy/Top1", val_epoch_acc, epoch)
        writer.add_scalar("val_Accuracy/Top3", val_epoch_acc_3, epoch)
        writer.add_scalar("val_Accuracy/Top5", val_epoch_acc_5, epoch)        

        # 文件日志
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_acc_3:.4f}\t{epoch_acc_5:.4f}\t{val_epoch_loss:.4f}\t{val_epoch_acc:.4f}\t{val_epoch_acc_3:.4f}\t{val_epoch_acc_5:.4f}\t{lr_now:.6f}\n")

        
        if epoch % 10 == 0:            
            model_save = os.path.join(output_dir, f"tapm_epoch_{epoch}.pt")
            th.save(model.state_dict(), model_save)
            print(f"✅ Growth_Angle_pred 模型 at epoch_{epoch} 已保存到 {model_save}")
            print(f"✅ 训练日志已保存到 {log_file}")
    
    writer.close()



if __name__ == "__main__":
    main()



