# train_growthpoint_EDlabels.py

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
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree
from tqdm import tqdm
from models.Mol_Generator import GrowthPointPredictionModel, TorsionAnglePredictionModel
from utils.DataProcessing import Mol2GraphGrowthPoint
from utils.EDExtract import Voxelize, Fcalc, FcalcPdb
from torch.utils.data import random_split

from utils.Utils import SetSeed, GetConfigs
import pickle
import os
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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


def _mp_init_global_voxpoints(center=(24.0, 24.0, 24.0), GridSize=12, SpacingCutoff=0.5):
    # 进程池 initializer：在每个worker进程里初始化全局 voxpoints & KDTree
    init_global_voxpoints(center=center, GridSize=GridSize, SpacingCutoff=SpacingCutoff)

def _ensure_global_voxpoints(center=(24.0, 24.0, 24.0), GridSize=12, SpacingCutoff=0.5):
    # 懒初始化（防止没用initializer时仍然可用）
    global GLOBAL_VOXPOINTS, GLOBAL_KDTREE
    if GLOBAL_VOXPOINTS is None or GLOBAL_KDTREE is None:
        init_global_voxpoints(center=(24.0, 24.0, 24.0), GridSize=12, SpacingCutoff=0.5)
  


def get_centroid(mol):
    """计算分子质心"""
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords.mean(axis=0)

def zzx_lig_ed_calc_0(mol, tmpdir=None, resolution=2.0):
    if mol.GetNumConformers() == 0:
        raise ValueError("输入的 mol 没有3D构象，请先用 AllChem.EmbedMolecule 生成。")

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()

    pdbf = os.path.join(tmpdir, "lig.pdb")
    Chem.MolToPDBFile(mol, pdbf)

    # 补 CRYST1
    with open(pdbf, "r") as f:
        pdb_lines = f.readlines()
    if not any(line.startswith("CRYST1") for line in pdb_lines):
        pdb_lines.insert(0, "CRYST1   90.000   90.000   90.000  90.00  90.00  90.00 P 1           1\n")
    with open(pdbf, "w") as f:
        f.writelines(pdb_lines)

    # 使用全局 voxpoints
    global GLOBAL_VOXPOINTS
    assert GLOBAL_VOXPOINTS is not None, "GLOBAL_VOXPOINTS 未初始化"
    rho_fcalc = Fcalc(pdbf, GLOBAL_VOXPOINTS, resolution=resolution)
    rho_fcalc = np.array(rho_fcalc).reshape(-1, 1)

    if GLOBAL_VOXPOINTS.shape[0] != rho_fcalc.shape[0]:
        raise ValueError(f"Mismatch: voxpoints={GLOBAL_VOXPOINTS.shape[0]}, rho={rho_fcalc.shape[0]}")

    return rho_fcalc

def zzx_lig_ed_calc(mol, tmpdir=None, resolution=2.0):
    if mol.GetNumConformers() == 0:
        raise ValueError("输入的 mol 没有3D构象，请先用 AllChem.EmbedMolecule 生成。")

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()

    pdbf = os.path.join(tmpdir, "lig.pdb")
    Chem.MolToPDBFile(mol, pdbf)

    voxpoints, rho_fcalc_pdb = FcalcPdb(pdbf, 24.0, 24.0, 24.0, tmpdir)

    return rho_fcalc_pdb



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


# -------------------- Dataset 333--------------------
class DatasetGrowthPointWithLabels:
    def __init__(self, mols_list, traj_index_list, cache_ed, cache_graph_dir="../cache_graph_GPPM"):   #  这里输入还是脏分子
        """
        Args:
            mols_list: 轨迹分子列表
            traj_index_list: 每个分子属于哪条轨迹
            cache_ed: {traj_idx: rho_path}
            cache_graph_dir: 计算并缓存图
        """
        self.mols = mols_list
        self.traj_index_list = traj_index_list
        self.cache_ed = cache_ed
        self.cache_graph_dir = cache_graph_dir
        os.makedirs(cache_graph_dir, exist_ok=True)

        assert GLOBAL_VOXPOINTS is not None, "请先调用 init_global_voxpoints()"
        assert GLOBAL_KDTREE is not None, "请先调用 init_global_voxpoints()"

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        cache_graph_file = os.path.join(self.cache_graph_dir, f"{idx}.pt")
        if os.path.exists(cache_graph_file):
            return torch.load(cache_graph_file)
        
        # === 原始图构建逻辑 ===
        mol_l = self.mols[idx]
        traj_idx = self.traj_index_list[idx]
        rho_path = self.cache_ed[traj_idx]

        # 加载 rho (float16 → float32)
        rho = np.load(rho_path, mmap_mode="r").astype(np.float32)
        EDgrid = np.hstack([GLOBAL_VOXPOINTS, rho])  # (N,4)

        # 转 torch.Tensor
        coords = torch.tensor(EDgrid[:, :3], dtype=torch.float32)  # 没用了
        edval  = torch.tensor(EDgrid[:, 3], dtype=torch.float32)

        # 处理 R/H 标记
        mol_new, old2new = remove_marker_atoms_with_map(mol_l)        

        y = torch.full((mol_new.GetNumAtoms(),), 2, dtype=torch.long)
        for atom in mol_l.GetAtoms():
            sym = atom.GetSymbol()
            if sym in ["R", "H"]:
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    nei_idx = neighbors[0].GetIdx()
                    if nei_idx in old2new:
                        new_idx = old2new[nei_idx]
                        y[new_idx] = 0 if sym == "R" else 1    # 先在R上面生长，下一步再在H上面生长
        # mol_new = translate_mol_to_center(mol_new, target_center=(24.0,24.0,24.0))
        graph = Mol2GraphGrowthPoint(mol_new, edval, GLOBAL_KDTREE)._create_graph()
        graph.y = y

        # === 存缓存 ===
        torch.save(graph, cache_graph_file)
        
        return graph


# -------------------- 并行计算 ED 222 --------------------
def _process_trajectory_compute_ed(traj_mols, file_tag, cache_ed_dir, traj_name, resolution=2.0):
    _ensure_global_voxpoints()
    ed_path = os.path.join(cache_ed_dir, f"{file_tag}_{traj_name}_rho.npy")

    # ---- 始终先过滤掉坏分子 ----
    mol_list = []
    clean_mol_list = []
    for mol in traj_mols:
        clean_mol, _ = remove_marker_atoms_with_map(mol)  
        if clean_mol is not None:
            mol_list.append(mol)   # 脏分子
            clean_mol_list.append(clean_mol)   # 干净分子

    if not mol_list:
        print(f"[WARN] 轨迹 {file_tag}:{traj_name} 没有可用分子，跳过。")
        return None

    # ---- 如果已有缓存，直接返回 ----
    if os.path.exists(ed_path):
        return {"mol_list": mol_list, "rho_path": ed_path}

    # ---- 否则计算并存储 rho ----
    last_mol = clean_mol_list[-1]  # ✅ 注意这里用干净分子
    try:
        last_mol_h = Chem.AddHs(last_mol, addCoords=True)
        # last_mol_h = translate_mol_to_center(last_mol_h, target_center=(24.0,24.0,24.0))
        rho = zzx_lig_ed_calc(last_mol_h, resolution=resolution)
    except Exception as e:
        print(f"[WARN] ED计算失败 {file_tag}:{traj_name}: {e}")
        return None

    np.save(ed_path, rho.astype(np.float16))
    return {"mol_list": mol_list, "rho_path": ed_path}  # 脏分子



from concurrent.futures import ProcessPoolExecutor, as_completed
# -------------------- 加载数据集111 --------------------
def load_grow_traj_dataset_from_sdf_parallel(
    sdf_dir,
    cache_ed_dir="../cache_ed",
    num_workers=32,
    skip_log="skipped.log",
    resolution=2.0,
):
    """
    并行加载 grow 轨迹数据，预处理阶段生成 rho-only 缓存。
    """
    os.makedirs(cache_ed_dir, exist_ok=True)
    if os.path.exists(skip_log):
        os.remove(skip_log)

    # 主进程也初始化一次（给 Dataset 用）
    init_global_voxpoints(center=(24, 24, 24), GridSize=12, SpacingCutoff=0.5)


    # 1. 收集轨迹
    all_trajs = []
    for f in os.listdir(sdf_dir):
        if not f.endswith(".sdf"):
            continue
        sdf_path = os.path.join(sdf_dir, f)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        if not suppl:
            continue

        current_traj, first_mol_name = [], None
        for mol in suppl:
            if mol is None:
                continue
            if first_mol_name is None and mol.HasProp("_Name"):
                first_mol_name = mol.GetProp("_Name")
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            # 轨迹分隔条件：名字以 _0 结尾
            if name and name.endswith("_0") and current_traj:
                all_trajs.append((current_traj, f, first_mol_name))
                current_traj, first_mol_name = [], mol.GetProp("_Name")
            current_traj.append(mol)
        if current_traj:
            all_trajs.append((current_traj, f, first_mol_name))

    print(f"[INFO] Collected {len(all_trajs)} trajectories. Start parallel ED computation...")

    mols_list, traj_index_list = [], []
    rho_cache, skipped_trajs = {}, []

    # 2. 并行处理
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_mp_init_global_voxpoints) as executor:
        # 字典推导式：
        future_to_traj = {
            executor.submit(
                _process_trajectory_compute_ed, traj_mols, file_tag, cache_ed_dir, traj_name, resolution
            ): (traj_mols, file_tag, traj_name)
            for traj_mols, file_tag, traj_name in all_trajs   # 输入： all_trajs 还是脏分子，但是 _process_trajectory_compute_ed 返回的还是脏分子
        }                                                                       # return {"mol_list": mol_list, "rho_path": ed_path}
        for traj_idx, future in enumerate(as_completed(future_to_traj)):
            traj_mols, file_tag, traj_name = future_to_traj[future]
            try:
                result = future.result()
                if result is None:
                    skipped_trajs.append(f"{file_tag}:{traj_name}")
                    continue
                rho_cache[traj_idx] = result["rho_path"]
                for mol in result["mol_list"]:
                    mols_list.append(mol)
                    traj_index_list.append(traj_idx)
            except Exception as e:
                print(f"[WARN] trajectory {file_tag}:{traj_name} failed: {e}")
                skipped_trajs.append(f"{file_tag}:{traj_name}")

    # 3. 记录跳过的轨迹
    if skipped_trajs:
        with open(skip_log, "w") as f:
            for t in skipped_trajs:
                f.write(t + "\n")
        print(f"[INFO] {len(skipped_trajs)} trajectories skipped. See {skip_log}")

    print(f"[INFO] Dataset prepared: {len(mols_list)} molecules across {len(rho_cache)} trajectories.")

    return DatasetGrowthPointWithLabels(mols_list, traj_index_list, rho_cache)

def statistic_of_model_params_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name}: {p.numel():,}")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():    
    device = th.device("cuda")
    batch_size = 160
    lr = 1e-3
    epochs = 2500
    output_dir = '../gppm_zzx_train'
    os.makedirs(output_dir, exist_ok=True)    
    SetSeed(42)    

    # === 2. 加载训练分子 ===
    sdf_dir = "../fragment_extension_dataset/translate_and_splited_sdf_subset1"  # 存放训练分子 
    full_dataset = load_grow_traj_dataset_from_sdf_parallel(sdf_dir)
    val_ratio = 0.2
    val_len = max(1, int(len(full_dataset) * val_ratio))
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32,   # 按 CPU 核数调
                                pin_memory=True, # CUDA 拷贝更快
                                prefetch_factor=2)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=32,   # 按 CPU 核数调
                                pin_memory=True, # CUDA 拷贝更快
                                prefetch_factor=2)

    # === 3. 模型、损失、优化器 ===
    model = GrowthPointPredictionModel().to(device)    
    # statistic_of_model_params_num(model)
    total_params = count_trainable_params(model)      
    # print(f"Total trainable parameters: {total_params:,}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6, verbose=True
    )

    # 已知分布
    class_counts = torch.tensor([0.06, 0.03, 0.91])  # 各类标记的概率占比
    # 用倒数作为权重（可再归一化）
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_val_loss = 0
        total_val_acc = 0
        # === 4. 训练 ===
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for graph in pbar:
            if len(graph) == 0:
                continue
            graph = graph.to(device)            
            out = model(graph.edge_index, graph.x, graph.pos, graph.edge_attr)
            loss = criterion(out, graph.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print("out.shape:", out.shape)       # 期望: [N_atoms, num_classes]
            # print("graph.y.shape:", graph.y.shape)
            # print("y unique 是否整型？:", torch.unique(graph.y))
            # print("y dtype:", graph.y.dtype)
            pred = out.argmax(dim=1)         # [N_atoms]
            correct = (pred == graph.y).sum().item()
            acc = correct / graph.y.size(0)
            total_acc += acc
            
            # 在进度条后面动态显示
            pbar.set_postfix({"Batch train Loss": f"{loss.item():.4f}", "Batch train Accuracy": f"{acc*100:.2f}%"})
        
        epoch_loss = total_loss / len(train_loader)   # 除以batch数
        epoch_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1}: Epoch train Loss={epoch_loss:.4f}, Epoch train Accuracy={epoch_acc*100:.2f}%, lr={lr}\n")
        scheduler.step(epoch_loss)

        pbar = tqdm(val_loader, desc="Eval ", leave=False)
        # === 5. 验证 ===
        model.eval()
        for graph in pbar:
            if len(graph) == 0:
                continue
            graph = graph.to(device)            
            out = model(graph.edge_index, graph.x, graph.pos, graph.edge_attr)
            loss = criterion(out, graph.y)           
            
            total_val_loss += loss.item()            
            pred = out.argmax(dim=1)         # [N_atoms]
            correct = (pred == graph.y).sum().item()
            acc = correct / graph.y.size(0)
            total_val_acc += acc
            
            # 在进度条后面动态显示
            pbar.set_postfix({"Batch val Loss": f"{loss.item():.4f}", "Batch val Accuracy": f"{acc*100:.2f}%"})
        
        epoch_val_loss = total_val_loss / len(val_loader)   # 除以batch数
        epoch_val_acc = total_val_acc / len(val_loader)
        print(f"Epoch {epoch+1}: Epoch val Loss={epoch_val_loss:.4f}, Epoch val Accuracy={epoch_val_acc*100:.2f}%, lr={lr}\n")
        
        if epoch%30 == 0:
            th.save(model.state_dict(), f'../gppm_zzx_train/gppm_epoch{epoch}.pt')
            print(f"✅ GrowthPoint 模型已保存到 ../gppm_zzx_train/gppm_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
