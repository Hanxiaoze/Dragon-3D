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


import os, sys, pickle, json
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_sort_pool
import torch.nn.functional as F
from torch.utils.data import random_split

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from collections import Counter
import numpy as np

import warnings
warnings.filterwarnings("ignore")



#=========222构建分子特征图===========
class Mol2Graph_for_frag_pred:
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

#=========111用于训练，含有图信息，根据分子当前分子结构，预测分子即将连接生长上的片段，训练数据为很多条分子生长的轨迹========
class Dataset_Grow_Frag_Labels(Dataset):
    """
    输入: sdf 轨迹文件夹，每条轨迹 n 个分子 → n-1 个训练样本
    每个样本的 label = 新生长的 fragment (基于坐标差分)
    输出: (core_graph, frag_graph, ED)
    """

    def __init__(self, sdf_trajs_dir, cache_dir="../cache_GFPM_dataset"):
        super().__init__()
        self.samples = []
        self.ED_center = (24.0, 24.0, 24.0)
        self.cache_dir = cache_dir

        # 子目录
        self.core_dir = os.path.join(cache_dir, "core_graphs")        
        os.makedirs(self.core_dir, exist_ok=True)        

        # fragment 词表文件
        self.vocab_file = os.path.join(cache_dir, "frag_vocab.json")
        # 缓存索引和训练文件path
        self.samples_file = os.path.join(cache_dir, "samples.pkl")

        # 如果已有缓存，直接加载
        if os.path.exists(self.samples_file) and os.path.exists(self.vocab_file):
            print(f"[INFO] 从缓存加载索引: {self.samples_file}")
            with open(self.samples_file, "rb") as f:
                self.samples = pickle.load(f)
            with open(self.vocab_file, "r") as f:
                self.frag_vocab = json.load(f)
        else:
            print(f"[INFO] 未找到缓存，开始生成...")
            self.frag_vocab, self.samples = self._build_vocab_and_samples(sdf_trajs_dir)

            # 保存缓存训练文件path
            with open(self.samples_file, "wb") as f:
                pickle.dump(self.samples, f)
            with open(self.vocab_file, "w") as f:
                json.dump(self.frag_vocab, f, indent=2)
            print(f"[INFO] 缓存完成, 样本数={len(self.samples)}, fragment类别数={len(self.frag_vocab)}")

    def _extract_new_fragment_by_coords(self, core, frag, tol=0.01):
        """基于坐标差分提取新 fragment"""
        core_conf, frag_conf = core.GetConformer(), frag.GetConformer()

        # === 找到 core 里的 R 原子索引 ===
        R_idx = None
        for atom in core.GetAtoms():
            if atom.GetSymbol() == "R":
                R_idx = atom.GetIdx()
                break
        if R_idx is None:
            raise ValueError("Core 中未找到 R 占位原子，请确认输入分子。")

        # 1. core 中需要删除的原子 (除 R 和 R 的邻居)
        R_neighbors = [a.GetIdx() for a in core.GetAtomWithIdx(R_idx).GetNeighbors()]
        core_keep = set([R_idx] + R_neighbors)
        core_remove = [i for i in range(core.GetNumAtoms()) if i not in core_keep]

        # 2. 在 frag 中找对应原子
        frag_remove = []
        frag_keep = []
        for f_idx in range(frag.GetNumAtoms()):
            f_pos = np.array(frag_conf.GetAtomPosition(f_idx))
            matched = False
            for c_idx in core_remove:
                c_pos = np.array(core_conf.GetAtomPosition(c_idx))
                if np.linalg.norm(f_pos - c_pos) < tol:
                    frag_remove.append(f_idx)
                    matched = True
                    break
            if not matched:
                frag_keep.append(f_idx)

        # 3. 构建新 fragment
        emol = Chem.RWMol(frag)
        for idx in sorted(frag_remove, reverse=True):
            emol.RemoveAtom(idx)

        # 4. 修改 R 邻居对应的原子 → Ru
        frag_conf_new = emol.GetConformer()
        for nb in R_neighbors:
            c_pos = np.array(core_conf.GetAtomPosition(nb))
            for f_idx in range(emol.GetNumAtoms()):
                f_pos = np.array(frag_conf_new.GetAtomPosition(f_idx))
                if np.linalg.norm(f_pos - c_pos) < tol:
                    atom = emol.GetAtomWithIdx(f_idx)
                    atom.SetAtomicNum(44)   # Ru
                    # atom.SetAtomicNum(0)            # dummy atom
                    # atom.SetProp("atomLabel", "X")  # 设置显示标签 X
                    break

        submol = emol.GetMol()
        submol_new, _ = remove_marker_atoms_with_map(submol)
        smiles = Chem.MolToSmiles(submol_new, canonical=True)
        smiles = smiles.replace("ru", "Ru")
        smiles = smiles.replace("-", "")        
        
        return smiles

    def _build_vocab_and_samples(self, sdf_trajs_dir):
        """两阶段：先扫描收集 fragment + 频数 → 构建 vocab → 再生成训练样本path"""
        frag_counter = Counter()
        task_list = []

        sdf_files = [os.path.join(sdf_trajs_dir, f) for f in os.listdir(sdf_trajs_dir) if f.endswith(".sdf")]
        # === 第一次扫描：收集 fragment ===
        for sdf_file in sdf_files:
            sdf_name = os.path.splitext(os.path.basename(sdf_file))[0]
            suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
            traj_mols = [m for m in suppl if m is not None]

            for i in range(len(traj_mols) - 1):
                core, frag = traj_mols[i], traj_mols[i+1]
                try:
                    frag_smiles = self._extract_new_fragment_by_coords(core, frag)
                    if frag_smiles is None:
                        continue
                    frag_counter[frag_smiles] += 1
                    task_list.append((sdf_file, sdf_name, i, frag_smiles))
                except Exception as e:
                    print(f"[WARN] 样本跳过: {e} (file={sdf_file}, step={i})")
                    continue

        # === 构建 vocab（按频数降序排序） ===
        frag_vocab = {}
        for idx, (smiles, freq) in enumerate(frag_counter.most_common()):  # 按频数降序
            frag_vocab[smiles] = {"id": idx, "freq": freq}

        samples = []

        # === 第二次扫描：生成样本 ===
        for sdf_file, sdf_name, i, frag_smiles in task_list:
            core_path = os.path.join(self.core_dir, f"{sdf_name}_{i}_core.pt")            

            # 重读分子
            suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
            traj_mols = [m for m in suppl if m is not None]
            core = traj_mols[i]

            # 生成图并存储
            if not os.path.exists(core_path):
                core_graph = Mol2Graph_for_frag_pred(core, self.ED_center)._create_graph()
                y = frag_vocab[frag_smiles]["id"]
                # 只选择高频率片段:
                if y <= 98:
                    core_graph.y = th.tensor(frag_vocab[frag_smiles]["id"], dtype=th.long)
                else:
                    core_graph.y = th.tensor(99, dtype=th.long)
                th.save(core_graph, core_path)
            
            ED_file = '../cache_ed/' + sdf_file.split('/')[-1] + '_' + sdf_file.split('_')[-1].split('.')[0] + '_0_rho.npy'
            if not os.path.exists(ED_file):
                continue            

            samples.append((core_path, ED_file))

        return frag_vocab, samples
    
    def get_class_weights(self, max_classes=100, smoothing="inverse"):
        """
        根据 frag_vocab 频数计算损失权重
        Args:
            max_classes (int): 限制类别数(比如只取前 99 类,后面当作 "other")
            smoothing (str): 权重计算方法:
                - "inverse": 1/freq
                - "sqrt": 1/sqrt(freq)
                - "balanced": N / (C * freq)，其中 N=总样本数, C=类别数
        Returns:
            torch.FloatTensor: [num_classes] 权重向量
        """        

        freqs = np.zeros(max_classes, dtype=np.float32)
        sum_freq_ = 0
        for smi, meta in self.frag_vocab.items():
            idx, freq = meta["id"], meta["freq"]
            if idx < max_classes-1:
                freqs[idx] = freq
            else:
                sum_freq_ += freq            
        freqs[max_classes-1] = sum_freq_

        # 避免0频率
        freqs[freqs == 0] = 1.0

        if smoothing == "inverse":
            weights = 1.0 / freqs
        elif smoothing == "sqrt":
            weights = 1.0 / np.sqrt(freqs)
        
        else:
            raise ValueError(f"未知 smoothing 方法: {smoothing}")

        # 归一化到平均值为1，避免 loss 太大或太小
        weights = weights / weights.mean()

        return th.tensor(weights, dtype=th.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        core_path, ED_path = self.samples[idx]        
        try:
                core_graph = th.load(core_path)                
                ED = th.tensor(np.load(ED_path), dtype=th.float32).reshape(-1, 48, 48, 48)
                return core_graph, ED
        except Exception as e:
            print(f"[WARN] 加载失败 idx={idx}: {e}")
        return None




class ResidualBlock(nn.Module):
    """
    3D 卷积版本的残差模块 (Residual Block)，和 ResNet 的基本残差单元类似，
    只不过是用在 三维数据（体数据，比如医学图像、分子密度图、点云体素化结果）
    模块的作用: 
        保证 信息可以绕过卷积层直接传递（残差思想）
        减少深层网络的梯度消失问题
        能够训练更深的 3D 卷积网络
    """

    def __init__(self, channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.padding = padding   # 用于提取 3D 特征。卷积核大小是 3×3×3
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)  # 批归一化，加快训练并稳定收敛
        self.relu = nn.ReLU(inplace=True)
        # conv1 和 conv2 的 stride 都是相同的，意味着残差分支（identity）和主分支的空间分辨率必须保持一致，否则会维度不匹配
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity      # 残差连接
        out = self.relu(out)
        return out



# 等变图神经网络 (Equivariant Graph Neural Network, EGNN)
class EGNNlayer(MessagePassing):
    """
    基于 PyTorch Geometric 的 MessagePassing。它能同时更新 节点特征 (x) 和 节点坐标 (coords),
    保证网络对 平移、旋转等操作保持等变性
    in_size: 输入节点特征的维度
    hid_size: 隐藏层的神经元数量
    out_size: 输出节点特征的维度
    edge_feat_size: 边特征的维度, 默认为 0 (即如果没有边特征)
    """

    def __init__(self, in_size, hid_size, out_size, edge_feat_size=0):
        super(EGNNlayer, self).__init__()
        #        边
        self.phi_e = nn.Sequential(nn.Linear(2 * in_size + edge_feat_size + 1,  # x_i, x_j（两个节点特征拼接）+ edge_feat + dist（节点间距离）
                                              hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, out_size),
                                  nn.ReLU())
        #        坐标
        self.phi_x = nn.Sequential(nn.Linear(out_size, hid_size),   # 输入来自 phi_e 的输出，表示边对节点坐标更新的影响
                                  nn.ReLU(),
                                  nn.Linear(hid_size, 1),  # 输出：一个标量 w（沿着边方向更新坐标的强度）
                                  nn.ReLU())
        #      节点特征更新网络
        self.phi_h = nn.Sequential(nn.Linear(in_size + out_size,  # 输入：原始节点特征 x + 来自 phi_e 的输出
                                             hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, out_size),  # 输出：更新后的节点特征 x_out
                                  nn.ReLU())
    
    def forward(self, edge_index, x, coords, edge_feat):

        rela_diff = coords[edge_index[0]] - coords[edge_index[1]]  # 两个节点的相对坐标差
        dist = th.norm(coords[edge_index[0]] - coords[edge_index[1]], dim = 1, keepdim = True)
        
        edge_feat = th.cat([edge_feat, dist], 1)   # 把距离拼到边特征里
        
        x_out, coords_out = self.propagate(edge_index, x = x, coords = coords,    # 然后调用 propagate（PyG 的消息传递机制）
                                           edge_feat = edge_feat, dist = dist, rela_diff = rela_diff)                
        return x_out, coords_out
        
    def propagate(self, edge_index, size = None, **kwargs):
        """
        在图结构数据上进行消息传递和节点更新
        edge_index: 表示图的边索引, 指示图中节点之间的连接关系
        size: 可选参数, 通常表示图的大小 (节点数)
        **kwargs: 允许传递额外的关键字参数
        self._check_input、self._collect、self.inspector.distribute、self.aggregate 
        都是 PyG 的 MessagePassing 基类里提供的封装工具
        """        
        size = self._check_input(edge_index, size)  # 验证 edge_index 和 size 的有效性，如果 size=None，PyG 会自动推断
        # 收集消息传递所需的特征:
        coll_dict = self._collect(self._user_args, edge_index, size,   # 把 全局特征拆分成消息传递需要的 x_i, x_j, edge-level 特征
                                         kwargs)
        
        # 分发消息参数
        msg_kwargs = self.inspector.distribute('message', coll_dict)  # self.inspector 会分析 message 和 update 函数的参数列表，然后自动把 coll_dict 里对应的张量分发给它们
        # 聚合函数的参数
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        msg = self.message(**msg_kwargs)   # 调用 message 方法！！！ 计算消息
        
        w = self.phi_x(msg)   # 先用 phi_x(msg) 得到权重 w
        rela_diff = kwargs["rela_diff"]
        aggr_w = self.aggregate(w * rela_diff, **aggr_kwargs)  # 沿着边方向更新坐标
        coords_out = kwargs["coords"] + aggr_w    # 聚合邻居贡献
        
        msg = self.aggregate( msg , **aggr_kwargs)
        
        x = kwargs["x"]
        x_out = self.update(x, msg)   # 调用 update 方法
        return x_out, coords_out
        
    def message(self, x_i, x_j, edge_feat):
        edge_feat = edge_feat.float()
        message = self.phi_e(th.cat([x_i, x_j, edge_feat], 1))
        return message

    def update(self, x, message):
        x_out = self.phi_h(th.cat([x, message], 1))
        return x_out


class Frag_Pred_Model_0(nn.Module):
    """
    多模态融合模型：同时利用    
    核心骨架 (core graph)
    三维网格 (grid, 体素化密度或电子密度？)
    来预测  分子的下一生长片断
    """

    def __init__(self, core_dim_in=35,  # 核心图的节点特征维度
                 egnn_dim_tmp=1024, egnn_dim_out=256, egnn_num_layers=7, dim_edge_feat=7,  # EGNN 参数
                 cnn_dim_in=1, cnn_dim_tmp=512, stride=2, padding=3,  # 3D CNN 参数, CNN 通道逐步扩展到 512
                 dim_tmp=128, dim_out=100, num_layers=6):   # MLP 参数  把片断离散成 100 个 类
        super(Frag_Pred_Model_0, self).__init__()        
        self.core_dim_in = core_dim_in
        self.egnn_dim_tmp = egnn_dim_tmp
        self.egnn_dim_out = egnn_dim_out
        self.egnn_num_layers = egnn_num_layers
        self.dim_edge_feat = dim_edge_feat
        self.cnn_dim_in = cnn_dim_in
        self.cnn_dim_tmp = cnn_dim_tmp
        self.stride = stride
        self.padding = padding
        self.dim_tmp = dim_tmp
        self.dim_out = dim_out
        self.num_layers = num_layers
        # 处理核心骨架
        self.layer_core = EGNNlayer(self.core_dim_in, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 将中间输出结果堆叠 多层 EGNN
        self.layer1 = EGNNlayer(self.egnn_dim_out, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 融合 核心 graph + 3D CNN 特征后，逐层映射到 100 类
        #                           256                 512*2             128*2
        self.fc = nn.Linear(self.egnn_dim_out + self.cnn_dim_tmp * 2, self.dim_tmp * 2)
        self.fc2 = nn.Linear(self.dim_tmp * 2, self.dim_tmp * 2)
        self.fc3 = nn.Linear(self.dim_tmp * 2, self.dim_tmp)
        self.fc1 = nn.Linear(self.dim_tmp, self.dim_out)
        self.bn1 = nn.BatchNorm1d(self.dim_tmp)

        # 3D CNN 部分，连续 4 层 3D 卷积 + 3D 残差模块：
        self.bn3d = nn.BatchNorm3d(self.cnn_dim_in)
        self.conv3D1 = nn.Conv3d(self.cnn_dim_in, self.cnn_dim_tmp // 8, kernel_size=7, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D2 = nn.Conv3d(self.cnn_dim_tmp // 8, self.cnn_dim_tmp // 4, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D3 = nn.Conv3d(self.cnn_dim_tmp // 4, self.cnn_dim_tmp // 2, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D4 = nn.Conv3d(self.cnn_dim_tmp // 2, self.cnn_dim_tmp, kernel_size=3, stride=self.stride, padding=self.padding, bias=False) 
        self.layer3D1 = self._make_layer(self.cnn_dim_tmp // 8)
        self.layer3D2 = self._make_layer(self.cnn_dim_tmp // 4)
        self.layer3D3 = self._make_layer(self.cnn_dim_tmp // 2)
        self.layer3D4 = self._make_layer(self.cnn_dim_tmp)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 最后 AdaptiveAvgPool3d 压缩成全局向量

    def _make_layer(self, channels, num_blocks=2):   # 堆叠多个 3D 残差块
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self,
                edge_index1, x1, coords1, edge_feat1, batch1,   # 核心图
                grid):                                          # 三维网格        

        # 核心 graph 特征提取：
        x_out1, coords_out1 = self.layer_core(edge_index1, x1, coords1, edge_feat1)
        x_out1 = x_out1.relu()
        for i in range(self.egnn_num_layers):
                                   # layer1 被复用 在 frag/core graph 上，这样参数是共享的
            x_out1, coords_out1 = self.layer1(edge_index1, x_out1, coords_out1, edge_feat1)  # EGNN 提取空间不变特征
            x_out1 = x_out1.relu()
        readout1 = global_sort_pool(x_out1, batch1, k=3) 

        # 3D CNN 特征提取，逐层卷积 → 残差块 → 下采样 → 全局池化，输出展平成一个向量：
        grid = self.bn3d(grid)
        grid = self.conv3D1(grid)
        grid = self.layer3D1(grid)  # 每次 conv3D 之后立刻跟一个 ResidualBlock，目的是让网络在提取特征的同时还能保持残差信息传递
        grid = self.conv3D2(grid)
        grid = self.layer3D2(grid)
        grid = self.conv3D3(grid)
        grid = self.layer3D3(grid)   
        grid = self.conv3D4(grid)
        grid = self.layer3D4(grid)
        grid = self.avgpool(grid)
        grid = th.flatten(grid, 1)

        readout = th.cat((readout1, grid), dim=1)  # 多模态融合，把 片段图、核心图、3D CNN 特征拼接
        readout = self.fc(readout)
        for i in range(self.num_layers):
            readout = self.fc2(readout).relu() + readout   # 残差式 MLP
        readout = self.fc3(readout).relu()
        readout = self.bn1(readout)
        readout = self.fc1(readout)
        readout = F.softmax(readout, dim=1)   # 输出 36 维向量 → softmax 得到角度类别概率分布
        return readout
    
class Frag_Pred_Model(nn.Module):
    """
    多模态融合模型：同时利用    
    核心骨架 (core graph)
    三维网格 (grid, 体素化密度或电子密度？)
    来预测  分子的下一生长片断
    """

    def __init__(self, core_dim_in=35,  # 核心图的节点特征维度
                 egnn_dim_tmp=512, egnn_dim_out=256, egnn_num_layers=5, dim_edge_feat=7,  # EGNN 参数
                 cnn_dim_in=1, cnn_dim_tmp=512, stride=2, padding=3,  # 3D CNN 参数, CNN 通道逐步扩展到 512
                 dim_tmp=128, dim_out=100, num_layers=4):   # MLP 参数  把片断离散成 100 个 类
        super(Frag_Pred_Model, self).__init__()        
        self.core_dim_in = core_dim_in
        self.egnn_dim_tmp = egnn_dim_tmp
        self.egnn_dim_out = egnn_dim_out
        self.egnn_num_layers = egnn_num_layers
        self.dim_edge_feat = dim_edge_feat
        self.cnn_dim_in = cnn_dim_in
        self.cnn_dim_tmp = cnn_dim_tmp
        self.stride = stride
        self.padding = padding
        self.dim_tmp = dim_tmp
        self.dim_out = dim_out
        self.num_layers = num_layers
        # 处理核心骨架
        self.layer_core = EGNNlayer(self.core_dim_in, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 将中间输出结果堆叠 多层 EGNN
        self.layer1 = EGNNlayer(self.egnn_dim_out, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 融合 核心 graph + 3D CNN 特征后，逐层映射到 100 类
        #                           256                 512*2             128*4
        self.fc = nn.Linear(self.egnn_dim_out + self.cnn_dim_tmp * 2, self.dim_tmp * 4)
        self.fc2 = nn.Linear(self.dim_tmp * 4, self.dim_tmp * 4)
        self.fc3 = nn.Linear(self.dim_tmp * 4, self.dim_tmp * 2)
        self.fc1 = nn.Linear(self.dim_tmp * 2, self.dim_out)
        self.bn1 = nn.BatchNorm1d(self.dim_tmp * 2)

        # 3D CNN 部分，连续 4 层 3D 卷积 + 3D 残差模块：
        self.bn3d = nn.BatchNorm3d(self.cnn_dim_in)
        self.conv3D1 = nn.Conv3d(self.cnn_dim_in, self.cnn_dim_tmp // 8, kernel_size=7, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D2 = nn.Conv3d(self.cnn_dim_tmp // 8, self.cnn_dim_tmp // 4, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D3 = nn.Conv3d(self.cnn_dim_tmp // 4, self.cnn_dim_tmp // 2, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D4 = nn.Conv3d(self.cnn_dim_tmp // 2, self.cnn_dim_tmp, kernel_size=3, stride=self.stride, padding=self.padding, bias=False) 
        self.layer3D1 = self._make_layer(self.cnn_dim_tmp // 8)
        self.layer3D2 = self._make_layer(self.cnn_dim_tmp // 4)
        self.layer3D3 = self._make_layer(self.cnn_dim_tmp // 2)
        self.layer3D4 = self._make_layer(self.cnn_dim_tmp)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 最后 AdaptiveAvgPool3d 压缩成全局向量

    def _make_layer(self, channels, num_blocks=2):   # 堆叠多个 3D 残差块
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self,
                edge_index1, x1, coords1, edge_feat1, batch1,   # 核心图
                grid):                                          # 三维网格        

        # 核心 graph 特征提取：
        x_out1, coords_out1 = self.layer_core(edge_index1, x1, coords1, edge_feat1)
        x_out1 = x_out1.relu()
        for i in range(self.egnn_num_layers):
                                   # layer1 被复用 在 frag/core graph 上，这样参数是共享的
            x_out1, coords_out1 = self.layer1(edge_index1, x_out1, coords_out1, edge_feat1)  # EGNN 提取空间不变特征
            x_out1 = x_out1.relu()
        readout1 = global_sort_pool(x_out1, batch1, k=3) 

        # 3D CNN 特征提取，逐层卷积 → 残差块 → 下采样 → 全局池化，输出展平成一个向量：
        grid = self.bn3d(grid)
        grid = self.conv3D1(grid)
        grid = self.layer3D1(grid)  # 每次 conv3D 之后立刻跟一个 ResidualBlock，目的是让网络在提取特征的同时还能保持残差信息传递
        grid = self.conv3D2(grid)
        grid = self.layer3D2(grid)
        grid = self.conv3D3(grid)
        grid = self.layer3D3(grid)   
        grid = self.conv3D4(grid)
        grid = self.layer3D4(grid)
        grid = self.avgpool(grid)
        grid = th.flatten(grid, 1)

        readout = th.cat((readout1, grid), dim=1)  # 多模态融合，把 片段图、核心图、3D CNN 特征拼接
        readout = self.fc(readout)
        for i in range(self.num_layers):
            readout = self.fc2(readout).relu() + readout   # 残差式 MLP
        readout = self.fc3(readout).relu()
        readout = self.bn1(readout)
        readout = self.fc1(readout)
        readout = F.softmax(readout, dim=1)   # 输出 36 维向量 → softmax 得到角度类别概率分布
        return readout


from utils.Utils import SetSeed, GetConfigs
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def collate_fn(batch):
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    core_graphs, EDs = zip(*batch)
    return core_graphs, EDs

def statistic_of_model_params_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name}: {p.numel():,}")

def main():
    device = th.device("cuda")
    batch_size = 64
    lr = 1e-5
    epochs = 2500
    output_dir = '../GFPM_zzx_train'
    os.makedirs(output_dir, exist_ok=True)
    model_save = os.path.join(output_dir, "gfpm.pt")
    log_file = os.path.join(output_dir, "GFPM_log.txt")
    SetSeed(42)

    # 数据集和 DataLoader
    sdf_trajs_dir = '../fragment_extension_dataset/translate_and_splited_sdf_subset1'
    full_dataset = Dataset_Grow_Frag_Labels(sdf_trajs_dir)
    weights = full_dataset.get_class_weights(max_classes=100, smoothing="inverse")
    weights = weights.to(device)

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
    model = Frag_Pred_Model().to(device)
    # statistic_of_model_params_num(model)
    # model = th.compile(model, backend="aot_eager")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6, verbose=True
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
    # 文件日志
    with open(log_file, "w") as f:
        f.write("Epoch\tLoss\tAcc@1\tAcc@3\tAcc@5\tLR\n")

    # AMP GradScaler
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc, total_acc_3, total_acc_5 = 0, 0, 0
        total_val_loss = 0
        total_val_acc, total_val_acc_3, total_val_acc_5 = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            if batch is None or (isinstance(batch, (list, tuple)) and any(b is None for b in batch)):
                continue

            core_graphs, EDs = batch
            core_batch = core_graphs.to(device)            
            ED_batch = EDs.to(device)

            optimizer.zero_grad()
            
            # === 自动混合精度前向计算 ===
            with autocast():
                out = model(core_batch.edge_index, core_batch.x, core_batch.pos, core_batch.edge_attr, core_batch.batch,
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
            topk_vals, topk_idx = th.topk(out, k=5, dim=1)
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

        print(f"Epoch {epoch+1}: train Loss={epoch_loss:.4f}, train Acc@1={epoch_acc*100:.2f}%, train Acc@3={epoch_acc_3*100:.2f}%, train Acc@5={epoch_acc_5*100:.2f}%, lr={lr_now:.6f}\n")

        # TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/Top1", epoch_acc, epoch)
        writer.add_scalar("Accuracy/Top3", epoch_acc_3, epoch)
        writer.add_scalar("Accuracy/Top5", epoch_acc_5, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        # 文件日志
        with open(log_file, "a") as f:
            f.write(f"train: {epoch+1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_acc_3:.4f}\t{epoch_acc_5:.4f}\t{lr_now:.6f}\n")

        scheduler.step(epoch_loss)

        model.eval()
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            if batch is None or (isinstance(batch, (list, tuple)) and any(b is None for b in batch)):
                continue

            core_graphs, EDs = batch
            core_batch = core_graphs.to(device)            
            ED_batch = EDs.to(device)            
            
            # === 自动混合精度前向计算 ===
            with autocast():
                out = model(core_batch.edge_index, core_batch.x, core_batch.pos, core_batch.edge_attr, core_batch.batch,
                            ED_batch.float())
                loss = criterion(out, core_batch.y)

            total_val_loss += loss.item()

            # Top-1
            pred = out.argmax(dim=1)
            correct = (pred == core_batch.y).sum().item()
            acc = correct / core_batch.y.size(0)

            # Top-3 / Top-5
            topk_vals, topk_idx = th.topk(out, k=5, dim=1)
            correct_top3 = (topk_idx[:, :3] == core_batch.y.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 = (topk_idx[:, :5] == core_batch.y.unsqueeze(1)).any(dim=1).sum().item()
            acc_top3 = correct_top3 / out.size(0)
            acc_top5 = correct_top5 / out.size(0)

            total_val_acc += acc
            total_val_acc_3 += acc_top3
            total_val_acc_5 += acc_top5

            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Acc@1": f"{acc*100:.2f}%",
                "Acc@3": f"{acc_top3*100:.2f}%",
                "Acc@5": f"{acc_top5*100:.2f}%"
            })

        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_acc = total_val_acc / len(val_loader)
        epoch_val_acc_3 = total_val_acc_3 / len(val_loader)
        epoch_val_acc_5 = total_val_acc_5 / len(val_loader)        

        print(f"Epoch {epoch+1}: val Loss={epoch_val_loss:.4f}, val Acc@1={epoch_val_acc*100:.2f}%, val Acc@3={epoch_val_acc_3*100:.2f}%, val Acc@5={epoch_val_acc_5*100:.2f}%\n")

        # TensorBoard
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Top1_val", epoch_val_acc, epoch)
        writer.add_scalar("Accuracy/Top3_val", epoch_val_acc_3, epoch)
        writer.add_scalar("Accuracy/Top5_val", epoch_val_acc_5, epoch)        

        # 文件日志
        with open(log_file, "a") as f:
            f.write(f"val: {epoch+1}\t{epoch_val_loss:.4f}\t{epoch_val_acc:.4f}\t{epoch_val_acc_3:.4f}\t{epoch_val_acc_5:.4f}\n")
        
        if epoch % 30 ==0:
            th.save(model.state_dict(), f'{output_dir}/gfpm_epoch{epoch}.pt')
            print(f"✅ Growth_Frag_pred 模型已保存到 {output_dir}/gfpm_epoch{epoch}.pt")
            print(f"✅ 训练日志已保存到 {log_file}")

    writer.close()
    


if __name__ == "__main__":
    main()
    