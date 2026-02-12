import copy
import warnings
warnings.filterwarnings("ignore")
import torch as th
import numpy as np
from math import pi
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from torch_geometric import loader
from torch_geometric.data import Data, Dataset, DataLoader
from utils.Utils import *


class Mol2GraphGrowthPoint:
    """
    把一个分子 (mol) 转换成图 (Graph)，其中节点是原子，边是化学键，并且额外加入了基于电子密度 (ED, electron density)
      的环境特征。最后会生成 PyTorch Geometric 里的 Data 图 (Graph) 对象
    """
    def __init__(self, mol, edval, tree):
        """
        mol: 一个 RDKit 分子对象
        edval: 3D 网格上的电子密度值
        tree: KDTree ( 用来快速查询某点附近哪些网格点在半径 r 内 )
        self.env: 每个原子的 电子密度环境向量
        """
        self.mol = mol
        self.env = self.calc_ed_env(mol, tree, edval)

    def _onehotencoding(self, lst, i):
        """
        把数值 i 编码成 one-hot 向量，例如：
            _onehotencoding([1.0, 2.0, 3.0], 2.0) → [0, 1, 0]
        """
        return list(map(lambda x: 1 if x == i else 0, lst))

    def _atom_featurization(self):
        """
        原子特征化
        positions: 原子坐标 (pos)
        atom_vector: 原子特征 (来源于 电子密度环境 self.env[idx]，不是传统的元素 one-hot)

        返回：
        atom_coords: 原子坐标张量
        atom_vector: 原子特征张量
        """
        atomic_num, atom_vector = [], []
        conformer = self.mol.GetConformer()
        positions = conformer.GetPositions()
        for idx, a in enumerate(self.mol.GetAtoms()):
            atomtype_encoding = self.env[idx]    # 从 电子密度环境 拿到向量
            atom_vector.append(atomtype_encoding)
        atom_coords = th.from_numpy(positions)
        atom_vector = th.tensor(atom_vector, dtype=th.float)
        return atom_coords, atom_vector   

    def _bond_featurization(self):
        """
        键特征化
        对每条键生成 双向边 (既有 A→B 也有 B→A)
        特征包含：
        bond 类型 one-hot (1,2,3,1.5,0)
        是否在环中 (IsInRing)
        是否共轭 (IsConjugated)
        """
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

    def _atom_label(self):
        """
        原子标签
        这是 监督学习的标签 y:
            1 → 连接点 (R 基/占位符 *)
            0 → 氢原子
            2 → 其他原子
        """
        label_vector = []
        for idx, a in enumerate(self.mol.GetAtoms()):
            if a.GetSymbol() in ["R", "*"]:   # attachment point
                label_vector.append([1])
            elif a.GetSymbol() in ["H"]:      # 氢原子
                label_vector.append([0])
            else:                             # 普通原子
                label_vector.append([2])
        return th.tensor(label_vector, dtype = th.long)

    def _create_graph(self):
        """
        创建图
        返回的是 PyTorch Geometric 的 Data 对象：
        x: 节点特征 (电子密度环境向量)
        edge_index: 边的连接关系
        edge_attr: 边特征
        pos: 节点坐标
        y: 标签 (原子类型分类标签)
        """
        bond_vector = self._bond_featurization()   # 调用 键特征化 方法
        atom_coords, atom_vector = self._atom_featurization()  # 调用 原子特征化 方法
        edge_index, edge_attr = bond_vector[:, :2], bond_vector[:, 2:]
        edge_index = edge_index.permute(1,0)  # 形状调整成 [2, num_edges]
        atom_vector = atom_vector.float()
        edge_attr = edge_attr.float()
        atom_label = self._atom_label()   # 调用 原子标签 方法
        graph = Data(x = atom_vector, edge_index = edge_index, edge_attr = edge_attr,  pos = atom_coords, y = atom_label)
        return graph

    def calc_ed_env(self, mol, tree, edval):
        """
        电子密度环境计算
        Input:
        mol: Mol Object

        tree: KDtree from 3D grid coordinates
        tree 是由 3D 网格点坐标 建出来的 KDTree。
        它的功能就是： 给定一个点坐标和半径 r, 快速找到 所有在 r 范围内的网格点索引。
        (KDTree 是一种空间划分的数据结构，用于加速最近邻/范围搜索。)
        如果直接暴力搜索 “某个原子周围 r Å 内有哪些网格点”，复杂度是 O(N) (N = 网格点数，通常很大)。
        KDTree 把复杂度降低到 O(log N)，所以效率大幅提升。

        edval: ED value of grid points.

        Output:
        10 dimension vector of ED feature, 每个原子的电子密度环境向量 (10 维度 = 半径 1, 1.5, 2, … 3.0, 每个半径两个指标）
        指标：
            ex_ratio: 该半径内电子密度网格中, 多少比例同时属于 “重原子周围的网格点”   (原子周围电子密度的 分布模式，既考虑自己周围的密度，也对比了邻居的密度)
            acc_ratio: 该半径内电子密度网格占比, 减去 ex_ratio
        """
        ha_coors = []; at_coors = {}
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = mol.GetConformer().GetAtomPosition(i)
            if atom.GetSymbol() not in ["R", "*", "H"]:
                ha_coors.append([pos.x, pos.y, pos.z])    # 只取 heavy atom
                at_coors[i] = [pos.x, pos.y, pos.z]
            else:
                at_coors[i] = [pos.x, pos.y, pos.z]
        
        if at_coors:
            ind1 = tree.query_ball_point(ha_coors, 1.5)   # 输入3D坐标，找到所有 heavy atom 周围 1.5Å 的 grid 点，ind1是点坐标ind的list
            atomexgrid = []
            for i in ind1:
                atomexgrid.extend(i)
            atomexgrid = np.array(list(set(atomexgrid)))
            atomexgrid = atomexgrid[edval[atomexgrid] > 0]   # 只保留电子密度 >0 的点
            # 把所有 heavy atom 周围 1.5 Å 内的网格点索引合并成一个集合，留电子密度实际存在的网格点。这就得到了全分子范围内，所有重原子周围的电子密度网格点
            # atomexgrid: 所有 heavy atom (非 H、R、*) 周围 1.5 Å 内的电子密度点（且 edval > 0）。可以理解为 分子中“重原子外层”电子密度点的全集

            
            
            env = defaultdict(list)
            for idx in at_coors:
                for r in np.arange(1, 3.5, 0.5):   # 半径逐步扩张
                    try:
                        ind = tree.query_ball_point(at_coors[idx], r)  # 半径 r 内的网格点，不考虑电子密度
                        ind = np.array(ind)
                        ingrid = ind[edval[ind] > 0]      # 筛出 电子密度 > 0 的点
                        ex_ratio = np.intersect1d(atomexgrid, ingrid).size / ind.size   # “这个原子 r 区域的网格点” 和 “全局重原子周围的网格点” 的交集
                        in_ratio = ingrid.size / ind.size
                        acc_ratio = in_ratio - ex_ratio   # 自己主导的电子密度比例
                        env[idx].extend([ex_ratio, acc_ratio])
                        """
                        每个原子最终得到的向量是：
                            [ex_ratio(r=1.0), acc_ratio(r=1.0),
                            ex_ratio(r=1.5), acc_ratio(r=1.5),
                            ...
                            ex_ratio(r=3.0), acc_ratio(r=3.0)]

                            共 10 维。
                            ex_ratio: 描述原子周围电子密度 和邻近原子区域的重叠程度
                            acc_ratio: 描述原子周围电子密度 主要属于自己贡献的程度
                        """
                    except:
                        env[idx].extend([0, 0])
        else:
            raise AssertionError("No attachment point")
        return env


class DatasetGrowthPoint(Dataset):
    def __init__(self, data_list,edval,tree):
        super(DatasetGrowthPoint, self).__init__()
        self.data_list = data_list
        self.edval = edval
        self.tree = tree

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        mol = Chem.AddHs(self.data_list[idx],addCoords=True)
        core2graph = Mol2GraphGrowthPoint(mol, self.edval, self.tree)
        return core2graph._create_graph()    # 调用 _create_graph()方法，返回分子图

    def get(self, idx):
        pass
    
    def len(self, idx):
        pass


def GrowthPointFilter(configs, cores, model, cores_dataset, step=0, growth_point_pos=None):
    """
    Filter the cores based on the growth points.
    在候选“核心片段 (cores)”中，筛选出可以作为生长点 (growth point) 的分子。
    分子里可能有 R、*、H 这样的“虚原子”或“挂点”。
    函数要么用一个给定的坐标 (growth_point_pos) 来确定生长点；
    要么用 模型预测结果 来判断哪些位置可以生长。
    最终分成两类：
        pass_cores_filter: 通过筛选的分子
        unpass_cores_filter: 未通过筛选的分子
    """
    batch_size = int(configs.get('sample', 'batch_size'))
    device_type = configs.get('sample', 'device_type')
    trainloader = loader.DataLoader(cores_dataset, batch_size=batch_size, shuffle = False, num_workers = 1,pin_memory=False, prefetch_factor = 8)
    device = th.device(device_type)
    model = model.to(device)
    model.eval()              # 模型切换到 eval() 推理模式
    pass_cores_filter = []
    unpass_cores_filter = []

    if (not step) and (growth_point_pos is not None):
        # Lead optimization only contain one ligand core，先导化合物优化，只有一个核心片段 (cores[0])。给定了一个三维坐标，表示目标的生长点
        mol = copy.deepcopy(cores[0])
        mol = Chem.AddHs(mol,addCoords=True)
        growth_point_index = 0
        min_dist = 999
        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() in ["R", "*", "H"]:
                atom_pos = mol.GetConformer().GetAtomPosition(idx)
                dist = np.linalg.norm(np.array(atom_pos) - np.array(growth_point_pos))
                if dist < min_dist:  # 取到生长点坐标 growth_point_pos 距离最小的原子作为候选生长点
                    min_dist = dist
                    growth_point_index = idx
        
        print(f"the min distance between the growth point and the atom is {min_dist}")
        try:
            mol.GetAtoms()[growth_point_index].SetAtomicNum(0)  # RDKit 里原子号=0 是“dummy atom”，表示挂点
            pass_cores_filter.append(mol)
        except:
            unpass_cores_filter.append(mol)
            print("Error: growth point is not in the molecule")

    # 没有给坐标，走模型预测:
    else:
        with th.no_grad():
            for batch, mols in zip(trainloader, [cores[i:i+batch_size] for i in range(0, len(cores), batch_size)]):
                batch = batch.to(device)
                outputs = model(batch.edge_index, batch.x, batch.pos, batch.edge_attr) # 输出每个原子的预测类别, 3类（比如：0=不是生长点，1=生长点）
                _, predicted_labels = outputs.cpu().max(dim=1)
                initial_index = 0
                for _mol in mols:
                    mol = copy.deepcopy(_mol)  # 复制一份，补氢
                    mol = Chem.AddHs(mol,addCoords=True)
                    atomic_probabilities = []  # 保存候选生长点和其预测概率

                    for idx, atom in enumerate(mol.GetAtoms()):
                        atom_idx = initial_index + idx
                        if atom.GetSymbol() in ["R", "*", "H"]:
                            # if predicted_labels[atom_idx].item() == 1 and atom.GetIsotope() != 2:
                            if predicted_labels[atom_idx].item() == 0 and atom.GetIsotope() != 2:  # 被预测为 生长点 (label=1)； 且同位素号不等于 2
                                atomic_probabilities.append([idx, _[atom_idx].item()])
                    atomic_probabilities = sorted(atomic_probabilities, key=lambda x: x[1], reverse=True)  # 选择最可能的生长点

                    try:
                        if atomic_probabilities[0][1]>0.5:  # 如果最高概率 > 0.5
                            growth_point_index = atomic_probabilities[0][0]
                            mol.GetAtoms()[growth_point_index].SetAtomicNum(0)  # 把对应原子改成 dummy atom
                            pass_cores_filter.append(mol)
                    except:
                        unpass_cores_filter.append(mol)

                    initial_index += (idx + 1)   # 把多个分子 cores_dataset 打包成一个 batch 输入，确定某个原子属于 batch 中的哪个分子
    
    return pass_cores_filter, unpass_cores_filter


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
        temp = [core_graph,frag_graph,self.ED]
        return temp

    def get(self, idx):
        pass

    def len(self, idx):
        pass


def get_geometric_center(mol, confId=0):
    """
    Calculate the geometric center (i.e. center of coordinates) of a molecule.
    
    Parameters:
      mol (rdkit.Chem.Mol): RDKit molecule with at least one conformer.
      confId (int): ID of the conformer to use (default is 0).
    
    Returns:
      center (tuple): (x, y, z) coordinates of the geometric center.
    """
    # Check if the molecule has any conformers
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule does not have any conformers.")

    conf = mol.GetConformer(confId)
    n_atoms = mol.GetNumAtoms()
    sum_x = sum_y = sum_z = 0.0

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        sum_x += pos.x
        sum_y += pos.y
        sum_z += pos.z

    center = (sum_x / n_atoms, sum_y / n_atoms, sum_z / n_atoms)
    return center


def Attach(model, frags, core, EDgrid, configs):    
    device_type = configs.get('sample', 'device_type')
    ED_center = EDgrid[:,:3].mean(0)
    ED_Tensor = th.tensor(EDgrid[:,-1].reshape(-1,48,48,48))
    # angles = TorsionAnglePred(model, frags, core, ED_center, ED_Tensor, device_type, int(configs.get('sample', 'batch_size')))
    angles = TorsionAnglePred_zzx(model, frags, core, ED_center, ED_Tensor, device_type, int(configs.get('sample', 'batch_size')))
    print('预测出的前top-k个角度值： \n', angles)
    outcores = []
    for i, frag in zip(angles, frags):
        for TosionAngle in i:
            # attachmol = RotateFragment(core, frag, TosionAngle, 1.5)
            attachmol = RotateFragment(core, frag, TosionAngle, 1.5)
            mol = Chem.AddHs(attachmol,addCoords=True)
            
            mol.SetProp("_Name", mol.GetProp("_Name")+"_"+frag.GetProp("_Name"))
            outcores.append(mol)
    
    opt = configs.getboolean('sample', 'opt') if configs.has_option('sample', 'opt') else False
    outcores = MolOpt(outcores, configs) if opt else outcores  # 使用 smina 软件，优化分子，考虑了受体口袋的 pdb
    
    cleancores = [core for core in outcores if CheckAtomCol(core)]   # 检查原子间碰撞
    return cleancores


def TorsionAnglePred(model, frags, core, center, grid, device_type, batch_size, k=3):
    # 指定设备类型
    device = th.device(device_type)

    # 创建数据集
    dataset = DatasetTorsionAngle(frags, core, center, grid)
     
    # 创建数据加载器
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        prefetch_factor=1
    )

    tensor_list = []
    with th.no_grad():
        # 遍历数据加载器中的每个批次
        for _, batch in enumerate(val_loader):
            # 将批次的各个部分移动到指定设备上
            edge_index = batch[1].edge_index.to(device)
            x = batch[1].x.to(device)
            pos = batch[1].pos.to(device)
            edge_attr = batch[1].edge_attr.to(device)
            batch_0 = batch[1].batch.to(device)

            edge_index1 = batch[0].edge_index.to(device)
            x1 = batch[0].x.to(device)
            pos1 = batch[0].pos.to(device)
            edge_attr_1 = batch[0].edge_attr.to(device)
            batch1 = batch[0].batch.to(device)

            grid2 = batch[2].float().to(device)

            # 使用模型进行预测
            val_pred = model(edge_index,   x,  pos,   edge_attr, batch_0, 
                             edge_index1, x1, pos1, edge_attr_1, batch1,  grid2)

            # 将预测结果添加到列表中
            tensor_list.append(val_pred.cpu())
            # 释放CUDA缓存
            th.cuda.empty_cache()

    # 将列表中的所有张量拼接在一起
    new_tensor = th.cat(tensor_list, dim=0)

    # 获取前 10 个最大值及其索引
    values, indices = th.topk(new_tensor, 10, dim=1)

    # 将索引转换为角度值
    topk_angle = (np.array(indices)*10+5)/360*2*pi-pi

    # 返回前 k 个角度值， k = 3
    return topk_angle[:, :k]

def TorsionAnglePred_zzx(model, frags, core, center, grid, device_type, batch_size, k=1):
    # 指定设备类型
    device = th.device(device_type)

    # 创建数据集
    dataset = DatasetTorsionAngle(frags, core, center, grid)     # [core_graph,frag_graph,self.ED]
     
    # 创建数据加载器
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4
    )

    tensor_list = []
    with th.no_grad():
        # 遍历数据加载器中的每个批次
        for _, batch in enumerate(val_loader):
            # 将批次的各个部分移动到指定设备上
            edge_index = batch[1].edge_index.to(device)    #  frag_graph
            x = batch[1].x.to(device)
            pos = batch[1].pos.to(device)
            edge_attr = batch[1].edge_attr.to(device)
            batch_0 = batch[1].batch.to(device)

            edge_index1 = batch[0].edge_index.to(device)   #  core_graph
            x1 = batch[0].x.to(device)
            pos1 = batch[0].pos.to(device)
            edge_attr_1 = batch[0].edge_attr.to(device)
            batch1 = batch[0].batch.to(device)

            grid2 = batch[2].float().to(device)

            # 使用模型进行预测
            # val_pred = model(edge_index,   x,  pos,   edge_attr, batch_0, 
            #                  edge_index1, x1, pos1, edge_attr_1, batch1,  grid2)
            
        # model(frag_batch.edge_index, frag_batch.x, frag_batch.pos, frag_batch.edge_attr, frag_batch.batch,
        #                     core_batch.edge_index, core_batch.x, core_batch.pos, core_batch.edge_attr, core_batch.batch,
        #                     ED_batch.float())
            val_pred = model(edge_index, x, pos, edge_attr, batch_0,
                            edge_index1, x1, pos1, edge_attr_1, batch1,
                            grid2)

            # 将预测结果添加到列表中
            tensor_list.append(val_pred.cpu())
            # 释放CUDA缓存
            th.cuda.empty_cache()

    # 将列表中的所有张量拼接在一起
    new_tensor = th.cat(tensor_list, dim=0)

    # 获取前 10 个最大值及其索引
    values, indices = th.topk(new_tensor, 10, dim=1)

    # 将索引转换为角度值: -pi ~ +pi
    topk_angle = (np.array(indices)*10+5)/360*2*pi-pi

    # 返回前 k 个角度值， k = 3
    return topk_angle[:, :k]
