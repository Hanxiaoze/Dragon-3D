import os
import stat
import copy
import math
import random
import subprocess
import configparser
import multiprocessing
import torch as th
import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import label
from itertools import permutations
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdMolTransforms import GetDihedralDeg, GetDihedralRad, SetBondLength, SetDihedralRad
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs import BulkTanimotoSimilarity


def SetSeed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def GetConfigs(config_path):
    configs = configparser.ConfigParser()
    configs.read(config_path)
    return configs


def MaxIteration(EDgrid, assign_iter, vol=1000):
    # 如果assign_iter有值，则直接使用assign_iter的值
    if assign_iter:
        iteration = int(assign_iter)
    else:
        # 筛选出EDgrid中第4列大于0的行
        grid_num = EDgrid[EDgrid[:,3]>0].shape[0]
        # 计算迭代次数，最大为5，最小为2，根据grid_num和vol计算
        iteration = min(max(round(grid_num / vol), 2), 5)
    # 返回迭代次数
    return iteration


def Segment3DGrid(grid):
    # 使用label函数对grid进行标记，返回标记后的数组和特征数量
    labeled_array, num_features = label(grid)   # 调用 scipy.ndimage.label 连通区域分割函数
    
    regions = []
    # 遍历每一个特征标签
    for label_num in range(1, num_features + 1):
        # 获取当前标签对应的坐标集合
        region = [(x, y, z) for x, y, z in zip(*np.where(labeled_array == label_num))]
        # 将当前标签对应的坐标集合添加到regions列表中
        regions.append(region)       
    
    return regions


def GetAtoms(pdbf):
    atomlines = []
    for line in open(pdbf).readlines():
        if line[:6] in ["ATOM  ", "HETATM"]:
            atomlines.append(line)
    return atomlines


def GetRatom(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ["R","*"]:
            break
    return atom

def GetRu_atom(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ["Ru"]:
            break
    return atom


def BondNeighbors(mol, a1, a2):
    """
    if x-a1-a2-y, get the neighbor atoms x and y of a1 and a2
    """
    a1n = [n.GetIdx() for n in mol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2]
    a2n = [n.GetIdx() for n in mol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1]
    return a1n ,a2n


def CalcAlpha(mol, a1, a2):
    """
    x and y are neighbors of a1 and a2, alpha is the angle of anta2(s[0], s[1]), where s is sum of cosxy and sinxy
    """
    a1n, a2n = BondNeighbors(mol, a1, a2)
    s = []
    conf = mol.GetConformer()
    for x in a1n:
        for y in a2n:
            angle = GetDihedralDeg(conf, x, a1, a2, y)
            cosxy = math.cos(math.pi*angle/180)
            sinxy = math.sin(math.pi*angle/180)
            s.append([cosxy, sinxy])
    s = 1000000*np.sum(np.array(s), 0)
    alpha = -math.atan2(s[0], s[1])#顺时针度数
    return alpha


def ConnectMols(mol1, mol2, atom1, atom2):
    """
    function borrowed from https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py
    """
    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()   # atom2 只有一个键
    emol.AddBond(neighbor1_idx,
                 neighbor2_idx + mol1.GetNumAtoms(),
                 order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    if neighbor1_idx > atom1_idx:
        update_nei1_idx = neighbor1_idx - 1
    else:
        update_nei1_idx = neighbor1_idx
    if neighbor2_idx > atom2_idx:
        update_nei2_idx = mol1.GetNumAtoms() - 1 + neighbor2_idx - 1
    else:
        update_nei2_idx = mol1.GetNumAtoms() - 1 + neighbor2_idx
    return mol, update_nei1_idx, update_nei2_idx


def RotateFragment(core, frag, torsion_angle, Set_bond_length):
    """
    torsion 角度的“0°”到底该指向哪? 这就是 CalcAlpha 计算平均角度, 消除了“邻居选择”的偶然性
        一个二面角是由 (x-a1-a2-y) 定义的。
        但是 a1、a2 各自可能有多个邻居：
            a1 有邻居 [x1, x2, …]
            a2 有邻居 [y1, y2, …]
        那就存在多个可能的二面角组合 (x_i, a1, a2, y_j)。
        如果只选一个邻居来当参考角度，会带来 不稳定性/偏差：
            有时选择的邻居不同 → fragment 接上去的“参考0°”会变。
            结果就是：同一个 torsion_angle, 得到的构象会不一致
    """
    atom1 = GetRatom(core)
    atom2 = GetRatom(frag)
    rdMolAlign.AlignMol(frag, core, atomMap=[(atom2.GetIdx(), atom1.GetNeighbors()[0].GetIdx()), (atom2.GetNeighbors()[0].GetIdx(), atom1.GetIdx())])
    newmol, a1, a2 = ConnectMols(core, frag, atom1, atom2)
    Chem.SanitizeMol(newmol)
    conf = newmol.GetConformer()
    x = [n.GetIdx() for n in newmol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2][0]
    y = [n.GetIdx() for n in newmol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1][0]
    angle = (-CalcAlpha(newmol, a1, a2)+torsion_angle)   # -现平均角度+真平均角度
    alpha_ori = GetDihedralRad(conf, x, a1, a2, y)    # 现二面角
    alpha = angle + alpha_ori    # 真二面角 = -现平均角度+真平均角度+现二面角
    #set dihedral, change bond length
    SetDihedralRad(conf, x, a1, a2, y, alpha)  # 设置 真二面角
    SetBondLength(conf, a1, a2, Set_bond_length)
    return newmol


def RotateFragment_zzx(core, frag, torsion_angle, Set_bond_length):
    """
    torsion 角度的“0°”到底该指向哪? 这就是 CalcAlpha 计算平均角度, 消除了“邻居选择”的偶然性
        一个二面角是由 (x-a1-a2-y) 定义的。
        但是 a1、a2 各自可能有多个邻居：
            a1 有邻居 [x1, x2, …]
            a2 有邻居 [y1, y2, …]
        那就存在多个可能的二面角组合 (x_i, a1, a2, y_j)。
        如果只选一个邻居来当参考角度，会带来 不稳定性/偏差：
            有时选择的邻居不同 → fragment 接上去的“参考0°”会变。
            结果就是：同一个 torsion_angle, 得到的构象会不一致
    """
    atom1 = GetRatom(core)
    atom2 = GetRu_atom(frag)
    rdMolAlign.AlignMol(frag, core, atomMap=[(atom2.GetIdx(), atom1.GetNeighbors()[0].GetIdx()), (atom2.GetNeighbors()[0].GetIdx(), atom1.GetIdx())])
    newmol, a1, a2 = ConnectMols(core, frag, atom1, atom2)
    Chem.SanitizeMol(newmol)
    conf = newmol.GetConformer()
    x = [n.GetIdx() for n in newmol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2][0]
    y = [n.GetIdx() for n in newmol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1][0]
    angle = (-CalcAlpha(newmol, a1, a2)+torsion_angle)   # -现平均角度+真平均角度
    alpha_ori = GetDihedralRad(conf, x, a1, a2, y)    # 现二面角
    alpha = angle + alpha_ori    # 真二面角 = -现平均角度+真平均角度+现二面角
    #set dihedral, change bond length
    SetDihedralRad(conf, x, a1, a2, y, alpha)  # 设置 真二面角
    SetBondLength(conf, a1, a2, Set_bond_length)
    return newmol


def CleanMol(mol):
    copy_mol = copy.deepcopy(mol)
    copy_mol = Chem.AddHs(copy_mol,addCoords = True)
    for a in copy_mol.GetAtoms():
        if a.GetSymbol() in ["R", "*", "H"]:
            a.SetAtomicNum(1)
    return copy_mol


# 检查原子间碰撞，环内原子间允许距离近
def CheckAtomCol(mol):    
    mol = CleanMol(mol)  # 清理分子结构    
    dm = Chem.Get3DDistanceMatrix(mol)  # 获取分子的 3D 距离矩阵    
    R = []  # 初始化一个空列表用于存储原子间的范德华半径之和的 60 %    
    n = mol.GetNumAtoms()  # 获取分子中的原子数量
    # 遍历分子中的所有原子
    for a1 in mol.GetAtoms():
        # 获取第一个原子的范德华半径
        rvdw1 = Chem.GetPeriodicTable().GetRvdw(a1.GetAtomicNum())
        # 再次遍历分子中的所有原子
        for a2 in mol.GetAtoms():
            # 获取第二个原子的范德华半径
            rvdw2 = Chem.GetPeriodicTable().GetRvdw(a2.GetAtomicNum())
            # 计算两个原子的范德华半径之和的60%，并添加到列表 R 中
            R.append((rvdw1 + rvdw2) * 0.6)
    # 将列表 R 转换为numpy数组，并重塑为 n 行 n 列的矩阵
    R = np.array(R).reshape(n, n)

    # 获取分子中的所有环
    rings = mol.GetRingInfo().AtomRings()
    # 获取分子的邻接矩阵，并添加一个单位矩阵以确保主对角线元素为 1
    am = Chem.GetAdjacencyMatrix(mol) + np.eye(n, dtype=int)
    # 遍历所有环
    # 如果仅依赖原始邻接矩阵，可能会忽略环内原子间的空间接近性，导致误判为碰撞
    for r in rings:
        # 获取环中所有原子的两两组合
        row = [i[0] for i in permutations(r, 2)]
        col = [i[1] for i in permutations(r, 2)]
        # 将邻接矩阵中对应位置的值设置为 1
        am[row, col] = 1
    
    # 计算 3D 距离矩阵中非邻接原子对之间的距离是否小于范德华半径之和的 60%
    coll = dm[np.logical_not(am)] < R[np.logical_not(am)]
    # 如果存在至少一对非邻接原子之间的距离小于范德华半径之和的 60%，则返回 False
    if coll.sum() > 0:
        return False
    else:
        # 否则返回True
        return True


def CalculateAngleBetweenVectors(vector1, vector2):
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle_in_radians = np.arccos(cosine_angle)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def Minimize(input_sdf, configs):
    smina = "./utils/smina.static"
    if not os.access(smina, os.X_OK):
        os.chmod(smina, os.stat(smina).st_mode | stat.S_IXUSR)
    try:
        pdb = configs.get('sample', 'receptor')
    except:
        pdb = configs.get('sample', 'receptor_A')
    tmpdir = os.path.join(configs.get('sample', 'output_dir'), 'tmp')
    output_sdf = os.path.join(tmpdir, input_sdf.split("/")[-1].split(".sdf")[0]+"_lig_score.sdf")
    command = f"{smina} -r {pdb} -l {input_sdf} -o {output_sdf} --minimize --minimize_iters 1000 --cpu 1"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return output_sdf


def MarkAxialBonds(mol):
    """
    detect axial bonds
    检测并标记分子中的轴向键(axial bonds)。轴向键通常指的是在环状结构中, 与环平面近似垂直的化学键
    """
    TargetSmiles = "C1****C1"  # 定义了一个SMARTS模式，用于匹配包含六个任意原子的六元碳环
    mol = Chem.AddHs(mol,addCoords=True)
    substructure = Chem.MolFromSmarts(TargetSmiles)
    matches = mol.GetSubstructMatches(substructure)
    Positions = np.array(mol.GetConformer().GetPositions())
    # 遍历匹配环
    for match in matches: 
        RingPositions = Positions[np.array(match).reshape(-1)]
        x, y, z = RingPositions.T
        A = np.column_stack((x, y, np.ones_like(x)))
        coefficients, _, _, _ = lstsq(A, z)
        a, b, c = coefficients
        NormalVector = np.array([a,b,-1])   # 使用最小二乘法拟合平面，得到平面的法向量
        # 遍历环中原子
        for i in match:
            Atom = mol.GetAtoms()[i]
            AtomPosition = Positions[i]
            angles = []
            for Neighbor in Atom.GetNeighbors():
                if Neighbor.GetAtomicNum() == 1:   # 只考虑邻居原子是 氢原子
                    NeighborPosition = Positions[Neighbor.GetIdx()]
                    BondVector = NeighborPosition - AtomPosition
                    angle = CalculateAngleBetweenVectors(BondVector,NormalVector)
                    angle = abs(angle-90)
                    angles.append(angle)
                    if angle > 60:
                        Neighbor.SetIsotope(2)  # 标记角度大于 60度的氢原子
            # 如果环原子恰好有两个氢原子邻居，找到夹角最大的氢原子，并将其最终标记为同位素 2
            if len(angles) == 2:
                AxialIndex = np.argmax(angles)  # 找到夹角最大的氢原子
                index = 0
                for Neighbor in Atom.GetNeighbors():
                    if Neighbor.GetAtomicNum() == 1:
                        if AxialIndex == index:
                            Neighbor.SetIsotope(2)
                        index+=1    
    return copy.deepcopy(mol)


# 使用smina软件，多进程优化分子
def MolOpt(mols, configs):
    tmpdir = os.path.join(configs.get('sample', 'output_dir'), 'tmp')
    num_cpu = int(configs.get('sample', 'num_cpu'))
    if len(mols) > num_cpu:
        pass
    else:
        num_cpu = len(mols)
    chunksize = math.ceil(len(mols) / num_cpu)
    mol_chunks = [mols[i:i+chunksize] for i in range(0, len(mols), chunksize)]
    input_sdfs = []
    for index,mols in enumerate(mol_chunks):
        writer = Chem.SDWriter(os.path.join(tmpdir,str(index)+".sdf"))
        input_sdfs.append(os.path.join(tmpdir,str(index)+".sdf"))
        for mol in mols:
            writer.write(mol)
        writer.close()
    pool = multiprocessing.Pool()
    results = [pool.apply_async(Minimize, args=[args, configs]) for args in input_sdfs]  # Minimize会调用smina软件
    pool.close()
    pool.join()
    optimized_molecules = []
    for result in results:
        output_sdf = result.get() 
        for mol in Chem.SDMolSupplier(output_sdf):
            if float(mol.GetProp("minimizedAffinity"))<0:
                optimized_molecules.append(mol)
        os.remove(output_sdf)
    for input_sdf in input_sdfs:
        os.remove(input_sdf)
    return optimized_molecules


def DistFromMol2Coords(mol, coord_array):
    mol_xyz = mol.GetConformer().GetPositions()
    distances = np.linalg.norm(coord_array - mol_xyz[:, np.newaxis], axis=2)
    min_distances = np.min(distances, axis=0)

    return np.min(min_distances)


def SortMolByDist(mols, coord_array):
    all_min_distances = [DistFromMol2Coords(mol, coord_array) for mol in mols]
    sorted_mols = [mol for _, mol in sorted(zip(all_min_distances, mols), key=lambda x: -np.max(x[0]))]
    return sorted_mols


def GetResCoords(res, recf):
    # get residue coordinates
    # e.g. res: "A809 sidechain"
    configs = GetConfigs()
    recf = configs.get('sample', 'receptor')
    chainid = res.split()[0][0]
    resnum = res.split()[0][1:]
    bbcoords = []; sccoords = []
    atmlines = GetAtoms(recf)
    for line in atmlines:
        if line[21] == chainid and line[22:26].strip() == resnum:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if line[12:16].strip() in ["N", "CA", "C", "O"]:
                bbcoords.append([x, y, z])
            else:
                sccoords.append([x, y, z])
    res_coords = np.array(bbcoords) if res.split()[1] == "backbone" else np.array(sccoords)
    return res_coords


class MolBucket:
    """
    put diverse molecules derived from the same core into the bucket with limited size
    把起源与同一母核的相似的分子放到桶里
    """
    def __init__(self, bucket_size, sim_threshold):
        self.bucket_size = bucket_size
        self.sim_threshold = sim_threshold

    def update(self, idx, query_mol, bucket_fps):
        """
        Calculate ECFP similiarity and decide whether to add the query into the bucket
        """
        qfp = GetMorganFingerprint(query_mol, 2)
        # tanimoto similarity profile
        sim = BulkTanimotoSimilarity(qfp, list(bucket_fps.values()))
        if sorted(sim)[-1] <= self.sim_threshold:
            bucket_fps.update({idx: qfp})
        
        return bucket_fps

    def add(self, ranked_mols):
        """
        Add mols on the ranked list sequentially into the bucket
        """
        bucket_fps = {0: GetMorganFingerprint(Chem.RemoveHs(ranked_mols[0]), 2)}
        for idx, m in enumerate(ranked_mols):
            if m:
                m = Chem.RemoveAllHs(m)
                if len(bucket_fps) < self.bucket_size:
                    bucket_fps = self.update(idx, m, bucket_fps)
                else:
                    break
        
        return [ranked_mols[idx] for idx in bucket_fps]