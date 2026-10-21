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
import scipy
import numpy as np
import pandas as pd
import iotbx.pdb
import mmtbx.model
from rdkit import Chem
from utils.Utils import GetAtoms
from collections import defaultdict


def IsinGrid(x, y, z, centerx, centery, centerz, grid_size):
    """
    Check a point in the grid
    """
    if (abs(centerx -x) <= grid_size) and (abs(centery -y) <= grid_size) and (abs(centerz -z) <= grid_size):
        return 1
    else:
        return 0

# 提取位于特定网格点附近的原子信息
def GetPocAtoms(pdbf, cx, cy, cz):
    atomlines = GetAtoms(pdbf)
    pocatoms = defaultdict(list)
    for line in atomlines:
        name = line[17: 20].strip()
        chain = line[21]
        num = int(line[22: 26].strip())
        x = float(line[30: 38].strip())
        y = float(line[38: 46].strip())
        z = float(line[46: 54].strip())
        atomtype = (line[76:78].strip())[0].upper()
        # 检查原子是否在盒子内，半边长为 15 
        if IsinGrid(x, y, z, cx, cy, cz, 15):
            pocatoms["name"].append(name)
            pocatoms["chain"].append(chain)
            pocatoms["num"].append(num)
            pocatoms["x"].append(x)
            pocatoms["y"].append(y)
            pocatoms["z"].append(z)
            pocatoms["atomtype"].append(atomtype)
    pocatomdf = pd.DataFrame(pocatoms)
    return pocatomdf


def Voxelize(centroid, GridSize=12, SpacingCutoff=0.5):
    xv = np.arange(0, GridSize*2, SpacingCutoff)
    yv = np.arange(0, GridSize*2, SpacingCutoff)
    zv = np.arange(0, GridSize*2, SpacingCutoff)
    Grid_x, Grid_y, Grid_z = np.meshgrid(xv, yv, zv)
    Grid_x = Grid_x + centroid[0] - GridSize   # 移动到给定的中心点
    Grid_y = Grid_y + centroid[1] - GridSize
    Grid_z = Grid_z + centroid[2] - GridSize
    points = np.hstack((Grid_x.flatten().reshape((-1, 1)),
                        Grid_y.flatten().reshape((-1, 1)),
                        Grid_z.flatten().reshape((-1, 1))))
    # 使用 flatten 将三维网格坐标数组 Grid_x, Grid_y, Grid_z 扁平化为一维数组，然后通过 reshape((-1, 1)) 将它们转换为列向量。
    # 使用 np.hstack 将这三个列向量水平堆叠起来，形成一个形状为 (n, 3) 的二维数组 points，其中 n 是网格点的总数。
    voxpoints = points + SpacingCutoff/2   # 将网格点移动到网格单元的中心
    return voxpoints


# 计算蛋白质晶体结构中电子密度（或称为结构因子）。使用cctbx（一个用于晶体学计算的C++库）的Python接口。
def Fcalc(pdbf,voxpoints,resolution=2.0):
    pdb_inp = iotbx.pdb.input(file_name = pdbf)
    model = mmtbx.model.manager(model_input=pdb_inp)
    xrs = model.get_xray_structure() # 从模型管理器中获取X射线结构对象
    fcalc = xrs.structure_factors(d_min = resolution).f_calc() # 使用X射线结构对象计算结构因子
    fft_map = fcalc.fft_map(resolution_factor = 0.25) # 通过结构因子计算FFT（快速傅里叶变换）电子密度图
    fft_map.apply_volume_scaling()  # 应用体积缩放
    fcalc_map_data = fft_map.real_map_unpadded()  # 获取未经过填充处理的真实密度图数据
    uc = fft_map.crystal_symmetry().unit_cell()  # 从密度图对象中获取晶体对称性和单元细胞信息
    rho_fcalc = []  # 初始化一个空列表rho_fcalc用于存储密度值
    for p in voxpoints:
        frac = uc.fractionalize(p)  # 将点p转换为分数坐标（相对于晶体单元）
        density = fcalc_map_data.value_at_closest_grid_point(frac)  # 在密度图中找到最接近该分数坐标的网格点的密度值。
        rho_fcalc.append(density)
    
    rho_fcalc = np.array(rho_fcalc).reshape(-1,1)  # 调整形状为列向量
    rho_fcalc = np.where(rho_fcalc < 0, 0, rho_fcalc)  # 使用np.where将所有负密度值替换为0（因为电子密度在物理上不应为负）
    return rho_fcalc


def FcalcPdb(pdbf, cx, cy, cz, tmpdir):
    """
    根据给定的PDB文件，计算其电子密度分布，并返回体素点和对应的电子密度分布。

    Args:
    pdbf (str): 输入的PDB文件路径。
    cx (float): 中心点的x坐标。

    Returns:
    tuple: 包含两个元素的元组，第一个元素是体素点的坐标数组，第二个元素是对应的电子密度分布数组。

    """
    pdbname2 = os.path.join(tmpdir, "rec_cell.pdb")
    with open(pdbname2, "w") as op:
        op.writelines("CRYST1   90.000   90.000   90.000  90.00  90.00  90.00 P 1           3          \n")
        for lines in open(pdbf).readlines():
            if "ANISOU" not in lines:  # 排除掉 "ANISOU"，其记录提供了原子的各向异性温度因子
                lines = lines[:56]+"1.00"+lines[60:61]+" 0.00"+lines[66:]
                # 忽略各向异性温度因子，只保留各向同性温度因子，前56个字符。这通常包括原子名称、残基名称、链标识符、残基序号、原子序号、x坐标、y坐标和z坐标
                # 第67列及之后通常包含其他信息，如元素符号、电荷等
                op.writelines(lines)
        op.close()    
    
    pocatoms = GetPocAtoms(pdbf, cx, cy, cz)    # 获取 cx, cy, cz 附近的原子
    pocatomcoords = pocatoms[["x", "y", "z"]].to_numpy()
    voxpoints = Voxelize([cx, cy, cz], GridSize = 12, SpacingCutoff = 0.5)
    rho_fcalc = Fcalc(pdbname2,voxpoints)
    dist = scipy.spatial.distance.cdist(voxpoints, pocatomcoords, metric = "euclidean")
    ptable = Chem.GetPeriodicTable()
    CutOff = np.array([ptable.GetRvdw(element) for element in pocatoms["atomtype"]])
    interval = (dist-CutOff)
    poc_close_p = np.where(interval < 0)[0]  # 找出距离小于范德华半径的体素点（即接近原子的点）
    rho_fcalc_pdb = np.zeros((48*48*48, 1))  # 初始化一个全零的电子密度数组 rho_fcalc_pdb，大小为 48x48x48
    rho_fcalc_pdb[poc_close_p, 0] = rho_fcalc[poc_close_p, 0]  # 将接近原子的体素点的电子密度值赋给 rho_fcalc_pdb 对应位置
    os.remove(pdbname2)

    return voxpoints, rho_fcalc_pdb