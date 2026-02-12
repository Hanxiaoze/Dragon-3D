import math
import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from itertools import product
from sklearn.cluster import DBSCAN
from rdkit.Geometry import Point3D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def GetMolCenter(mol):
    center = np.mean(np.array(mol.GetConformer().GetPositions()), 0) 
    return center


# 平移母核
def CoreTranslate(Core, TargetCenter):   
    conf = Core.GetConformer()
    positions = conf.GetPositions()
    center = GetMolCenter(Core)
    new_positions = np.array(positions)+np.array(TargetCenter)-np.array(center)
    for i in range(Core.GetNumAtoms()):
        X,Y,Z = np.array(new_positions)[i]
        Core.GetConformer().SetAtomPosition(i, Point3D(X,Y,Z))     
    Core.UpdatePropertyCache()
    return Core


# 旋转母核
def CoreRotate(Core, Pitch, Yaw, Roll):    
    """
    Core: 分子母核
    Pitch, Yaw, Roll: 分别表示绕 X 轴、Y 轴和 Z 轴的旋转角度，以弧度为单位。
    """
    core_copy = copy.deepcopy(Core)  # 深度复制 Core 对象  
    center = GetMolCenter(core_copy)  # 获取分子的中心坐标   
    conf = core_copy.GetConformer()  # 获取分子的构象
    
    # 获取分子的原子位置
    positions = conf.GetPositions()
    positions = np.array(positions)
    
    # 创建绕 X 轴旋转的矩阵
    ex_arr = np.array([[1,0,0],[0,math.cos(Pitch),-math.sin(Pitch)],[0,math.sin(Pitch),math.cos(Pitch)]])
    # 创建绕 Y 轴旋转的矩阵
    ey_arr = np.array([[math.cos(Yaw), 0, math.sin(Yaw)],[0, 1, 0],[-math.sin(Yaw), 0,math.cos(Yaw)]])
    # 创建绕 Z 轴旋转的矩阵
    ez_arr = np.array([[math.cos(Roll), -math.sin(Roll),0],[math.sin(Roll), math.cos(Roll), 0],[0, 0,1]])
    
    # 对分子位置进行平移，旋转，平移
    temp = np.transpose(ez_arr.dot(np.transpose(positions-center)))
    temp = np.transpose(ey_arr.dot(np.transpose(temp)))
    temp = np.transpose(ex_arr.dot(np.transpose(temp)))
    new_positions = temp+center
    
    # 更新分子的原子位置
    for i in range(core_copy.GetNumAtoms()):
        X,Y,Z = np.array(new_positions)[i]
        core_copy.GetConformer().SetAtomPosition(i, Point3D(X,Y,Z)) 
    
    # 更新分子的属性缓存
    core_copy.UpdatePropertyCache()
    
    # 返回旋转后的分子对象
    return core_copy


def RetainMaxValues(tensor, alpha=0.4):
    """
    该函数的主要目的是通过阈值筛选和最大池化操作，保留输入张量中的显著最大值，
    并将其他值设为零。这在某些特征提取或图像处理任务中可能非常有用。
    """
    
    stride = (1, 1, 1)     # 池化步长
    pool_size = (4, 4, 4)  # 池化大小
    
    tensor_copy = tensor.clone()  # 创建张量的副本    
    max_value = tensor_copy.max().item()  # 找到张量中的最大值    
    threshold = max_value * alpha   # 计算阈值    
    tensor_copy[tensor_copy < threshold] = 0   # 将小于阈值的元素设置为 0

    # 使用 最大池化 操作获取最大池化和索引
    max_values, indices = F.max_pool3d(tensor_copy, pool_size, stride, return_indices=True)
    
    result = th.zeros_like(tensor_copy)  # 创建一个与张量副本形状相同的零张量    
    result = result.view(-1)  # 将结果张量展平    
    result[indices.view(-1)] = tensor_copy.view(-1)[indices.view(-1)]  # 使用索引将最大值复制到结果张量中    
    result = result.view(tensor_copy.shape)  # 将结果张量恢复为原始形状
    
    mask = max_values != 0  # 创建一个掩码，标记最大值不为 0 的位置    
    filtered_indices = indices[mask]  # 过滤索引，只保留最大值不为0的索引
    
    return result


def GetClusterCenters(merged_arr, eps=1, min_samples=2):
    """
    Get the centers of the clusters
    """
    center_list = []
    merged_arr = merged_arr[merged_arr[:, -1] != 0]
    merged_arr = merged_arr[:, :3]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(merged_arr)
    unique_values, counts = np.unique(labels, return_counts=True)

    sorted_indices = np.argsort(counts)[::-1]
    top_three_unique = unique_values[sorted_indices][:3]

    for label in top_three_unique:
        indices = labels == label
        cluster_points = merged_arr[indices]
        center_list.append(np.mean(cluster_points, axis=0))
    return center_list 


def CoreId(core, cluster_centers, num_divisions=9):
    new_cores = []
    name = None
    if core.HasProp("_Name"):
        name = core.GetProp("_Name")
        print('当前核心名字： ', name)
    # 遍历聚类中心
    for center in (cluster_centers):       
        core = CoreTranslate(core, center)   # 将母核平移至聚类中心
        # 生成所有可能的旋转组合
        for x_a, y_a, z_a in product(range(num_divisions), range(num_divisions), range(num_divisions)):             
            core = CoreRotate(core, x_a, y_a, z_a)  # 将母核旋转    
            if name:
                core.SetProp("_Name", name)      
            new_cores.append(core)
    return new_cores