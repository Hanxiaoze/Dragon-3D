import pandas as pd
import torch
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
import copy
import random
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch as th
from tqdm import tqdm
from rdkit import Chem
from traceback import print_exc
from scipy.spatial import KDTree
from models.ED_Generator import ED_generator
from models.Mol_Generator import GrowthPointPredictionModel, TorsionAnglePredictionModel
from utils.Score import Scorer
from utils.EDExtract import FcalcPdb
from utils.CorePlacement import RetainMaxValues, GetClusterCenters, CoreId
from utils.DataProcessing import DatasetGrowthPoint, GrowthPointFilter, Attach
from utils.Utils import *
from zzx_GFPM import Frag_Pred_Model
from multiprocessing import Pool

def LoadModels(configs):
    gppm_path = configs.get('model', 'GPPM')
    gfpm_path = configs.get('model', 'GFPM')
    tapm_path = configs.get('model', 'TAPM')
    device_type = configs.get('sample', 'device_type')
    device = th.device(device_type)

    GPPM = GrowthPointPredictionModel().to(device)
    GPPM.load_state_dict(th.load(gppm_path, map_location=th.device(device_type)))
    GPPM.eval()

    GFPM = Frag_Pred_Model().to(device)
    GFPM.load_state_dict(th.load(gfpm_path, map_location=th.device(device_type)))
    GFPM.eval()

    TAPM = TorsionAnglePredictionModel().to(device)
    # pretrained_dict = th.load(tapm_path, map_location=th.device(device_type))
    # model_dict = TAPM.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # TAPM.load_state_dict(model_dict)
    TAPM.load_state_dict(th.load(tapm_path, map_location=th.device(device_type)))
    TAPM.eval()

    # print('成功运行到这里啦222222！！！！！！')

    return GPPM, GFPM, TAPM


def LoadLibs(configs):
    cores = Chem.SDMolSupplier(configs.get('lib', 'Cores'))
    cores = [mol for mol in cores]
    frags = Chem.SDMolSupplier(configs.get('lib', 'Frags'))
    frags = [mol for mol in frags]
    return cores, frags

from utils.EDExtract import Voxelize
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from biNet_VAE_model_with_SO3 import *
from zzx_pocED_2_ligED_train_biNet_0 import infer_biNet_VAE

class Pocket_dual_infer_dataset(Dataset):
    def __init__(self, pock_ed_1, pock_ed_2, dtype=th.float32):
        super().__init__()
        self.dtype = dtype
        self.pock_ed_1_data = []
        self.pock_ed_2_data = []
        if np.shape(pock_ed_1) == (48*48*48, 1): 
            self.pock_ed_1_data.append(pock_ed_1.squeeze().reshape((48, 48, 48))[None, ...])
            self.pock_ed_2_data.append(pock_ed_2.squeeze().reshape((48, 48, 48))[None, ...])
        else:
            print('Data shape for pocket ED should be (48*48*48, 1) !')
    def __len__(self):
        return len(self.pock_ed_1_data)
    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor]:            
        pocket_A = th.tensor(self.pock_ed_1_data[idx], dtype=self.dtype)   # (1,48,48,48)
        pocket_B = th.tensor(self.pock_ed_2_data[idx], dtype=self.dtype)   # (1,48,48,48) 
        return pocket_A, pocket_B
    

def LigEDgen(configs, output_dir, tmp_dir):
    """
    build ligand ED
    recpdb: receptor PDB file
    center_x, center_y, center_z: pocket center
    """
    pocgrid_A, pocED_A = FcalcPdb(configs.get('sample', 'receptor_A'), float(configs.get('sample', 'x_A')), 
                     float(configs.get('sample', 'y_A')), float(configs.get('sample', 'z_A')), tmp_dir)
    pocgrid_B, pocED_B = FcalcPdb(configs.get('sample', 'receptor_B'), float(configs.get('sample', 'x_B')), 
                     float(configs.get('sample', 'y_B')), float(configs.get('sample', 'z_B')), tmp_dir)
    pocgrid = Voxelize((24.0,24.0,24.0), 12, 0.5)    # (48*48*48, 3)

    pocket_dual_infer_dataset = Pocket_dual_infer_dataset(pocED_A, pocED_B)
    infer_biNet_VAE_loader = DataLoader(pocket_dual_infer_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    batch_sampletime_results = infer_biNet_VAE(infer_biNet_VAE_loader, configs.get('model', 'Encode_pro'),
                                                configs.get('model', 'Decode_lig'), configs.get('model', 'Encode_lig'),
                                                configs.get('model', 'Fus_model'),"cuda", 1, configs.get('sample', 'output_dir')) 
    print('biNet lig result shape: ', batch_sampletime_results.shape)   
    
    ligED = torch.tensor(batch_sampletime_results)
    ligED[ligED<0.1]=0
    ligED = ligED.reshape(48,48,48)
    gengrid = np.hstack([pocgrid, ligED.reshape((-1,1))])
    # pocED = th.where(pocED.float().reshape(48,48,48) > 0, th.tensor(0), th.tensor(1)) # 创建一个与口袋电子密度形状相同的掩码（pocED），口袋外部为1，内部为0。
    # ligED = ligED*pocED  # 仅保留口袋内部的配体电子密度
    segments = Segment3DGrid(copy.deepcopy(ligED.reshape(48,48,48)))
    max_list = max(segments, key=len)  # 对配体电子密度进行3D网格分割，并找到最大的片段
    liggridED = th.zeros_like(ligED)
    for x,y,z in max_list:
        liggridED[x,y,z] = ligED[x,y,z]
    gengrid = np.hstack([pocgrid, np.array(liggridED.reshape(-1,1))])
    np.save(os.path.join(output_dir, "ligED.npy"), gengrid)
    
    with open(os.path.join(output_dir, "ligED.pdb"), 'w') as f:
        for temp in gengrid:
            coord = temp[:3]
            intensity = temp[3]
            if intensity.item() <= 0.0:
                continue
            intensity = intensity.item()
            f.write(f'ATOM  {10000:>4}  X   MOL     1    {coord[0]:>7.3f} {coord[1]:>7.3f} {coord[2]:>7.3f} 0.00  {intensity:>6.2f}      MOL\n')
        f.write('END\n')
        
    # print('成功运行到这里啦1111111！！！！！')
    return gengrid

from torch.utils.data import Dataset, DataLoader

class SingleMolDataset(Dataset):
    def __init__(self, core_graph, ED_tensor):
        super().__init__()
        if ED_tensor.dim() == 3:
            ED_tensor = ED_tensor.unsqueeze(0) # [1, D, H, W]
        self.graph = core_graph
        self.ED_tensor = ED_tensor.float()

    def __len__(self):
        return 1  # 只有一个分子

    def __getitem__(self, idx):
        return self.graph, self.ED_tensor
   
def collate_fn(batch):
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    core_graphs, EDs = zip(*batch)
    
    from torch_geometric.data import Batch
    batch_graph = Batch.from_data_list(list(core_graphs))
    batch_grid = torch.stack(EDs, dim=0)  # [B, C, D, H, W]
    return batch_graph, batch_grid

from rdkit import Chem

def replace_Ru_with_R(frag_pred_list):
    """
    将每个 fragment 中的 Ru 原子替换成占位原子 R (*)
    
    Args:
        frag_pred_list (List[Tuple[Chem.Mol, float]]): [(Mol, prob), ...]
    
    Returns:
        List[Chem.Mol]: 替换后的分子列表
    """
    new_frags = []
    for mol, _ in frag_pred_list:  # 只取 Mol
        emol = Chem.RWMol(mol)
        for atom in emol.GetAtoms():
            if atom.GetSymbol() == "Ru":
                atom.SetAtomicNum(0)  # dummy atom
        new_frags.append(emol.GetMol())
    return new_frags


from zzx_GFPM import Mol2Graph_for_frag_pred
import json
from rdkit.Chem import AllChem

def predict_topk_fragments(GFPM_model, core_mol, ED_tensor, k=5, device='cuda'):
    """
    给定一个 core (rdkit.Mol) 和 ED 网格张量，预测概率最高的前 k 个 fragment。
    Args:
        model: 已训练好的 Frag_Pred_Model_0 模型
        core_mol: rdkit.Chem.Mol 对象，表示核心分子
        ED_tensor: torch.Tensor, 形状 [C, X, Y, Z], 电子密度或3D网格
        frag_id2smiles: dict[int -> str], 模型输出类别对应的 SMILES
        k: int, 返回前 k 个片段
        device: 'cuda' 或 'cpu'
    Returns:
        List[Tuple[rdkit.Chem.Mol, float]]: [(fragment_mol, probability), ...]
    """    

    # === 1. 预处理分子 ===
    ED_center = (24.0, 24.0, 24.0)
    # with open("../cache_GFPM_dataset/frag_vocab.json", "r") as f:
    with open("./frag_vocab.json", "r") as f:
        frag_vocab = json.load(f)
    frag_id2smiles = {frag_vocab[smi]['id']:smi for smi in frag_vocab}
    core_graph = Mol2Graph_for_frag_pred(core_mol, ED_center)._create_graph()
    core_graph = core_graph.to(device)       

    dataset = SingleMolDataset(core_graph, ED_tensor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad(): 
        for batch_graph, batch_grid in loader:      
            core_batch = batch_graph

            # === 3. 前向传播 ===
            logits = GFPM_model(core_batch.edge_index, core_batch.x, core_batch.pos, core_batch.edge_attr, core_batch.batch,
                                batch_grid)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # [n_class]

            # === 4. 取前 k 个 fragment ===
            topk_probs, topk_idx = torch.topk(probs, k)
            topk_probs = topk_probs.cpu().numpy()
            topk_idx = topk_idx.cpu().numpy()

        results = []
        for idx, p in zip(topk_idx, topk_probs):
            smile = frag_id2smiles.get(int(idx), None)
            print(f'预测的前 {k} 个片断的SMILES: ', smile)
            if smile is not None:
                try:
                    frag_mol = Chem.MolFromSmiles(smile)
                    frag_mol = Chem.AddHs(frag_mol)
                    AllChem.EmbedMolecule(frag_mol, randomSeed=42, maxAttempts=10)
                    if frag_mol is not None:
                        frag_mol.SetProp("_Name", str(idx))
                        results.append((frag_mol, float(p)))
                except:
                        
                    fix_dict = {
                        '[Ru]c1ncnn1': '[Ru]-C1=NC=NN1',
                        '[Ru]c1ccnn1': '[Ru]-C1=CC=NN1',
                        '[Ru]c1cnnc1': '[Ru]-C1=CN=NC1'
                    }
                    if smile in fix_dict:
                        smile = fix_dict[smile]
                        frag_mol = Chem.MolFromSmiles(smile)
                        frag_mol = Chem.AddHs(frag_mol)
                        AllChem.EmbedMolecule(frag_mol, randomSeed=42, maxAttempts=10)
                        if frag_mol is not None:
                            frag_mol.SetProp("_Name", str(idx))
                            results.append((frag_mol, float(p)))
                    else:
                        print(f'RDkit 生成分子片段 {smile} 不在显式价键修复字典 fix_dict 中 。。。')

                    


    return results

from multiprocessing import Pool
from functools import partial

def process_one_CoreId_index(args):    
    idx, mol, cluster_centers, scorer, EDgrid, tolerence = args
    part_mols = CoreId(mol, cluster_centers)
    scoredcores, scores = scorer.QScore(part_mols, EDgrid, tol=tolerence)
    resu = []
    # 返回 index 带回，用于恢复名字
    for i in range(1):  # top1
        resu.append((idx, scoredcores[i], scores[i]))
    return resu  

def parallel_CoreId_process(cleancores, cluster_centers, scorer, EDgrid, tolerence, num_workers=24):
    # 准备任务 (index, mol, cluster_centers, scorer, EDgrid, tolerence)
    tasks = [(i, mol, cluster_centers, scorer, EDgrid, tolerence)
             for i, mol in enumerate(cleancores)]
    
    with Pool(num_workers) as pool:
        results = pool.map(process_one_CoreId_index, tasks)

    # flatten
    aligned_mols_with_index = [item for sublist in results for item in sublist]

    # 根据 index 恢复名字
    for idx, mol, score in aligned_mols_with_index:
        name = cleancores[idx].GetProp("_Name") if cleancores[idx].HasProp("_Name") else ""        
        if name:
            mol.SetProp("_Name", name)        
        mol.SetProp("qscore", str(score))

    # 只返回 mol 列表
    aligned_mols = [mol for idx, mol, score in aligned_mols_with_index]
    return aligned_mols
    
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdGeometry
import math

def fix_SO2H_func(mol):
    pattern = Chem.MolFromSmarts("[S](=O)(=O)[H]")
    S_C_DIST = 1.77780   # S-C bond length
    C_H_DIST = 1.10949   # C-H bond length
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""  
    qscore = mol.GetProp("qscore") if mol.HasProp("qscore") else ""  
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return mol        

    rw = Chem.RWMol(mol)
    conf = rw.GetConformer()

    for match in matches:
        S_idx, H_idx = match[0], match[-1]

        # H → C
        at = rw.GetAtomWithIdx(H_idx)
        at.SetAtomicNum(6)
        at.SetFormalCharge(0)
        at.SetNoImplicit(True)
        at.SetNumExplicitHs(0)  # 我们手动添加 H

        # 调整 S–C 距离
        pS = conf.GetAtomPosition(S_idx)
        pC = conf.GetAtomPosition(H_idx)
        vx, vy, vz = pC.x - pS.x, pC.y - pS.y, pC.z - pS.z
        v_norm = math.sqrt(vx*vx + vy*vy + vz*vz)
        if v_norm < 1e-6:
            vx, vy, vz = 1.0, 0.0, 0.0
            v_norm = 1.0
        ux, uy, uz = vx/v_norm, vy/v_norm, vz/v_norm
        newC_pos = rdGeometry.Point3D(
            pS.x + ux * S_C_DIST,
            pS.y + uy * S_C_DIST,
            pS.z + uz * S_C_DIST
        )
        conf.SetAtomPosition(H_idx, newC_pos)

        # === 手动添加三个 H ===
        # 生成三个 H 的理想正四面体方向
        import numpy as np
        # 基于 S→C 方向
        bond_vec = np.array([ux, uy, uz])
        # 找两个垂直向量构建局部坐标系
        if abs(bond_vec[0]) < 0.9:
            ortho1 = np.array([1.0,0.0,0.0])
        else:
            ortho1 = np.array([0.0,1.0,0.0])
        ortho1 -= ortho1.dot(bond_vec)*bond_vec  # 正交化
        ortho1 /= np.linalg.norm(ortho1)
        ortho2 = np.cross(bond_vec, ortho1)

        # 正四面体角度约 109.5°
        angle = 109.5 / 180 * np.pi
        h_positions = []
        for phi in [0, 2*np.pi/3, 4*np.pi/3]:
            dir_vec = (
                -bond_vec*np.cos(angle) + 
                ortho1*np.sin(angle)*np.cos(phi) + 
                ortho2*np.sin(angle)*np.sin(phi)
            )
            h_pos = np.array([newC_pos.x, newC_pos.y, newC_pos.z]) + dir_vec*C_H_DIST
            h_positions.append(h_pos)

        for h_pos in h_positions:
            H_idx_new = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(H_idx, H_idx_new, Chem.rdchem.BondType.SINGLE)
            conf.SetAtomPosition(H_idx_new, rdGeometry.Point3D(*h_pos))

    mol2 = rw.GetMol() 
    # 优化分子结构：   (UFF 优化需要原子的 explicit valence 和 implicit valence 一致)
    Chem.SanitizeMol(mol2)   #先更新原子信息
    Chem.AssignAtomChiralTagsFromStructure(mol2)
    # AllChem.UFFOptimizeMolecule(mol2)
    mol2.SetProp("_Name", name)
    mol2.SetProp("qscore", qscore)
    return mol2

def MolGrow(EDgrid, configs):
    '''
    Main function for generating molecules.
    '''
    initial_cores, frags = LoadLibs(configs)
    frags = frags[ :8]       # 可调
    # 加载模型
    GPPM, GFPM, TAPM = LoadModels(configs)
    scorer = Scorer(configs)
    # 获取参考核心
    ref_core = configs.get('sample', 'reference_core')
    # 获取容忍度
    tolerence = float(configs.get('sample','tolerence')) if configs.has_option('sample','tolerence') else 0.0

    try:
        # 获取生长点位置
        growth_point_pos = [float(configs.get('sample', 'grow_x')), float(configs.get('sample', 'grow_y')), float(configs.get('sample', 'grow_z'))]
    except:
        # 如果没有指定生长点位置，则设置为 None
        growth_point_pos = None
    
    # 获取最大迭代次数
    iteration = MaxIteration(EDgrid, configs.get('sample', 'iteration'))
    print("**********Run {}-step Molecule Generation**********".format(str(iteration)))

    generate_molecules = []
    for step in range(iteration):
        pop_mols = []
        if not step:   # step = 0 最初第一步
            # 如果没有参考母核
            if not ref_core:                
                EDvalue = th.tensor(EDgrid)[:,-1].reshape(1,48,48,48)  # 获取电子密度值                
                max_grid = RetainMaxValues(EDvalue)  # 获取 最大池化 后的 电子密度值
                # 合并坐标和电子密度值
                merged_coords_max_grid = np.hstack((EDgrid[:, :-1], max_grid.numpy().reshape(-1, 1)))
                # 获取聚类中心
                cluster_centers = GetClusterCenters(merged_coords_max_grid)
                # 遍历初始母核，进行平移，旋转放置和评分
                # for core in tqdm(initial_cores[:10], desc="Placing Initial Fragments"):                   # 可调
                #     core_enumeration = CoreId(core, cluster_centers)   # 进行母核的平移，旋转放置
                #     core_mol_list, qscores = scorer.QScore(core_enumeration, EDgrid, tol=tolerence)
                #     pop_mols.append(core_mol_list[:2])                                                  # 可调
                #     print(pop_mols)

                aligned_mols = parallel_CoreId_process(initial_cores, cluster_centers, scorer, EDgrid, tolerence, num_workers=24)
                aligned_mols = sorted(aligned_mols, key=lambda x: float(x.GetProp('qscore')), reverse=True)
                pop_mols.append(aligned_mols[:300])                                                  # 可调
                print(pop_mols)

                    
            # 有参考母核
            else:
                # 检查参考母核的原子碰撞
                CheckAtomCol(Chem.SDMolSupplier(ref_core)[0])
                # 遍历参考母核，进行放置
                pop_mols = [[c] for c in tqdm(Chem.SDMolSupplier(ref_core), desc="Placing Reference Fragments") if c]
        
        else:            
            # 遍历母核，进行生长和评分
            EDvalue = th.tensor(EDgrid)[:,-1].reshape(1,48,48,48).to('cuda')
            for core in tqdm(cores, desc="Growing Fragments--Step {}".format(str(step+1))):
                # zzx新增的片断选择部分（cores是上一轮更新的，frags是我模型预测推荐的）
                frags_pred = predict_topk_fragments(GFPM, core, EDvalue, 64, 'cuda')    # 24 可调
                frags = replace_Ru_with_R(frags_pred)
                try:
                    # 连接碎片
                    cleancores = Attach(TAPM, frags, core, EDgrid, configs)

                    # zzx新增并行叠合, 并行调用 CoreId 和  scorer.QScore 进行叠合和打分                                     
                    aligned_mols = parallel_CoreId_process(cleancores, cluster_centers, scorer, EDgrid, tolerence, num_workers=24)
                    aligned_mols = sorted(aligned_mols, key=lambda x: float(x.GetProp('qscore')), reverse=True)
                    
                    pop_mols.append(aligned_mols[:200])                 # 可调
                    # pop_mols.append(cleancores)

                    for i in aligned_mols[:10]:
                        name = i.GetProp("_Name")
                        qscore = i.GetProp("qscore")
                        print('并行之后的核心名字和qscore： ', name, '     ', qscore)
                except:
                    # 打印异常信息
                    print_exc()
        
        grow_mols = []; candidate_mols = []; scored_mols = []
        # 获取相似性阈值
        sim_threshold = float(configs.get('sample', 'sim_threshold')) if step else 1.0
        # 获取桶大小
        bucket_size = float(configs.get('sample', 'bucket_size'))

        # 预测生长点
        for core_mol_list in pop_mols:            
            core_mol_list = [MarkAxialBonds(CleanMol(mol)) for mol in core_mol_list]  # 标记轴向键
            # 创建生长点数据集
            cores_dataset = DatasetGrowthPoint(core_mol_list, EDgrid[:,-1], KDTree(EDgrid[:,:3]))   # 将分子数据 转成 特征图数据！！！
            # 使用 GPPM 模型预测生长点：
            gpcores, nogpcores = GrowthPointFilter(configs, core_mol_list, GPPM, cores_dataset, step, growth_point_pos)
            # 把起源与同一母核的相似的分子放到桶里：
            bucket = MolBucket(bucket_size=bucket_size, sim_threshold=sim_threshold)
            # 获取关键残基
            interact_res = configs.get('sample', 'key_res')

            if nogpcores:  # 没有生长点              
                nogpcores_keep = bucket.add(nogpcores)  # 将没有生长点的核心添加到桶中
                if interact_res:
                    # 根据关键残基进行评分
                    nogpcores_keep = scorer.InteractionScore(nogpcores_keep)
                    # 保留评分为 1 的核心， 1 是有接触
                    nogpcores_keep = [_c for _c,score in nogpcores_keep if score == 1 ]
                # 将没有生长点但有关键接触=1的添加到评分分子列表中
                scored_mols.extend(nogpcores_keep)
            
            if gpcores:  # 有生长点                
                gpcores_keep = bucket.add(gpcores)
                if interact_res:
                    # 根据关键残基进行评分
                    gpcores_score = scorer.InteractionScore(gpcores_keep)
                    for _c, score in gpcores_score:
                        if score:                            
                            grow_mols.append(_c)  # 将有关键作用 =1 的母核添加到生长分子列表中
                        else:                            
                            candidate_mols.append(_c)  # 将评分为 0 的核心添加到候选分子列表中
                else:                    
                    grow_mols.extend(gpcores_keep)  # 将有生长点的核心添加到生长分子列表中
                # 将有生长点的核心添加到评分分子列表中
                scored_mols.extend(gpcores_keep)

        if step:
            # 获取保留的分子数
            keep = int(configs.get('sample', 'retain_mols_num')) if configs.has_option('sample','retain_mols_num') else 100
        else:
            # 获取保留的核心数
            keep = int(configs.get('sample', 'retain_cores_num')) if configs.has_option('sample','retain_cores_num') else 100

        if len(grow_mols) > keep:
            # 随机选择保留的分子
            cores = random.sample(grow_mols, keep)
        else:
            # 如果没有足够的生长分子，则使用【候选分子】补充
            cores = grow_mols
            if len(candidate_mols) > keep - len(grow_mols):
                if interact_res:
                    # 获取关键残基的坐标
                    query_coords = GetResCoords(interact_res)
                    # 根据距离排序候选分子，并选择需要的数量
                    cores.extend(SortMolByDist(candidate_mols, query_coords)[:keep-len(grow_mols)])
                else:
                    # 直接选择需要的数量的候选分子
                    cores.extend(candidate_mols[:keep-len(grow_mols)])
            else:
                # 将所有候选分子添加到核心列表中
                cores.extend(candidate_mols)
        
        if step:
            # 将评分分子添加到生成分子列表中
            generate_molecules.extend(scored_mols)            
            # 创建SD文件写入器
            record = Chem.SDWriter(os.path.join(configs.get('sample', 'output_dir'), f"step{step}_output.sdf"))
            # 遍历选定的分子，写入SD文件
            for idx, _c in enumerate(generate_molecules):
                _c = CleanMol(_c)
                _c = fix_SO2H_func(_c)
                _c.SetProp("generation number", str(idx))
                record.write(_c)
            # 关闭SD文件写入器
            record.close()            
    
    # 获取输出的分子数
    num_molecules = int(configs.get('sample','output_mols_num')) if configs.has_option('sample','output_mols_num') else 1000
    # 随机选择输出的分子，如果生成的分子数少于需要的数量，则输出所有生成的分子
    selected_molecules = random.sample(generate_molecules, num_molecules) if len(generate_molecules) >= num_molecules else generate_molecules
    
    # 创建SD文件写入器
    record = Chem.SDWriter(os.path.join(configs.get('sample', 'output_dir'), f"output.sdf"))
    # 遍历选定的分子，写入SD文件
    for idx, _c in enumerate(selected_molecules):
        _c = CleanMol(_c)
        _c = fix_SO2H_func(_c)
        _c.SetProp("generation number", str(idx))
        if 300 < Descriptors.MolWt(_c) < 500:
            record.write(_c)
    # 关闭SD文件写入器
    record.close()
    print("**********Complete Molecule Generation**********")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    

    parser.add_argument('--config', type=str, default='./configs/zzx_gen_GSK3_JNK3.yml')
    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_RORrt_DHODH.yml')

    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_PARP1_BRD4.yml')
    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_CDK7_PRMT5.yml')
    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_CDK12_PRMT5.yml')
    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_BRD4_FGFR3.yml')

    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_PARP1_PARP1.yml')
    # parser.add_argument('--config', type=str, default='./configs/zzx_gen_BRD4_BRD4.yml')
    args = parser.parse_args()
    configs = GetConfigs(args.config)

    SetSeed(int(configs.get('sample', 'seed')))
    output_dir = configs.get('sample', 'output_dir')
    tmp_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    ligED = LigEDgen(configs, output_dir, tmp_dir)
    MolGrow(ligED, configs)
    shutil.rmtree(tmp_dir)
