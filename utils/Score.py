import os
import shutil
import numpy as np
from rdkit import Chem
from scipy.spatial import KDTree
from plip.structure.preparation import PDBComplex
from concurrent.futures import ProcessPoolExecutor, as_completed


class Scorer:
    def __init__(self, configs):
        try:
            self.protf = configs.get('sample', 'receptor')
        except:
            self.protf = configs.get('sample', 'receptor_A')
        self.query_res = configs.get('sample', 'key_res')
        self.tmpdir = os.path.join(configs.get('sample', 'output_dir'), 'tmp')
        self.num_cpu = int(configs.get('sample', 'num_cpu'))
    
    # 这个 QScore 在文章中有详细的解释
    def QScore(self, mols, density, tol):
        """
        tol: 容忍阈值（浮点），用于判断有多少原子对应到密度为 0 的体素时直接把 q_score 置为 0
        """
        scoredmols = []
        all_mols = []

        voxpoints = density[:, :-1]
        T = KDTree(voxpoints)
        for mol in mols:
            z = []; ha_coors = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ["R", "*", "H"]:
                    z.append(atom.GetAtomicNum())
                    ha_coors.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
            z = np.array(z)
            dis, idx = T.query(ha_coors, k = 1)
            rho = density[:, -1][idx]

            if (rho == 0).sum() / rho.size > tol:
                q_score = len(ha_coors)*np.sum(z*rho)/np.sum(z)
                mol.SetProp('qscore', str(q_score))
                all_mols.append(mol)
            else:
                q_score = len(ha_coors)*np.sum(z*rho)/np.sum(z)
                mol.SetProp('qscore', str(q_score))
                scoredmols.append(mol)
                all_mols.append(mol)
        
        scoredmols = sorted(scoredmols, key=lambda x: float(x.GetProp('qscore')), reverse=True)
        all_mols = sorted(all_mols, key=lambda x: float(x.GetProp('qscore')), reverse=True)
        
        if len(scoredmols) == 0:
            if len(all_mols) > 3:
                scoredmols = all_mols[:3]
            else:
                scoredmols = all_mols
        qscores = [float(i.GetProp('qscore')) for i in scoredmols]
        return scoredmols, qscores
    
    def InteractionMatch(self, compf, query_res, lig_idx):
        """
        detect if the molecule interacts with query residue
        """
        # plip analysis
        mol = PDBComplex()
        mol.load_pdb(compf) 
        bsid = 'UNL:Z:1'   #  这里是 PDBComplex 中 PLIP (Protein–Ligand Interaction Profiler) 的特殊语法。
        mol.analyze()
        interaction = mol.interaction_sets[bsid]  # 获取指定绑定位点的相互作用集合

        # list all side chains or backbones of interacting residues
        interaction_res = []
        # 相互作用残基列表构建
        for i in interaction.all_itypes:  # 遍历 所有相互作用类型的 集合
            i_info = i._asdict()  # 将每个相互作用类型对象 i 转换为字典 i_info
            if i_info['restype'] not in ['LIG', 'HOH']:
                if i_info.get('sidechain', True):  # 属性判断是侧链还是主链相互作用
                    portion = 'sidechain'
                else:
                    portion = 'backbone'
                res = ''.join([str(i.reschain), str(i.resnr)])  # 构建残基标识符 res，由残基链标识和残基序号组成
                interaction_res.append(res + " " + portion)
        interaction_res = list(set(interaction_res))  # 使用 set 去重，然后转换回列表

        if query_res not in interaction_res:
            return lig_idx, False
        return lig_idx, True

    # 用于评估一组分子（配体）与某个蛋白质目标的相互作用得分
    def InteractionScore(self, ligs):
        """
        ligs: 一个包含分子对象的列表

        label each molecule in the population
        return nested array of [ligand_index, match_query]
        [[0, True], [1, False]...]
        """
        # 从文件 self.protf（蛋白质结构文件）中读取所有以 "ATOM " 或 "HETATM" 开头的行，这些行通常包含原子的坐标信息
        reclines = [l for l in open(self.protf).readlines() if l[:6] in ["ATOM  ", "HETATM"]]
        lig_hit = []  # 用于存储配体的命中情况
        plipdir = os.path.join(self.tmpdir, "plip")
        os.mkdir(plipdir)

        # 并行处理配体
        with ProcessPoolExecutor(max_workers=self.num_cpu) as ex:
            futures = []
            for lig_idx, l in enumerate(ligs):  # 遍历 ligs 列表中的每个配体
                lname = str(lig_idx)
                outfname = os.path.join(plipdir, lname + ".pdb")
                l = Chem.RemoveHs(l)
                ligpdb = Chem.PDBWriter(outfname)
                ligpdb.write(l)
                ligpdb.close()
                ligatoms = [l for l in open(outfname).readlines() if l[:6] == "HETATM"]
                ligconect = [l for l in open(outfname).readlines() if l[:6] == "CONECT"]

                # 合并配体和蛋白质结构
                with open(outfname, "w") as comp:
                    for _, l in enumerate(ligatoms):
                        comp.writelines(l[:21] + "Z" + l[22:])
                    for _, l in enumerate(reclines):
                        serial = _ + len(ligatoms) + 1
                        comp.writelines(l[:6] + (5-len(str(serial)))*" " + str(serial) + l[11:])  # 调整原子序号
                    comp.writelines(ligconect)
                    comp.writelines("END")
                
                # 提交并行进程，调用 InteractionMatch，计算相互作用
                futures.append(ex.submit(self.InteractionMatch, outfname, self.query_res, lig_idx))
             
            # 使用 as_completed 迭代已完成的 Future 对象，获取配体索引和命中情况，并添加到 lig_hit 列表中
            for future in as_completed(futures):
                lig_idx, hit = future.result()
                lig_hit.append([lig_idx, hit])

            mol_plipscore = []
            # 遍历 lig_hit 列表，根据命中情况设置 PLIP 得分（1 或 0），并将配体及其得分添加到 mol_plipscore 列表中
            for idx, hit in lig_hit:
                plipscore = 1 if hit else 0
                l = ligs[idx]
                mol_plipscore.append([l, plipscore])

        shutil.rmtree(plipdir)
        return mol_plipscore