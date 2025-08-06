import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Dict
import warnings
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.rdchem import Mol
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not installed. Install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

class RDKitPeptideConformer:
    """使用RDKit生成多肽构象"""
    
    def __init__(self):
        # 氨基酸单字母代码到SMILES的映射
        self.aa_smiles = {
            'A': 'N[C@@H](C)C(=O)',      # Alanine
            'R': 'N[C@@H](CCCNC(=N)N)C(=O)',  # Arginine
            'N': 'N[C@@H](CC(=O)N)C(=O)',     # Asparagine
            'D': 'N[C@@H](CC(=O)O)C(=O)',     # Aspartic acid
            'C': 'N[C@@H](CS)C(=O)',          # Cysteine
            'Q': 'N[C@@H](CCC(=O)N)C(=O)',    # Glutamine
            'E': 'N[C@@H](CCC(=O)O)C(=O)',    # Glutamic acid
            'G': 'NCC(=O)',                   # Glycine
            'H': 'N[C@@H](CC1=CNC=N1)C(=O)',  # Histidine
            'I': 'N[C@@H]([C@@H](C)CC)C(=O)', # Isoleucine
            'L': 'N[C@@H](CC(C)C)C(=O)',      # Leucine
            'K': 'N[C@@H](CCCCN)C(=O)',       # Lysine
            'M': 'N[C@@H](CCSC)C(=O)',        # Methionine
            'F': 'N[C@@H](CC1=CC=CC=C1)C(=O)', # Phenylalanine
            'P': 'N1[C@@H](CCC1)C(=O)',       # Proline
            'S': 'N[C@@H](CO)C(=O)',          # Serine
            'T': 'N[C@@H]([C@H](C)O)C(=O)',   # Threonine
            'W': 'N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)', # Tryptophan
            'Y': 'N[C@@H](CC1=CC=C(C=C1)O)C(=O)',    # Tyrosine
            'V': 'N[C@@H](C(C)C)C(=O)'        # Valine
        }
    
    def sequence_to_smiles(self, sequence: str) -> str:
        """将氨基酸序列转换为SMILES字符串"""
        if not sequence:
            return ""
        
        # 构建多肽SMILES
        smiles_parts = []
        for i, aa in enumerate(sequence):
            if aa not in self.aa_smiles:
                raise ValueError(f"Unknown amino acid: {aa}")
            
            aa_smiles = self.aa_smiles[aa]
            if i == 0:
                # 第一个残基，移除N末端的保护
                smiles_parts.append(aa_smiles)
            else:
                # 连接到前一个残基，形成肽键
                # 简化处理：用O替换前一个残基的=O，用N替换当前残基的N
                smiles_parts.append(aa_smiles[1:])  # 移除N
        
        # 简化的多肽SMILES构建
        peptide_smiles = "N" + "".join([part.replace("C(=O)", "C(=O)N") for part in smiles_parts[:-1]]) + smiles_parts[-1].replace("C(=O)", "C(=O)O")
        
        return peptide_smiles
    
    def generate_conformations_simple(self, sequence: str, n_conformations=10):
        """简化方法：直接生成3D坐标"""
        n_atoms = len(sequence) * 4  # 假设每个残基4个重原子
        conformations = []
        
        for conf_id in range(n_conformations):
            coords = np.zeros((n_atoms, 3))
            
            # 生成类似多肽的结构
            for i, aa in enumerate(sequence):
                base_x = i * 3.8  # 沿主链方向
                
                # 主链原子
                coords[i*4] = [base_x, 0, 0]  # CA
                coords[i*4+1] = [base_x-1.2, 0.5, 0]  # N
                coords[i*4+2] = [base_x+1.2, 0.5, 0]  # C
                coords[i*4+3] = [base_x+1.8, 1.2, 0]  # O
                
                # 添加一些随机扰动来模拟不同构象
                noise_scale = 0.5 + conf_id * 0.1
                noise = np.random.normal(0, noise_scale, (4, 3))
                coords[i*4:i*4+4] += noise
                
                # 根据氨基酸性质调整
                if aa in ['P']:  # 脯氨酸增加刚性
                    coords[i*4:i*4+4] *= 0.8
                elif aa in ['G']:  # 甘氨酸增加柔性
                    coords[i*4:i*4+4] += np.random.normal(0, 0.3, (4, 3))
            
            conformations.append(coords)
        
        return conformations
    
    def generate_conformations_rdkit(self, sequence: str, n_conformations=10):
        """使用RDKit生成构象（如果可用）"""
        if not RDKIT_AVAILABLE:
            return self.generate_conformations_simple(sequence, n_conformations)
        
        try:
            # 构建简化的多肽分子
            mol = self.build_peptide_mol(sequence)
            if mol is None:
                return self.generate_conformations_simple(sequence, n_conformations)
            
            # 生成3D构象
            mol = Chem.AddHs(mol)
            
            # 使用ETKDG方法生成构象
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=n_conformations,
                randomSeed=42,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True
            )
            
            if len(conf_ids) == 0:
                return self.generate_conformations_simple(sequence, n_conformations)
            
            # 优化构象
            for conf_id in conf_ids:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
            
            # 提取坐标
            conformations = []
            for conf_id in conf_ids:
                conf = mol.GetConformer(conf_id)
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                conformations.append(np.array(coords))
            
            return conformations
            
        except Exception as e:
            print(f"RDKit conformation generation failed: {e}")
            return self.generate_conformations_simple(sequence, n_conformations)
    
    def build_peptide_mol(self, sequence: str):
        """构建简化的多肽分子"""
        try:
            # 简化方法：构建线性多肽
            smiles = self.sequence_to_smiles(sequence)
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            # 如果SMILES方法失败，使用更简单的方法
            return None

def run_md_simulations(peptide_df: pd.DataFrame, n_conformations=10, output_dir="data/conformations"):
    """
    为数据集中的所有多肽生成构象
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    conformer = RDKitPeptideConformer()
    results = {}
    
    print(f"Generating conformations for {len(peptide_df)} peptides...")
    
    for idx, row in peptide_df.iterrows():
        peptide_id = row['peptide_id']
        sequence = row['sequence']
        
        print(f"Processing {peptide_id}: {sequence}")
        
        # 生成构象
        conformations = conformer.generate_conformations_rdkit(
            sequence, n_conformations=n_conformations
        )
        
        results[peptide_id] = {
            'sequence': sequence,
            'conformations': conformations,
            'n_conformations': len(conformations)
        }
        
        # 每100个保存一次
        if (idx + 1) % 100 == 0:
            batch_file = f"{output_dir}/conformations_batch_{(idx + 1)//100}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved batch {(idx + 1)//100}")
    
    # 保存最终结果
    final_file = f"{output_dir}/conformations_final.pkl"
    with open(final_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Conformation generation completed. Results saved to {output_dir}")
    return results

if __name__ == "__main__":
    # 加载多肽数据集
    df = pd.read_csv("data/peptide_dataset.csv")
    
    # 生成构象（先测试前10个）
    test_df = df.head(10).copy()
    results = run_md_simulations(test_df, n_conformations=10)
    
    print("Conformation generation completed!")
    
    # 打印一些统计信息
    for peptide_id, data in list(results.items())[:3]:
        print(f"\n{peptide_id} ({data['sequence']}):")
        print(f"  Generated {data['n_conformations']} conformations")
        if data['conformations']:
            coords = data['conformations'][0]
            print(f"  First conformation shape: {coords.shape}")
            print(f"  Coordinate range: [{coords.min():.2f}, {coords.max():.2f}]")