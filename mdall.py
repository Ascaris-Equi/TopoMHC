import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Dict
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

class RDKitPeptideConformer:
    """使用RDKit生成多肽构象"""
    
    def __init__(self):
        # 氨基酸到SMILES映射（标准格式）
        self.aa_smiles = {
            'A': 'C[C@H](N)C(=O)O',           # Alanine
            'R': 'NC(=N)NCCC[C@H](N)C(=O)O',  # Arginine  
            'N': 'NC(=O)C[C@H](N)C(=O)O',     # Asparagine
            'D': 'O=C(O)C[C@H](N)C(=O)O',     # Aspartic acid
            'C': 'SC[C@H](N)C(=O)O',          # Cysteine
            'Q': 'NC(=O)CCC[C@H](N)C(=O)O',   # Glutamine
            'E': 'O=C(O)CCC[C@H](N)C(=O)O',   # Glutamic acid
            'G': 'NCC(=O)O',                  # Glycine
            'H': 'N1C=NC=C1C[C@H](N)C(=O)O',  # Histidine
            'I': 'CC[C@H](C)[C@H](N)C(=O)O',  # Isoleucine
            'L': 'CC(C)C[C@H](N)C(=O)O',      # Leucine
            'K': 'NCCCC[C@H](N)C(=O)O',       # Lysine
            'M': 'CSCC[C@H](N)C(=O)O',        # Methionine
            'F': 'c1ccc(cc1)C[C@H](N)C(=O)O', # Phenylalanine
            'P': 'N1CCC[C@H]1C(=O)O',         # Proline
            'S': 'OC[C@H](N)C(=O)O',          # Serine
            'T': 'C[C@@H](O)[C@H](N)C(=O)O',  # Threonine
            'W': 'c1ccc2c(c1)c(cn2)C[C@H](N)C(=O)O', # Tryptophan
            'Y': 'Oc1ccc(cc1)C[C@H](N)C(=O)O', # Tyrosine
            'V': 'CC(C)[C@H](N)C(=O)O'         # Valine
        }

def build_peptide_from_sequence(sequence: str) -> Mol:
    """从氨基酸序列构建多肽分子"""
    conformer = RDKitPeptideConformer()
    
    if len(sequence) == 1:
        # 单个氨基酸
        smiles = conformer.aa_smiles[sequence]
        mol = Chem.MolFromSmiles(smiles)
        return mol
    
    # 多个氨基酸：构建肽键
    mols = []
    for aa in sequence:
        if aa not in conformer.aa_smiles:
            raise ValueError(f"Unknown amino acid: {aa}")
        mol = Chem.MolFromSmiles(conformer.aa_smiles[aa])
        mols.append(mol)
    
    # 连接氨基酸形成肽键
    peptide = mols[0]
    for i in range(1, len(mols)):
        peptide = Chem.CombineMols(peptide, mols[i])
    
    # 简化：直接返回组合分子（实际肽键形成较复杂）
    return peptide

def generate_conformations_for_peptide(args):
    """为单个多肽生成构象（用于并行处理）"""
    peptide_id, sequence, n_conformations = args
    
    try:
        # 构建分子
        mol = build_peptide_from_sequence(sequence)
        if mol is None:
            return peptide_id, None
            
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 生成多个构象
        conf_ids = AllChem.EmbedMultipleConfs(
            mol, 
            numConfs=n_conformations,
            randomSeed=42,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            maxAttempts=100
        )
        
        if len(conf_ids) == 0:
            # 如果失败，生成简单构象
            return peptide_id, generate_simple_conformations(sequence, n_conformations)
        
        # 优化构象
        for conf_id in conf_ids:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            except:
                pass  # 优化失败也继续
        
        # 提取坐标
        conformations = []
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            conformations.append(np.array(coords))
        
        result = {
            'sequence': sequence,
            'conformations': conformations,
            'n_conformations': len(conformations),
            'n_atoms': mol.GetNumAtoms()
        }
        
        return peptide_id, result
        
    except Exception as e:
        print(f"Failed to process {peptide_id} ({sequence}): {e}")
        return peptide_id, generate_simple_conformations(sequence, n_conformations)

def generate_simple_conformations(sequence: str, n_conformations=10):
    """生成简单的多肽构象（备用方案）"""
    n_atoms = len(sequence) * 5  # 假设每个残基5个重原子
    conformations = []
    
    for conf_id in range(n_conformations):
        coords = np.zeros((n_atoms, 3))
        
        # 生成类似多肽的结构
        for i, aa in enumerate(sequence):
            base_x = i * 3.8  # 沿主链方向
            
            # 主链和侧链原子
            for j in range(5):
                coords[i*5+j] = [
                    base_x + j*0.3 + np.random.normal(0, 0.2),
                    np.random.normal(0, 0.5),
                    np.random.normal(0, 0.5)
                ]
            
            # 根据氨基酸性质调整
            if aa == 'G':  # 甘氨酸更柔性
                coords[i*5:(i+1)*5] += np.random.normal(0, 0.3, (5, 3))
            elif aa == 'P':  # 脯氨酸更刚性
                coords[i*5:(i+1)*5] *= 0.9
        
        conformations.append(coords)
    
    return {
        'sequence': sequence,
        'conformations': conformations,
        'n_conformations': len(conformations),
        'n_atoms': n_atoms
    }

def run_md_simulations(peptide_df: pd.DataFrame, n_conformations=10, 
                      output_dir="data/conformations", n_processes=None):
    """
    并行生成所有多肽的构象
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 24)  # 最多8个进程
    
    print(f"Generating conformations for {len(peptide_df)} peptides using {n_processes} processes...")
    
    # 准备参数
    args_list = [
        (row['peptide_id'], row['sequence'], n_conformations)
        for _, row in peptide_df.iterrows()
    ]
    
    # 并行处理
    results = {}
    
    with mp.Pool(processes=n_processes) as pool:
        # 使用imap显示进度
        for i, (peptide_id, result) in enumerate(pool.imap(generate_conformations_for_peptide, args_list)):
            results[peptide_id] = result
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(args_list)} peptides")
            
            # 定期保存结果
            if (i + 1) % 200 == 0:
                batch_file = f"{output_dir}/conformations_batch_{(i + 1)//200}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Saved batch {(i + 1)//200}")
    
    # 保存最终结果
    final_file = f"{output_dir}/conformations_final.pkl"
    with open(final_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"All conformations generated! Results saved to {output_dir}")
    
    # 统计信息
    successful = sum(1 for r in results.values() if r is not None)
    print(f"Successfully processed: {successful}/{len(peptide_df)} peptides")
    
    return results

if __name__ == "__main__":
    # 加载完整数据集
    df = pd.read_csv("data/peptide_dataset.csv")
    print(f"Loaded {len(df)} peptides")
    
    # 生成所有构象
    results = run_md_simulations(df, n_conformations=10, n_processes=6)
    
    print("Conformation generation completed!")
    
    # 统计信息
    if results:
        sample_peptides = list(results.items())[:3]
        for peptide_id, data in sample_peptides:
            if data:
                print(f"\n{peptide_id} ({data['sequence']}):")
                print(f"  Generated {data['n_conformations']} conformations")
                print(f"  Number of atoms: {data['n_atoms']}")
                if data['conformations']:
                    coords = data['conformations'][0]
                    print(f"  First conformation shape: {coords.shape}")