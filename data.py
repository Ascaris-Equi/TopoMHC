import random
import pandas as pd
import numpy as np
from pathlib import Path

def generate_peptide_dataset(n_peptides=1000, min_length=8, max_length=14, seed=42):
    """
    生成假的多肽数据集
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 20种标准氨基酸单字母代码
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    peptides = []
    labels = []
    
    for i in range(n_peptides):
        # 随机选择多肽长度
        length = random.randint(min_length, max_length)
        
        # 生成随机多肽序列
        sequence = ''.join(random.choices(amino_acids, k=length))
        
        # 生成假的免疫原性标签（这里用一些简单的规则模拟真实情况）
        # 实际应用中这些标签来自实验数据
        immunogenic = generate_fake_label(sequence)
        
        peptides.append(sequence)
        labels.append(immunogenic)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'peptide_id': [f'peptide_{i:04d}' for i in range(n_peptides)],
        'sequence': peptides,
        'immunogenic': labels
    })
    
    return df

def generate_fake_label(sequence):
    """
    基于序列特征生成假的免疫原性标签
    这里使用一些简化的规则，实际应用中应该是实验数据
    """
    # 简单规则：包含某些氨基酸组合的更可能免疫原性
    immunogenic_motifs = ['KR', 'RK', 'KK', 'RR', 'FY', 'YF', 'WF', 'FW']
    hydrophobic_aa = set(['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'])
    
    score = 0
    
    # 检查免疫原性基序
    for motif in immunogenic_motifs:
        score += sequence.count(motif) * 0.3
    
    # 疏水性氨基酸比例
    hydrophobic_ratio = sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence)
    score += hydrophobic_ratio * 0.2
    
    # 长度因子
    if 9 <= len(sequence) <= 11:  # MHC-I结合的最佳长度
        score += 0.2
    
    # 添加随机噪声
    score += random.uniform(-0.3, 0.3)
    
    return 1 if score > 0.3 else 0

if __name__ == "__main__":
    # 生成数据集
    df = generate_peptide_dataset()
    
    # 保存数据
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/peptide_dataset.csv", index=False)
    
    # 打印统计信息
    print(f"Generated {len(df)} peptides")
    print(f"Sequence length range: {df['sequence'].str.len().min()}-{df['sequence'].str.len().max()}")
    print(f"Immunogenic ratio: {df['immunogenic'].mean():.3f}")
    print(f"Sample sequences:")
    print(df.head())