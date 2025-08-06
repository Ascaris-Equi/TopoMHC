import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    print("Warning: GUDHI not installed. Install with: conda install -c conda-forge gudhi")
    GUDHI_AVAILABLE = False

from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing as mp
from functools import partial

class TopologicalFeatureExtractor:
    """拓扑特征提取器"""
    
    def __init__(self, max_dimension=3, max_edge_length=15.0):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
    
    def compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        return euclidean_distances(coords)
    
    def compute_persistent_homology(self, coords: np.ndarray) -> Dict:
        """计算持续同调"""
        if not GUDHI_AVAILABLE:
            return self.compute_simple_topo_features(coords)
        
        try:
            # 构建Rips复合体
            rips_complex = gd.RipsComplex(
                points=coords, 
                max_edge_length=self.max_edge_length
            )
            
            # 创建单纯复合体
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
            
            # 计算持续同调
            persistence = simplex_tree.persistence()
            
            # 提取特征
            features = self.extract_persistence_features(persistence)
            return features
            
        except Exception as e:
            print(f"GUDHI computation failed: {e}")
            return self.compute_simple_topo_features(coords)
    
    def extract_persistence_features(self, persistence: List) -> Dict:
        """从持续同调结果提取特征"""
        features = {}
        
        # 按维度分组
        by_dimension = {}
        for dim, (birth, death) in persistence:
            if dim not in by_dimension:
                by_dimension[dim] = []
            
            # 处理无穷大死亡时间
            if death == float('inf'):
                death = self.max_edge_length
            
            by_dimension[dim].append((birth, death))
        
        # 为每个维度计算特征
        for dim in range(self.max_dimension + 1):
            intervals = by_dimension.get(dim, [])
            
            # Betti数
            features[f'betti_{dim}'] = len(intervals)
            
            if intervals:
                # 生命周期统计
                lifetimes = [death - birth for birth, death in intervals]
                features[f'lifetime_mean_{dim}'] = np.mean(lifetimes)
                features[f'lifetime_std_{dim}'] = np.std(lifetimes)
                features[f'lifetime_max_{dim}'] = np.max(lifetimes)
                features[f'lifetime_sum_{dim}'] = np.sum(lifetimes)
                
                # 出生时间统计
                births = [birth for birth, death in intervals]
                features[f'birth_mean_{dim}'] = np.mean(births)
                features[f'birth_std_{dim}'] = np.std(births)
                
                # 死亡时间统计
                deaths = [death for birth, death in intervals]
                features[f'death_mean_{dim}'] = np.mean(deaths)
                features[f'death_std_{dim}'] = np.std(deaths)
                
                # 持续性景观特征（简化版）
                features[f'landscape_integral_{dim}'] = np.sum([
                    (death - birth) * birth for birth, death in intervals
                ])
            else:
                # 没有该维度的特征时填充0
                for stat in ['lifetime_mean', 'lifetime_std', 'lifetime_max', 'lifetime_sum',
                           'birth_mean', 'birth_std', 'death_mean', 'death_std', 'landscape_integral']:
                    features[f'{stat}_{dim}'] = 0.0
        
        return features
    
    def compute_simple_topo_features(self, coords: np.ndarray) -> Dict:
        """计算简化的拓扑特征（GUDHI不可用时）"""
        features = {}
        
        # 距离矩阵
        dist_matrix = self.compute_distance_matrix(coords)
        n_atoms = len(coords)
        
        # 基于距离的简单特征
        features['n_atoms'] = n_atoms
        features['mean_distance'] = np.mean(dist_matrix[np.triu_indices(n_atoms, k=1)])
        features['std_distance'] = np.std(dist_matrix[np.triu_indices(n_atoms, k=1)])
        features['max_distance'] = np.max(dist_matrix)
        features['min_distance'] = np.min(dist_matrix[dist_matrix > 0])
        
        # 近邻特征
        for threshold in [3.0, 5.0, 8.0, 10.0]:
            close_pairs = np.sum(dist_matrix < threshold) - n_atoms  # 减去对角线
            features[f'close_pairs_{threshold}'] = close_pairs
            features[f'connectivity_{threshold}'] = close_pairs / (n_atoms * (n_atoms - 1))
        
        # 几何特征
        features['radius_gyration'] = self.compute_radius_of_gyration(coords)
        features['asphericity'] = self.compute_asphericity(coords)
        
        # 填充Betti数（简化计算）
        for dim in range(self.max_dimension + 1):
            features[f'betti_{dim}'] = 0
            for stat in ['lifetime_mean', 'lifetime_std', 'lifetime_max', 'lifetime_sum',
                       'birth_mean', 'birth_std', 'death_mean', 'death_std', 'landscape_integral']:
                features[f'{stat}_{dim}'] = 0.0
        
        return features
    
    def compute_radius_of_gyration(self, coords: np.ndarray) -> float:
        """计算回转半径"""
        centroid = np.mean(coords, axis=0)
        distances_sq = np.sum((coords - centroid) ** 2, axis=1)
        return np.sqrt(np.mean(distances_sq))
    
    def compute_asphericity(self, coords: np.ndarray) -> float:
        """计算非球性"""
        centered = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # 降序排列
        
        if len(eigenvals) >= 3:
            return eigenvals[0] - (eigenvals[1] + eigenvals[2]) / 2
        else:
            return 0.0

def compute_features_for_peptide(args):
    """为单个多肽计算拓扑特征（用于并行处理）"""
    peptide_id, peptide_data = args
    
    try:
        extractor = TopologicalFeatureExtractor()
        conformations = peptide_data['conformations']
        
        # 为每个构象计算特征
        all_features = []
        for conf_idx, coords in enumerate(conformations):
            if isinstance(coords, np.ndarray) and coords.shape[0] > 0:
                features = extractor.compute_persistent_homology(coords)
                features['conformation_id'] = conf_idx
                all_features.append(features)
        
        if not all_features:
            return peptide_id, None
        
        # 聚合所有构象的特征
        aggregated_features = aggregate_conformation_features(all_features)
        aggregated_features['sequence'] = peptide_data['sequence']
        aggregated_features['n_conformations'] = len(all_features)
        
        return peptide_id, aggregated_features
        
    except Exception as e:
        print(f"Failed to compute features for {peptide_id}: {e}")
        return peptide_id, None

def aggregate_conformation_features(feature_list: List[Dict]) -> Dict:
    """聚合多个构象的特征"""
    if not feature_list:
        return {}
    
    aggregated = {}
    feature_keys = [k for k in feature_list[0].keys() if k != 'conformation_id']
    
    for key in feature_keys:
        values = [f[key] for f in feature_list if key in f]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
    
    return aggregated

def compute_topological_features(conformations_file: str, output_file: str, n_processes=None):
    """计算所有多肽的拓扑特征"""
    
    # 加载构象数据
    with open(conformations_file, 'rb') as f:
        conformations_data = pickle.load(f)
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 48)
    
    print(f"Computing topological features for {len(conformations_data)} peptides using {n_processes} processes...")
    
    # 准备参数
    args_list = list(conformations_data.items())
    
    # 并行计算
    results = {}
    
    with mp.Pool(processes=n_processes) as pool:
        for i, (peptide_id, features) in enumerate(pool.imap(compute_features_for_peptide, args_list)):
            if features is not None:
                results[peptide_id] = features
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(args_list)} peptides")
    
    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Topological features computed and saved to {output_file}")
    print(f"Successfully processed: {len(results)}/{len(conformations_data)} peptides")
    
    return results

if __name__ == "__main__":
    # 计算拓扑特征
    conformations_file = "data/conformations/conformations_final.pkl"
    output_file = "data/topological_features.pkl"
    
    if Path(conformations_file).exists():
        features = compute_topological_features(conformations_file, output_file)
        
        # 显示一些示例特征
        if features:
            sample_id = list(features.keys())[0]
            sample_features = features[sample_id]
            print(f"\nSample features for {sample_id}:")
            for key, value in list(sample_features.items())[:10]:
                print(f"  {key}: {value}")
    else:
        print(f"Conformations file not found: {conformations_file}")