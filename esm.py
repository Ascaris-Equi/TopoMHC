import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: Transformers not installed. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

class ESMFeatureExtractor:
    """ESM序列特征提取器"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device=None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading ESM model: {model_name}")
        print(f"Using device: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("ESM model loaded successfully")
    
    def extract_sequence_features(self, sequence: str) -> Dict[str, np.ndarray]:
        """提取单个序列的特征"""
        try:
            # Tokenize序列
            inputs = self.tokenizer(
                sequence, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 提取特征
            # last_hidden_state: [batch_size, seq_len, hidden_size]
            hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_size]
            
            # 移除特殊token ([CLS], [SEP])
            if len(hidden_states) > 2:
                hidden_states = hidden_states[1:-1]  # 移除第一个和最后一个token
            
            # 转换为numpy
            hidden_states = hidden_states.cpu().numpy()
            
            # 计算各种聚合特征
            features = {
                'embeddings_mean': np.mean(hidden_states, axis=0),
                'embeddings_max': np.max(hidden_states, axis=0),
                'embeddings_min': np.min(hidden_states, axis=0),
                'embeddings_std': np.std(hidden_states, axis=0),
                'embeddings_first': hidden_states[0] if len(hidden_states) > 0 else np.zeros(hidden_states.shape[1]),
                'embeddings_last': hidden_states[-1] if len(hidden_states) > 0 else np.zeros(hidden_states.shape[1]),
                'embeddings_raw': hidden_states  # 保存原始embeddings用于attention
            }
            
            return features
            
        except Exception as e:
            print(f"Failed to extract features for sequence {sequence}: {e}")
            # 返回零特征
            hidden_size = 320  # ESM2-t6的hidden size
            return {
                'embeddings_mean': np.zeros(hidden_size),
                'embeddings_max': np.zeros(hidden_size),
                'embeddings_min': np.zeros(hidden_size),
                'embeddings_std': np.zeros(hidden_size),
                'embeddings_first': np.zeros(hidden_size),
                'embeddings_last': np.zeros(hidden_size),
                'embeddings_raw': np.zeros((len(sequence), hidden_size))
            }
    
    def extract_batch_features(self, sequences: List[str], batch_size=32) -> Dict[str, Dict]:
        """批量提取序列特征"""
        results = {}
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            for j, sequence in enumerate(batch_sequences):
                peptide_id = f"peptide_{i+j:04d}"
                features = self.extract_sequence_features(sequence)
                results[peptide_id] = features
                
                if (i + j + 1) % 100 == 0:
                    print(f"Processed {i + j + 1} sequences")
        
        return results

def compute_esm_features(peptide_file: str, output_file: str, model_name="facebook/esm2_t6_8M_UR50D"):
    """计算所有多肽的ESM特征"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available, generating dummy features")
        return generate_dummy_esm_features(peptide_file, output_file)
    
    # 加载多肽数据
    df = pd.read_csv(peptide_file)
    print(f"Computing ESM features for {len(df)} peptides...")
    
    # 初始化特征提取器
    extractor = ESMFeatureExtractor(model_name=model_name)
    
    # 提取特征
    results = {}
    for idx, row in df.iterrows():
        peptide_id = row['peptide_id']
        sequence = row['sequence']
        
        features = extractor.extract_sequence_features(sequence)
        features['sequence'] = sequence
        results[peptide_id] = features
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} peptides")
    
    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"ESM features computed and saved to {output_file}")
    return results

def generate_dummy_esm_features(peptide_file: str, output_file: str):
    """生成假的ESM特征（用于测试）"""
    df = pd.read_csv(peptide_file)
    results = {}
    
    hidden_size = 320  # ESM2-t6的hidden size
    
    for idx, row in df.iterrows():
        peptide_id = row['peptide_id']
        sequence = row['sequence']
        seq_len = len(sequence)
        
        # 生成基于序列的假特征
        np.random.seed(hash(sequence) % 2**32)  # 确保可重现
        
        # 生成原始embeddings
        embeddings_raw = np.random.randn(seq_len, hidden_size) * 0.1
        
        # 添加一些基于序列的结构
        for i, aa in enumerate(sequence):
            aa_code = ord(aa) - ord('A')
            embeddings_raw[i] += aa_code * 0.01
        
        features = {
            'embeddings_mean': np.mean(embeddings_raw, axis=0),
            'embeddings_max': np.max(embeddings_raw, axis=0),
            'embeddings_min': np.min(embeddings_raw, axis=0),
            'embeddings_std': np.std(embeddings_raw, axis=0),
            'embeddings_first': embeddings_raw[0],
            'embeddings_last': embeddings_raw[-1],
            'embeddings_raw': embeddings_raw,
            'sequence': sequence
        }
        
        results[peptide_id] = features
    
    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Dummy ESM features generated and saved to {output_file}")
    return results

if __name__ == "__main__":
    # 计算ESM特征
    peptide_file = "data/peptide_dataset.csv"
    output_file = "data/esm_features.pkl"
    
    features = compute_esm_features(peptide_file, output_file)
    
    # 显示一些示例特征
    if features:
        sample_id = list(features.keys())[0]
        sample_features = features[sample_id]
        print(f"\nSample ESM features for {sample_id}:")
        print(f"  Sequence: {sample_features['sequence']}")
        print(f"  Embeddings mean shape: {sample_features['embeddings_mean'].shape}")
        print(f"  Embeddings raw shape: {sample_features['embeddings_raw'].shape}")