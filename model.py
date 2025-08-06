import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class CrossAttentionModule(nn.Module):
    """Cross-attention模块，用于融合拓扑和序列特征"""
    
    def __init__(self, topo_dim: int, seq_dim: int, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 投影层
        self.topo_projection = nn.Linear(topo_dim, hidden_dim)
        self.seq_projection = nn.Linear(seq_dim, hidden_dim)
        
        # Multi-head attention
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, topo_features: torch.Tensor, seq_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            topo_features: [batch_size, topo_dim] 拓扑特征
            seq_features: [batch_size, seq_len, seq_dim] 序列特征
        Returns:
            fused_features: [batch_size, hidden_dim] 融合特征
        """
        batch_size = topo_features.size(0)
        
        # 投影到相同维度
        topo_proj = self.topo_projection(topo_features)  # [batch_size, hidden_dim]
        seq_proj = self.seq_projection(seq_features)     # [batch_size, seq_len, hidden_dim]
        
        # 扩展拓扑特征维度用于attention
        topo_proj = topo_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention: 拓扑特征作为query，序列特征作为key和value
        attended = self.multi_head_attention(topo_proj, seq_proj, seq_proj)
        attended = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Residual connection + layer norm
        attended = self.norm1(attended + topo_proj.squeeze(1))
        
        # Feed forward
        ff_out = self.ff(attended)
        output = self.norm2(attended + ff_out)
        
        return output
    
    def multi_head_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = key.size()
        
        # 投影到多头
        Q = self.query_projection(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        
        # 输出投影
        output = self.out_projection(attended)
        return output

class ImmunogenicityPredictor(nn.Module):
    """免疫原性预测模型"""
    
    def __init__(self, topo_feature_dim: int, seq_feature_dim: int, 
                 hidden_dim: int = 256, num_heads: int = 8, num_classes: int = 2):
        super().__init__()
        
        self.topo_feature_dim = topo_feature_dim
        self.seq_feature_dim = seq_feature_dim
        self.hidden_dim = hidden_dim
        
        # Cross-attention模块
        self.cross_attention = CrossAttentionModule(
            topo_dim=topo_feature_dim,
            seq_dim=seq_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # 额外的特征处理
        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.seq_encoder = nn.Sequential(
            nn.Linear(seq_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, topo_features: torch.Tensor, seq_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            topo_features: [batch_size, topo_feature_dim] 拓扑特征
            seq_features: [batch_size, seq_len, seq_feature_dim] 序列特征
        Returns:
            logits: [batch_size, num_classes] 分类logits
        """
        # 序列特征聚合（取平均）
        seq_mean = torch.mean(seq_features, dim=1)  # [batch_size, seq_feature_dim]
        
        # Cross-attention融合
        fused_features = self.cross_attention(topo_features, seq_features)
        
        # 独立编码
        topo_encoded = self.topo_encoder(topo_features)
        
        # 拼接所有特征
        combined_features = torch.cat([fused_features, topo_encoded], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict_proba(self, topo_features: torch.Tensor, seq_features: torch.Tensor) -> torch.Tensor:
        """返回预测概率"""
        logits = self.forward(topo_features, seq_features)
        return F.softmax(logits, dim=1)

class FeatureProcessor:
    """特征预处理器"""
    
    def __init__(self):
        self.topo_mean = None
        self.topo_std = None
        self.seq_mean = None
        self.seq_std = None
    
    def fit_topo_features(self, features_list):
        """拟合拓扑特征的标准化参数"""
        all_features = np.vstack(features_list)
        self.topo_mean = np.mean(all_features, axis=0)
        self.topo_std = np.std(all_features, axis=0) + 1e-8
    
    def fit_seq_features(self, features_list):
        """拟合序列特征的标准化参数"""
        all_features = np.vstack([f.reshape(-1, f.shape[-1]) for f in features_list])
        self.seq_mean = np.mean(all_features, axis=0)
        self.seq_std = np.std(all_features, axis=0) + 1e-8
    
    def transform_topo_features(self, features):
        """标准化拓扑特征"""
        if self.topo_mean is None:
            return features
        return (features - self.topo_mean) / self.topo_std
    
    def transform_seq_features(self, features):
        """标准化序列特征"""
        if self.seq_mean is None:
            return features
        return (features - self.seq_mean) / self.seq_std
    
    def save(self, filepath):
        """保存预处理器"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'topo_mean': self.topo_mean,
                'topo_std': self.topo_std,
                'seq_mean': self.seq_mean,
                'seq_std': self.seq_std
            }, f)
    
    def load(self, filepath):
        """加载预处理器"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.topo_mean = data['topo_mean']
            self.topo_std = data['topo_std']
            self.seq_mean = data['seq_mean']
            self.seq_std = data['seq_std']

if __name__ == "__main__":
    # 测试模型
    topo_dim = 100
    seq_dim = 320
    batch_size = 4
    seq_len = 10
    
    model = ImmunogenicityPredictor(
        topo_feature_dim=topo_dim,
        seq_feature_dim=seq_dim,
        hidden_dim=256,
        num_heads=8
    )
    
    # 创建测试数据
    topo_features = torch.randn(batch_size, topo_dim)
    seq_features = torch.randn(batch_size, seq_len, seq_dim)
    
    # 前向传播
    logits = model(topo_features, seq_features)
    probas = model.predict_proba(topo_features, seq_features)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Probabilities shape: {probas.shape}")
    print(f"Sample probabilities: {probas[0]}")