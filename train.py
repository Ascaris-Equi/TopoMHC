import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from model import ImmunogenicityPredictor, FeatureProcessor

class PeptideDataset(Dataset):
    """多肽数据集"""
    
    def __init__(self, peptide_ids, topo_features, seq_features, labels, processor=None):
        self.peptide_ids = peptide_ids
        self.topo_features = topo_features
        self.seq_features = seq_features
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.peptide_ids)
    
    def __getitem__(self, idx):
        peptide_id = self.peptide_ids[idx]
        topo_feat = self.topo_features[idx]
        seq_feat = self.seq_features[idx]
        label = self.labels[idx]
        
        # 应用预处理
        if self.processor:
            topo_feat = self.processor.transform_topo_features(topo_feat)
            seq_feat = self.processor.transform_seq_features(seq_feat)
        
        return {
            'peptide_id': peptide_id,
            'topo_features': torch.FloatTensor(topo_feat),
            'seq_features': torch.FloatTensor(seq_feat),
            'label': torch.LongTensor([label])
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    peptide_ids = [item['peptide_id'] for item in batch]
    topo_features = torch.stack([item['topo_features'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    
    # 处理变长序列特征
    seq_features = [item['seq_features'] for item in batch]
    max_len = max(seq.size(0) for seq in seq_features)
    
    # Padding
    padded_seq = []
    for seq in seq_features:
        if seq.size(0) < max_len:
            padding = torch.zeros(max_len - seq.size(0), seq.size(1))
            seq = torch.cat([seq, padding], dim=0)
        padded_seq.append(seq)
    
    seq_features = torch.stack(padded_seq)
    
    return {
        'peptide_ids': peptide_ids,
        'topo_features': topo_features,
        'seq_features': seq_features,
        'labels': labels
    }

def load_and_prepare_data():
    """加载和准备数据"""
    print("Loading data...")
    
    # 加载原始数据
    df = pd.read_csv("data/peptide_dataset.csv")
    
    # 加载拓扑特征
    with open("data/topological_features.pkl", 'rb') as f:
        topo_data = pickle.load(f)
    
    # 加载序列特征
    with open("data/esm_features.pkl", 'rb') as f:
        seq_data = pickle.load(f)
    
    # 对齐数据
    peptide_ids = []
    topo_features = []
    seq_features = []
    labels = []
    
    for _, row in df.iterrows():
        peptide_id = row['peptide_id']
        
        if peptide_id in topo_data and peptide_id in seq_data:
            # 拓扑特征
            topo_feat = extract_topo_feature_vector(topo_data[peptide_id])
            
            # 序列特征
            seq_feat = seq_data[peptide_id]['embeddings_raw']
            
            peptide_ids.append(peptide_id)
            topo_features.append(topo_feat)
            seq_features.append(seq_feat)
            labels.append(row['immunogenic'])
    
    print(f"Successfully loaded {len(peptide_ids)} samples")
    return peptide_ids, topo_features, seq_features, labels

def extract_topo_feature_vector(topo_data: Dict) -> np.ndarray:
    """从拓扑数据中提取特征向量"""
    feature_vector = []
    
    # 按固定顺序提取特征
    feature_keys = [
        'n_conformations',
        # Betti数
        'betti_0_mean', 'betti_1_mean', 'betti_2_mean', 'betti_3_mean',
        'betti_0_std', 'betti_1_std', 'betti_2_std', 'betti_3_std',
        # 生命周期特征
        'lifetime_mean_0_mean', 'lifetime_mean_1_mean', 'lifetime_mean_2_mean', 'lifetime_mean_3_mean',
        'lifetime_std_0_mean', 'lifetime_std_1_mean', 'lifetime_std_2_mean', 'lifetime_std_3_mean',
        'lifetime_max_0_mean', 'lifetime_max_1_mean', 'lifetime_max_2_mean', 'lifetime_max_3_mean',
        # 几何特征
        'radius_gyration_mean', 'asphericity_mean',
        'mean_distance_mean', 'std_distance_mean', 'max_distance_mean',
        # 连通性特征
        'connectivity_3.0_mean', 'connectivity_5.0_mean', 'connectivity_8.0_mean', 'connectivity_10.0_mean'
    ]
    
    for key in feature_keys:
        value = topo_data.get(key, 0.0)
        if isinstance(value, (int, float)):
            feature_vector.append(float(value))
        else:
            feature_vector.append(0.0)
    
    return np.array(feature_vector)

def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu'):
    """训练模型"""
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            topo_feat = batch['topo_features'].to(device)
            seq_feat = batch['seq_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(topo_feat, seq_feat)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                topo_feat = batch['topo_features'].to(device)
                seq_feat = batch['seq_features'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(topo_feat, seq_feat)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true_labels, val_predictions)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}")
    
    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            topo_feat = batch['topo_features'].to(device)
            seq_feat = batch['seq_features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(topo_feat, seq_feat)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # AUC (如果是二分类)
    if len(set(true_labels)) == 2:
        probs_pos = [p[1] for p in probabilities]
        auc = roc_auc_score(true_labels, probs_pos)
    else:
        auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    """主训练流程"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    peptide_ids, topo_features, seq_features, labels = load_and_prepare_data()
    
    # 划分数据集
    train_ids, test_ids, train_topo, test_topo, train_seq, test_seq, train_labels, test_labels = train_test_split(
        peptide_ids, topo_features, seq_features, labels, 
        test_size=0.2, random_state=42, stratify=labels
    )
    
    train_ids, val_ids, train_topo, val_topo, train_seq, val_seq, train_labels, val_labels = train_test_split(
        train_ids, train_topo, train_seq, train_labels,
        test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # 创建特征处理器
    processor = FeatureProcessor()
    processor.fit_topo_features(train_topo)
    processor.fit_seq_features(train_seq)
    
    # 创建数据集
    train_dataset = PeptideDataset(train_ids, train_topo, train_seq, train_labels, processor)
    val_dataset = PeptideDataset(val_ids, val_topo, val_seq, val_labels, processor)
    test_dataset = PeptideDataset(test_ids, test_topo, test_seq, test_labels, processor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 确定特征维度
    topo_dim = len(train_topo[0])
    seq_dim = train_seq[0].shape[1]
    
    print(f"Topology feature dim: {topo_dim}")
    print(f"Sequence feature dim: {seq_dim}")
    
    # 创建模型
    model = ImmunogenicityPredictor(
        topo_feature_dim=topo_dim,
        seq_feature_dim=seq_dim,
        hidden_dim=256,
        num_heads=8,
        num_classes=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    history = train_model(model, train_loader, val_loader, num_epochs=50, device=device)
    
    # 评估模型
    test_results = evaluate_model(model, test_loader, device)
    
    print("\nTest Results:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  F1-score: {test_results['f1']:.4f}")
    if test_results['auc']:
        print(f"  AUC: {test_results['auc']:.4f}")
    
    # 保存模型和预处理器
    Path("models").mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'topo_dim': topo_dim,
        'seq_dim': seq_dim,
        'test_results': test_results
    }, "models/immunogenicity_model.pth")
    
    processor.save("models/feature_processor.pkl")
    
    print("Model and processor saved!")

if __name__ == "__main__":
    main()