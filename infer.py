import torch
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from model import ImmunogenicityPredictor, FeatureProcessor
from esm import ESMFeatureExtractor
from topo import TopologicalFeatureExtractor
from md import RDKitPeptideConformer

class ImmunogenicityPredictor_Pipeline:
    """免疫原性预测管道"""
    
    def __init__(self, model_path="models/immunogenicity_model.pth", 
                 processor_path="models/feature_processor.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = ImmunogenicityPredictor(
            topo_feature_dim=checkpoint['topo_dim'],
            seq_feature_dim=checkpoint['seq_dim'],
            hidden_dim=256,
            num_heads=8,
            num_classes=2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 加载特征处理器
        print("Loading feature processor...")
        self.processor = FeatureProcessor()
        self.processor.load(processor_path)
        
        # 初始化特征提取器
        print("Initializing feature extractors...")
        try:
            self.esm_extractor = ESMFeatureExtractor()
            self.esm_available = True
        except:
            print("ESM not available, using dummy features")
            self.esm_available = False
        
        self.conformer = RDKitPeptideConformer()
        self.topo_extractor = TopologicalFeatureExtractor()
        
        print("Pipeline initialized successfully!")
    
    def extract_topo_features(self, sequence: str) -> np.ndarray:
        """提取拓扑特征"""
        try:
            # 生成构象
            conformations = self.conformer.generate_conformations_rdkit(sequence, n_conformations=5)
            
            # 计算拓扑特征
            all_features = []
            for coords in conformations:
                if isinstance(coords, np.ndarray) and coords.shape[0] > 0:
                    features = self.topo_extractor.compute_persistent_homology(coords)
                    all_features.append(features)
            
            if not all_features:
                return self.get_dummy_topo_features()
            
            # 聚合特征
            aggregated = self.aggregate_topo_features(all_features)
            return self.topo_features_to_vector(aggregated)
            
        except Exception as e:
            print(f"Topology feature extraction failed: {e}")
            return self.get_dummy_topo_features()
    
    def aggregate_topo_features(self, feature_list):
        """聚合拓扑特征"""
        aggregated = {}
        
        if not feature_list:
            return aggregated
        
        feature_keys = [k for k in feature_list[0].keys()]
        
        for key in feature_keys:
            values = [f.get(key, 0) for f in feature_list]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        aggregated['n_conformations'] = len(feature_list)
        return aggregated
    
    def topo_features_to_vector(self, topo_data):
        """将拓扑特征转换为向量"""
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
        
        feature_vector = []
        for key in feature_keys:
            value = topo_data.get(key, 0.0)
            feature_vector.append(float(value))
        
        return np.array(feature_vector)
    
    def get_dummy_topo_features(self):
        """获取虚拟拓扑特征"""
        return np.zeros(28)  # 与训练时的特征维度一致
    
    def extract_seq_features(self, sequence: str) -> np.ndarray:
        """提取序列特征"""
        if self.esm_available:
            try:
                features = self.esm_extractor.extract_sequence_features(sequence)
                return features['embeddings_raw']
            except Exception as e:
                print(f"ESM feature extraction failed: {e}")
        
        # 生成虚拟序列特征
        hidden_size = 320
        seq_len = len(sequence)
        
        np.random.seed(hash(sequence) % 2**32)  # 确保可重现
        embeddings = np.random.randn(seq_len, hidden_size) * 0.1
        
        # 添加一些基于序列的结构
        for i, aa in enumerate(sequence):
            aa_code = ord(aa) - ord('A')
            embeddings[i] += aa_code * 0.01
        
        return embeddings
    
    def predict(self, sequence: str) -> dict:
        """预测单个序列的免疫原性"""
        print(f"Predicting immunogenicity for: {sequence}")
        
        # 提取特征
        print("Extracting topology features...")
        topo_features = self.extract_topo_features(sequence)
        
        print("Extracting sequence features...")
        seq_features = self.extract_seq_features(sequence)
        
        # 应用预处理
        topo_features = self.processor.transform_topo_features(topo_features)
        seq_features = self.processor.transform_seq_features(seq_features)
        
        # 转换为tensor
        topo_tensor = torch.FloatTensor(topo_features).unsqueeze(0).to(self.device)
        seq_tensor = torch.FloatTensor(seq_features).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(topo_tensor, seq_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        # 解析结果
        prob_non_immunogenic = float(probabilities[0][0])
        prob_immunogenic = float(probabilities[0][1])
        predicted_class = int(prediction[0])
        
        result = {
            'sequence': sequence,
            'predicted_class': predicted_class,
            'predicted_label': 'Immunogenic' if predicted_class == 1 else 'Non-immunogenic',
            'probability_non_immunogenic': prob_non_immunogenic,
            'probability_immunogenic': prob_immunogenic,
            'confidence': max(prob_non_immunogenic, prob_immunogenic)
        }
        
        return result
    
    def predict_batch(self, sequences: list) -> list:
        """批量预测"""
        results = []
        for seq in sequences:
            result = self.predict(seq)
            results.append(result)
        return results

def main():
    """主函数 - 命令行使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict peptide immunogenicity")
    parser.add_argument("--sequence", type=str, help="Peptide sequence to predict")
    parser.add_argument("--file", type=str, help="File containing peptide sequences (one per line)")
    parser.add_argument("--model", type=str, default="models/immunogenicity_model.pth", 
                       help="Path to model file")
    parser.add_argument("--processor", type=str, default="models/feature_processor.pkl",
                       help="Path to feature processor file")
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"Model file not found: {args.model}")
        print("Please run train.py first to train the model.")
        return
    
    # 初始化预测器
    predictor = ImmunogenicityPredictor_Pipeline(args.model, args.processor)
    
    # 单个序列预测
    if args.sequence:
        result = predictor.predict(args.sequence)
        print("\nPrediction Result:")
        print(f"Sequence: {result['sequence']}")
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        print(f"  Non-immunogenic: {result['probability_non_immunogenic']:.4f}")
        print(f"  Immunogenic: {result['probability_immunogenic']:.4f}")
    
    # 批量预测
    elif args.file:
        if not Path(args.file).exists():
            print(f"File not found: {args.file}")
            return
        
        with open(args.file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        print(f"Predicting {len(sequences)} sequences...")
        results = predictor.predict_batch(sequences)
        
        print("\nBatch Prediction Results:")
        for result in results:
            print(f"{result['sequence']}: {result['predicted_label']} "
                  f"(confidence: {result['confidence']:.4f})")
    
    else:
        # 交互模式
        print("Interactive mode - Enter peptide sequences (or 'quit' to exit):")
        while True:
            sequence = input("Sequence: ").strip()
            if sequence.lower() in ['quit', 'exit', 'q']:
                break
            
            if sequence:
                try:
                    result = predictor.predict(sequence)
                    print(f"Prediction: {result['predicted_label']} "
                          f"(confidence: {result['confidence']:.4f})")
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    main()