Peptide Immunogenicity Prediction using Topological and Sequential Features

Python • PyTorch • RDKit • License

A deep learning framework for predicting peptide immunogenicity by integrating topological features from molecular dynamics simulations and sequential features from protein language models.
🚀 Features

    Multi-modal Feature Integration: Combines topological data analysis (TDA) with protein language model embeddings
    Cross-Attention Architecture: Novel attention mechanism to fuse structural and sequential information
    Comprehensive Pipeline: End-to-end workflow from sequence input to immunogenicity prediction
    Parallel Processing: Efficient batch processing for large-scale datasets
    Easy Inference: Simple command-line interface for single sequence or batch predictions

🏗️ Architecture Overview
mermaid

flowchart TD
    A[Input Peptide Sequence] --> B1[Conformation Generation (RDKit)]
    A --> B2[ESM-2 Protein Language Model Embeddings]
    A --> B3[Topological Data Analysis (Persistent Homology)]
    B1 --> C1[Structural Features]
    B2 --> C2[Sequential Features]
    B3 --> C3[Topological Features]
    C1 --> D[Cross-Attention Fusion]
    C2 --> D
    C3 --> D
    D --> E[Classification Head]
    E --> F[Immunogenicity Score]

📋 Requirements
Core Dependencies

    Python >= 3.8
    PyTorch >= 1.9
    RDKit >= 2022.03
    NumPy >= 1.21
    Pandas >= 1.3
    Scikit-learn >= 1.0

Optional (for enhanced features)

    transformers >= 4.21 <sub>For ESM-2 embeddings</sub>
    gudhi >= 3.4 <sub>For topological data analysis</sub>
    matplotlib >= 3.5 <sub>For visualization</sub>

🛠️ Installation
Option 1: Conda (Recommended)
bash

# Clone repository
git clone https://github.com/yourusername/peptide-immunogenicity-prediction.git
cd peptide-immunogenicity-prediction

# Create conda environment
conda create -n peptide-pred python=3.8
conda activate peptide-pred

# Install dependencies
conda install -c conda-forge rdkit pytorch pandas scikit-learn
conda install -c conda-forge gudhi  # Optional
pip install transformers            # Optional, for ESM-2

Option 2: Pip
bash

git clone https://github.com/yourusername/peptide-immunogenicity-prediction.git
cd peptide-immunogenicity-prediction

pip install -r requirements.txt

🚀 Quick Start

    Generate Dataset
    bash

python data.py

Extract Features
bash

# Generate molecular conformations
python md.py
# Compute topological features
python topo.py
# Extract ESM-2 sequence embeddings
python esm.py

Train Model
bash

python train.py

Make Predictions
bash

    # Single sequence prediction
    python infer.py --sequence "NPCNNPLKARCMK"
    # Batch prediction from file
    python infer.py --file peptides.txt
    # Interactive mode
    python infer.py

📁 Project Structure
stylus

peptide-immunogenicity-prediction/
├── data.py               # Dataset generation
├── md.py                 # Molecular dynamics & conformations
├── topo.py               # Topological feature extraction
├── esm.py                # ESM-2 sequence feature extraction
├── model.py              # Neural network architecture
├── train.py              # Training pipeline
├── infer.py              # Inference and prediction
├── requirements.txt      # Python dependencies
├── data/                 # Generated datasets and features
│   ├── peptide_dataset.csv
│   ├── conformations/
│   ├── topological_features.pkl
│   └── esm_features.pkl
├── models/               # Trained models
│   ├── immunogenicity_model.pth
│   └── feature_processor.pkl
└── README.md

🔬 Technical Details
Topological Features

    Persistent Homology: Betti numbers across multiple dimensions
    Geometric Descriptors: Radius of gyration, asphericity
    Connectivity Analysis: Distance-based molecular connectivity

Sequential Features

    ESM-2 Embeddings: State-of-the-art protein language model representations
    Multi-scale Aggregation: Mean, max, std pooling across sequence positions

Model Architecture

    Cross-Attention Mechanism
    Multi-head Attention: 8 attention heads
    Residual Connections: Skip connections with layer normalization
    Classification Head: Multi-layer perceptron with dropout

📊 Performance
Metric	Score
Accuracy	[TODO]
Precision	[TODO]
Recall	[TODO]
F1-Score	[TODO]
AUC-ROC	[TODO]
💡 Usage Examples
Python API
python

from infer import ImmunogenicityPredictor_Pipeline

# Initialize predictor
predictor = ImmunogenicityPredictor_Pipeline()

# Single prediction
result = predictor.predict("NPCNNPLKARCMK")
print(f"Immunogenicity: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
sequences = ["NPCNNPLKARCMK", "LCKTATFEDVERR", "QAINYRQMWGDR"]
results = predictor.predict_batch(sequences)

Command Line
bash

# Predict single sequence
python infer.py --sequence "NPCNNPLKARCMK"
# Output:
# Sequence: NPCNNPLKARCMK
# Prediction: Immunogenic
# Confidence: 0.8945
# Probabilities:
#   Non-immunogenic: 0.1055
#   Immunogenic: 0.8945

# Batch prediction
echo -e "NPCNNPLKARCMK\nLCKTATFEDVERR\nQAINYRQMWGDR" > sequences.txt
python infer.py --file sequences.txt

🔧 Configuration

Model Parameters

    hidden_dim: Hidden dimension size (default: 256)
    num_heads: Number of attention heads (default: 8)
    dropout: Dropout rate (default: 0.1)
    learning_rate: Learning rate (default: 1e-3)

Feature Parameters

    n_conformations: Number of conformations per peptide (default: 10)
    max_dimension: Maximum homological dimension (default: 3)
    max_edge_length: Maximum edge length for Rips complex (default: 15.0)

🤝 Contributing

    Fork the repository
    Create a feature branch (git checkout -b feature/amazing-feature)
    Commit your changes (git commit -m 'Add amazing feature')
    Push to the branch (git push origin feature/amazing-feature)
    Open a Pull Request

📝 Citation

If you use this work in your research, please cite:
bibtex

@article{peptide_immunogenicity_2024,
  title={Peptide Immunogenicity Prediction using Topological and Sequential Features},
  author={Your Name},
  journal={Journal of Computational Biology},
  year={2024},
  volume={X},
  pages={XXX-XXX}
}

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    ESM-2 for protein language model embeddings
    RDKit for molecular representation and conformation generation
    GUDHI for topological data analysis
    PyTorch for deep learning framework

📞 Contact

    Author: TO BE ADD
    Email: admin@bayvaxbop.com

🐛 Known Issues

    ESM-2 model requires significant GPU memory (>8GB recommended)
    Large peptide sequences (>100 residues) may require increased memory
    GUDHI installation can be challenging on some systems

🔮 Future Work

    Integration with AlphaFold2 structures
    Support for modified amino acids
    Web interface development
    Multi-species immunogenicity prediction
    Real-time streaming predictions

⭐ If you find this project useful, please give it a star!
❤ Happy Researching!
