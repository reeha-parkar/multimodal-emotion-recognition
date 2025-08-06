# Enhancing Contrastive Reconstruction for Multimodal Emotion Recognition with Temporal Preservation and Continuous Affect Modelling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation of multimodal emotion recognition using the Cross-Modal Adaptive Representation with Attention Transformer (CARAT) architecture, featuring variable-length temporal processing and transfer learning for continuous emotion modeling.

## Overview

This project introduces two novel contributions to multimodal emotion recognition:

1. **Custom Variable-Length CMU-MOSEI Dataset**: A preprocessed version of the CMU-MOSEI dataset that preserves temporal authenticity by maintaining variable sequence lengths across modalities (up to 217x more temporal information than traditional fixed-alignment approaches).

2. **Valence-Arousal Transfer Learning Framework**: A transfer learning approach that leverages continuous valence-arousal annotations from the OMGEmotion dataset to model continuous-to-discrete-label and modality-to-continuous-label dependencies for enhanced emotion recognition.

## Key Features

- **Variable-Length Processing**: Handles unaligned multimodal sequences without preprocessing synchronization
- **Adaptive CTC Alignment**: Dynamic temporal alignment using Connectionist Temporal Classification
- **Cross-Modal Attention**: Sophisticated attention mechanisms for multimodal fusion
- **Prototype-Based Contrastive Learning**: Enhanced discriminative feature learning
- **Transfer Learning Integration**: Continuous emotion modeling with V/A regression
- **Comprehensive Evaluation**: Detailed analysis and visualization of model performance

## Repository Structure

```
multimodal-emotion-recognition/
├── CARAT/                              # Main CARAT implementation
│   ├── models/                         # Model architectures and components
│   │   ├── models.py                   # CARAT model implementation
│   │   ├── module_encoder.py           # Transformer encoder modules
│   │   ├── until_module.py             # Utility modules (CTC, MLAttention)
│   │   ├── losses.py                   # Loss functions implementation
│   │   ├── optimization.py             # BertAdam optimizer
│   │   ├── file_utils.py               # File utilities
│   │   ├── until_config.py             # Configuration utilities
│   │   └── configs/                    # Model configuration files
│   ├── dataloaders/                    # Data loading and preprocessing
│   │   └── cmu_dataloader.py           # CMU-MOSEI data loader for custom data
│   ├── utils/                          # Utility functions
│   │   ├── eval.py                     # Evaluation metrics
│   │   └── m3ed_reader.py              # M3ED dataset reader
│   ├── data/                           # Dataset files (not tracked)
│   │   ├── cmu_mosei_unaligned_ree.pt  # Custom unaligned dataset
│   │   ├── omg_emotion_data.pt         # OMG emotion dataset
│   │   ├── train_valid_test.pt         # Original aligned dataset
│   │   └── readme.txt                  # Data directory readme
│   ├── model_saved/                    # Trained model checkpoints
│   │   ├── aligned/                    # Custom unaligned model checkpoints with logs
│   │   ├── unaligned/                  # Unaligned model checkpoints with logs
│   ├── main.py                         # Main training script
│   ├── util.py                         # General utilities
│   ├── train.sh                        # Original training script
│   ├── train_custom.sh                 # Training script for custom dataset
│   ├── train_m3ed.sh                   # Training script for M3ED dataset
│   ├── testing_transfer_learning.ipynb # V/A transfer learning experiments
│   ├── custom-cmu-mosei-model-analysis.ipynb # Comprehensive results analysis
│   ├── run-CARAT.ipynb                 # Interactive CARAT execution
│   ├── analyze_seq_lengths_for_training.py # Sequence length analysis
│   └── README.md                       # CARAT-specific documentation
├── My-Experiments/                     # Experimental notebooks and analysis
│   ├── cmu-mosei-exploration.ipynb     # CMU-MOSEI data exploration
│   ├── cmu-mosei-create-custom-unaligned.ipynb # Custom CMU-MOSEI dataset creation
│   ├── cmu-mosei-compare-visual-features.ipynb # Visual feature comparison
│   ├── custom-unaligned-check-data.ipynb # Dataset validation
│   ├── omgemotion-data-exploration.ipynb # OMGEmotion dataset analysis
│   ├── omgemotion-create-unaligned.ipynb # OMG unaligned data preprocessing
│   └── omgemotion-feature-extraction.ipynb # OMG feature extraction
├── CMU-MultimodalSDK/                  # CMU Multimodal SDK (submodule)
├── CMU-MultimodalSDK-Tutorials/        # SDK tutorials and examples
│   └── tutorial_interactive.ipynb      # Interactive SDK tutorial
├── OMGEmotionChallenge/                # OMGEmotion dataset and utilities
│   ├── omg_TrainVideos.csv             # Training video annotations
│   ├── omg_ValidationVideos.csv        # Validation video annotations
│   ├── omg_TestVideos_WithLabels.csv   # Test video annotations
│   ├── omg_TestVideos_WithoutLabels.csv # Test video annotations (unlabeled)
│   ├── omg_TrainTranscripts.csv        # Training transcripts
│   ├── omg_ValidationTranscripts.csv   # Validation transcripts
│   ├── omg_TestTranscripts.tsv         # Test transcripts
│   ├── prepare_data.py                 # Data preparation script
│   ├── calculateEvaluationCCC.py       # Evaluation metrics
│   ├── ListOFVideosNotAvailable_TestSet # Missing video list
│   ├── LICENSE                         # OMGEmotion license
│   ├── README.md                       # OMGEmotion documentation
│   └── requirements.txt                # OMGEmotion dependencies
├── DECLARATION.txt                     # Academic declaration
├── LICENSE                             # Project license
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Datasets

### CMU-MOSEI Dataset

The **CMU Multimodal Opinion Sentiment and Emotion Intensity (CMU-MOSEI)** dataset is a large-scale multimodal sentiment and emotion analysis dataset containing:

- **23,453 video clips** from **5,000 videos** and **1,000 distinct speakers**
- **Multimodal features**: Text (GloVe embeddings), Visual (FacetNet), Audio (COVAREP)
- **Emotion labels**: 6 discrete emotions (Happy, Sad, Anger, Surprise, Disgust, Fear)
- **Source**: [CMU Multicomp Lab](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

#### Custom Variable-Length Preprocessing

Our custom preprocessing maintains natural temporal dynamics:
- **Text**: 16-374 timesteps (GloVe 300-dimensional)
- **Visual**: 126-3,140 timesteps (FacetNet 35-dimensional)
- **Audio**: 400-10,891 timesteps (COVAREP 74-dimensional)

This preserves **up to 217x more temporal information** compared to fixed-alignment approaches.

### OMGEmotion Dataset

The **OMGEmotion Challenge Dataset** provides continuous emotion annotations:

- **Valence-Arousal annotations** for emotional expression analysis
- **Video-based emotion recognition** with temporal dynamics
- **Continuous emotion modeling** capabilities
- **Source**: [OMGEmotion Challenge](https://www2.informatik.uni-hamburg.de/wtm/OMG-Emotion/)

Used for transfer learning to model continuous emotion dependencies in our discrete emotion recognition framework.

## Novel Contributions

### 1. Custom Variable-Length CMU-MOSEI Dataset

**Problem**: Traditional multimodal emotion recognition approaches use fixed-alignment preprocessing that discards substantial temporal information (typically reducing sequences to 50-100 timesteps).

**Solution**: Our custom dataset preparation preserves natural temporal variability:
- Maintains original sequence lengths across all modalities
- Implements dynamic batch-level padding for computational efficiency
- Preserves temporal authenticity for more naturalistic emotion modeling

**Impact**: Enables the model to learn from complete temporal dynamics, capturing subtle emotional transitions and sustained emotional states that are lost in fixed-alignment approaches.

### 2. Valence-Arousal Transfer Learning Framework

**Problem**: Discrete emotion labels provide limited information about emotional intensity and dimensional relationships between emotions.

**Solution**: Transfer learning approach leveraging continuous V/A annotations:
- **Stage 1**: Train V/A regressors on OMGEmotion dataset with continuous annotations
- **Stage 2**: Transfer learned representations to CARAT model for discrete emotion recognition
- **Benefits**: Models continuous-to-discrete-label dependencies and modality-to-continuous-label relationships

**Impact**: Enhanced emotion recognition through dimensional emotion understanding and cross-dataset knowledge transfer.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (for large dataset processing)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/reeha-parkar/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Initialize CMU Multimodal SDK**:
```bash
cd CMU-MultimodalSDK
python setup.py install
cd ..
```

4. **Download datasets** (follow dataset-specific instructions in respective folders)

## Usage

### Training CARAT Model

1. **Prepare custom dataset**:
```bash
cd My-Experiments
jupyter notebook create_custom_unaligned.ipynb
```

2. **Train the model**:
```bash
cd CARAT
chmod +x train_custom.sh
./train_custom.sh
```

3. **Evaluate results**:
```bash
jupyter notebook custom-cmu-mosei-model-analysis.ipynb
```

### Transfer Learning Experiments

```bash
cd CARAT
jupyter notebook testing_transfer_learning.ipynb
```

### Key Training Parameters

- **Batch Size**: 64
- **Learning Rate**: 3e-5 with linear warmup (20% of training)
- **Hidden Size**: 512 dimensions
- **Epochs**: 15
- **Binary Threshold**: 0.2 (optimized for recall)
- **CTC Target Length**: 400 timesteps

## Model Architecture

The CARAT (Cross-Modal Adaptive Representation with Attention Transformer) architecture features:

- **Modality-Specific Encoders**: Transformer-based encoders for each modality
- **Adaptive CTC Alignment**: Dynamic sequence alignment without preprocessing
- **Cross-Modal Attention**: Learned attention mechanisms for multimodal fusion
- **Prototype-Based Learning**: Contrastive learning with emotion prototypes
- **Multi-Component Loss**: BCE, reconstruction, and contrastive loss components

## Results

### Performance Metrics (Custom CMU-MOSEI Test Set: 4,659 samples)

| Emotion  | F1-Score | Precision | Recall | Samples |
|----------|----------|-----------|--------|---------|
| Happy    | 0.7317   | 0.5919    | 0.9580 | 2,502   |
| Disgust  | 0.5058   | 0.4051    | 0.6733 | 805     |
| Sad      | 0.4808   | 0.3582    | 0.7307 | 1,129   |
| Anger    | 0.4828   | 0.4004    | 0.6078 | 1,071   |
| Surprise | 0.1200   | 0.2264    | 0.0816 | 441     |
| Fear     | 0.0424   | 0.2250    | 0.0234 | 385     |

**Overall Performance**:
- **Macro F1-Score**: 0.3939
- **Micro F1-Score**: 0.5628
- **Weighted F1-Score**: 0.5316

### Comparative Analysis

Our enhanced CARAT implementation demonstrates significant improvements over baseline methods:

| Method | Unaligned Performance | Improvement |
|--------|----------------------|-------------|
| **Our CARAT** | **Micro-F1: 0.5628** | **Baseline** |
| Original CARAT | Micro-F1: 0.544 | **+3.4% improvement** |
| Best Competing (AMP) | Micro-F1: 0.535 | **+5.2% improvement** |
| TAILOR | Micro-F1: 0.529 | **+6.4% improvement** |


### Key Findings

1. **State-of-the-art multimodal emotion recognition** with 0.563 Micro-F1, surpassing original CARAT by 3.4%
2. **Excellent performance on high-frequency emotions** (Happy: F1=0.73, Disgust: F1=0.51) demonstrating effective feature learning
3. **Successfully processes variable-length sequences** with up to 217x temporal variation preserved without preprocessing alignment
4. **Superior unaligned data handling** achieving best-in-class performance on naturalistic multimodal sequences


## Citation

If you use this work in your research, please cite:

```bibtex
@misc{parkar2025multimodal,
  title={Enhancing Contrastive Reconstruction for Multimodal Emotion Recognition with Temporal Preservation and Continuous Affect Modelling},
  author={Reeha Karim Parkar},
  year={2025},
  institution={King's College London},
  type={MSc Dissertation}
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authorship Declaration

This project comprises original work and acknowledges the use of external components - see the [DECLARATION](DECLARATION.txt) file.

## Contact

**Author**: [Reeha Karim Parkar]  
**Email**: [reeha_karim.parkar@kcl.ac.uk]  
**Institution**: King's College London  
**Department**: [Department of Informatics]  
**Program**: MSc Artificial Intelligence  

**Project Supervisor**: Dr. Helen Yannakoudakis  
**Academic Year**: 2024-2025

---

For detailed technical information, please refer to the comprehensive documentation files in the repository root or the analysis notebooks in `CARAT/` and `My-Experiments/`.