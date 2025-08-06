# CARAT Data Directory

This directory contains the datasets required for training and evaluating the CARAT model.

## Required Data Files

### 1. Custom Unaligned CMU-MOSEI Dataset
- **File**: `cmu_mosei_unaligned_ree.pt`
- **Description**: Custom preprocessed CMU-MOSEI dataset preserving variable-length temporal sequences
- **Features**: 
  - Text: 16-374 timesteps (GloVe 300-dim)
  - Visual: 126-3,140 timesteps (FacetNet 35-dim) 
  - Audio: 400-10,891 timesteps (COVAREP 74-dim)
- **Creation**: Generated using `../My-Experiments/cmu-mosei-create_custom_unaligned.ipynb`
- Hosted on [HuggingFace](https://huggingface.co/datasets/reeha-parkar/custom-unaligned-CMU-MOSEI)

### 2. Original Aligned CMU-MOSEI Dataset (Optional)
- **File**: `train_valid_test.pt`
- **Description**: Traditional fixed-length aligned version of CMU-MOSEI for baseline metrics
- **Source**: Original CARAT implementation or CMU-MultimodalSDK processing

### 3. OMGEmotion Features (For Transfer Learning)
- **File**: `omg_emotion_data.pt`
- **Description**: Preprocessed OMGEmotion dataset with V/A annotations
- **Creation**: Generated using `../My-Experiments/omgemotion-feature-extraction.ipynb` and then `../My-Experiments/omgemotion-create-unaligned.ipynb`

## Data Acquisition

### Option 1: Create Custom Dataset (Recommended)
1. Download raw CMU-MOSEI using CMU-MultimodalSDK
2. Run the preprocessing notebook:
   ```bash
   cd ../My-Experiments
   jupyter notebook create_custom_unaligned.ipynb
   ```

### Option 2: Use Preprocessed Files
- Download preprocessed datasets from [your data source/drive link]
- Place files directly in this directory

### Option 3: Original CARAT Data
- Download from original CARAT repository data sources
- Extract .csd files and convert using provided scripts

## File Structure
```
data/
├── cmu_mosei_unaligned_ree.pt    # Custom variable-length dataset (primary)
├── train_valid_test.pt           # Original aligned dataset (optional)
├── omg_emotion_data.pt           # OMGEmotion V/A dataset (for transfer learning)
└── readme.md                     # This file
```

## Data Formats

All `.pt` files are PyTorch serialized dictionaries containing:
- `train`: Training split data
- `val`: Validation split data  
- `test`: Test split data

Each split contains:
- `src-text`: Text features (GloVe embeddings)
- `src-visual`: Visual features (FacetNet)
- `src-audio`: Audio features (COVAREP)
- `labels`: Emotion labels (6-class discrete emotions)

## Usage Notes

- **Primary dataset**: Use `cmu_mosei_unaligned_ree.pt` for variable-length experiments
- **Transfer learning**: Requires `omg_emotion_data.pt` for V/A regression experiments
- **Memory requirements**: Large files (~GB size) - ensure sufficient disk space
- **Preprocessing time**: Custom dataset creation may take several hours

## Troubleshooting

- **Missing files**: Check `.gitignore` - data files are not tracked in git
- **Memory errors**: Reduce batch size or use data streaming for large files
- **Format errors**: Ensure PyTorch compatibility and proper tensor formatting

For detailed preprocessing steps, refer to the notebooks in `../My-Experiments/`.