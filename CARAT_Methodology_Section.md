# Methodology: Cross-Modal Adaptive Representation with Attention Transformer (CARAT) for Variable-Length Multimodal Emotion Recognition

## 3.1 Conceptual Foundation and Theoretical Background

### 3.1.1 Attention Mechanisms in Deep Learning

The attention mechanism represents a fundamental advancement in neural network architectures, enabling models to dynamically focus on relevant portions of input sequences during processing. Originally introduced by Bahdanau et al. (2014) for neural machine translation, attention mechanisms compute context-dependent representations by learning to weight different parts of the input sequence based on their relevance to the current processing step.

The mathematical foundation of attention can be expressed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

where Q represents queries, K represents keys, V represents values, and d_k is the dimension of the key vectors. This scaled dot-product attention enables the model to establish relationships between different temporal positions within and across modalities.

### 3.1.2 Cross-Modal Attention and Multimodal Fusion

Cross-modal attention extends the attention mechanism to handle heterogeneous data modalities by learning inter-modal correspondences. Unlike traditional concatenation-based fusion approaches, cross-modal attention enables dynamic weighting of modality contributions based on the current context and task requirements. This approach addresses the fundamental challenge of temporal misalignment between modalities in naturalistic multimodal data.

The cross-modal attention mechanism computes attention weights across modalities using:

```
α_ij = exp(e_ij) / Σ_k exp(e_ik)
e_ij = a(s_i-1, h_j)
```

where α_ij represents the attention weight for modality j at time step i, and a(·) is a learned alignment function that measures the compatibility between the decoder state s_i-1 and the encoder hidden state h_j.

### 3.1.3 Reconstruction and Contrastive Learning Paradigms

Reconstruction learning enforces consistency constraints across modalities by requiring the model to recover original representations from learned embeddings. This approach promotes the learning of robust, generalizable features that capture essential semantic content while filtering out modality-specific noise. The reconstruction objective can be formulated as:

```
L_recon = ||f_decode(f_encode(x)) - x||_2^2
```

Contrastive learning, particularly through the framework of supervised contrastive loss, enhances discriminative feature learning by maximizing similarity between samples of the same class while minimizing similarity between samples of different classes. This approach is particularly effective for addressing class imbalance in emotion recognition tasks.

### 3.1.4 Connectionist Temporal Classification (CTC) for Sequence Alignment

Connectionist Temporal Classification addresses the fundamental challenge of aligning variable-length sequences without requiring explicit temporal correspondences. CTC introduces a "blank" symbol that allows the model to learn implicit alignments through a dynamic programming-based forward-backward algorithm. The CTC loss function marginalizes over all possible alignments:

```
L_CTC = -log Σ_π∈A(y) Π_t p(π_t|x)
```

where A(y) represents all valid alignments for output sequence y, and π represents a specific alignment path.

## 3.2 CARAT Architecture Overview

### 3.2.1 Model Architecture Philosophy

The Cross-Modal Adaptive Representation with Attention Transformer (CARAT) architecture implements a sophisticated multimodal fusion strategy designed to handle variable-length temporal sequences without preprocessing alignment. The architecture consists of modality-specific encoder modules, adaptive alignment components, cross-modal attention mechanisms, and prototype-based contrastive learning modules.

The overall architecture follows the principle of modular design, where each modality is processed through dedicated transformer encoders before being fused through learned attention mechanisms. This approach preserves modality-specific temporal dynamics while enabling effective cross-modal information exchange.

### 3.2.2 Modality-Specific Feature Representations

#### Text Modality: GloVe Embeddings

The text modality utilizes Global Vectors for Word Representation (GloVe) embeddings, which provide 300-dimensional dense vector representations for textual input. GloVe embeddings capture both local contextual information and global semantic relationships through matrix factorization of word co-occurrence statistics. The text sequences in the custom CMU-MOSEI dataset exhibit temporal lengths ranging from 16 to 374 timesteps, preserving the natural variability of conversational speech patterns.

The text encoder processes these embeddings through a multi-layer transformer architecture:

```
H_text = TfModel_text(Normalize(GloVe_embeddings))
```

where TfModel_text represents a configurable transformer encoder with 4 hidden layers (as specified in the training configuration).

#### Visual Modality: FacetNet Features

The visual modality employs 35-dimensional FacetNet features extracted from facial expression analysis. These features capture geometric and appearance-based facial characteristics relevant to emotion recognition. The visual sequences demonstrate extreme temporal variability, ranging from 126 to 3,140 timesteps, representing the natural variation in video segment lengths.

Visual features undergo normalization and transformer encoding:

```
H_visual = TfModel_visual(Normalize(FacetNet_features))
```

The visual transformer encoder utilizes 3 hidden layers optimized for processing facial expression dynamics.

#### Audio Modality: COVAREP Features

The audio modality utilizes Computational Paralinguistics Challenge (COVAREP) features, providing 74-dimensional acoustic representations that capture prosodic, voice quality, and spectral characteristics. Audio sequences exhibit the most extreme temporal variability, spanning 400 to 10,891 timesteps, reflecting natural speech pattern variations.

Audio processing follows the same architectural pattern:

```
H_audio = TfModel_audio(Normalize(COVAREP_features))
```

The audio transformer encoder employs 3 hidden layers specifically configured for processing extended temporal acoustic sequences.

## 3.3 Core Architectural Components

### 3.3.1 Transformer Encoder Modules (TfModel)

The TfModel components implement standard transformer encoder architectures adapted for multimodal processing. Each encoder consists of multiple TfLayer modules that combine self-attention mechanisms with feed-forward networks.

#### TfAttention Mechanism

The TfAttention module implements multi-head self-attention with the following computational flow:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

The attention mechanism enables each modality to capture long-range temporal dependencies within its feature space. The configurable number of attention heads (default: 12) allows for multiple parallel attention patterns to be learned simultaneously.

#### TfPooler and Representation Learning

The TfPooler module aggregates sequence-level representations from the transformer encoder outputs. It applies a linear transformation followed by hyperbolic tangent activation to the first token representation (analogous to BERT's [CLS] token):

```
pooled_output = tanh(W_pool · h_0 + b_pool)
```

This pooled representation serves as a fixed-dimensional summary of the variable-length input sequence.

### 3.3.2 Adaptive CTC Alignment Modules

The Adaptive CTC modules address the fundamental challenge of aligning variable-length sequences across modalities without explicit temporal correspondence annotations. Unlike traditional CTC that requires fixed target lengths, the adaptive implementation dynamically adjusts target sequence lengths based on input characteristics.

#### CTC Length Determination

The CTC target length parameter (ctc_target_length = 400) establishes the common temporal resolution for cross-modal alignment. This parameter balances computational efficiency with temporal resolution preservation. The adaptive mechanism ensures that sequences are aligned to appropriate target lengths while maintaining essential temporal dynamics:

```
target_length = min(max(input_length * compression_ratio, min_target_len), max_target_len)
```

The compression ratio adapts based on the maximum position embeddings constraints:
- Text: max_position_embeddings = 600
- Visual: max_position_embeddings = 1200  
- Audio: max_position_embeddings = 1200

#### CTC Alignment Process

The CTCModule performs alignment from source modality A to target modality B through learned position prediction:

```
P_alignment = softmax(LSTM(features_A))
aligned_features = P_alignment · features_A
```

The LSTM-based position predictor learns to map input temporal positions to output positions, with an additional "blank" symbol enabling flexible alignment patterns.

### 3.3.3 Multi-Label Attention (MLAttention)

The MLAttention modules implement label-specific attention mechanisms that enable the model to focus on emotion-relevant features for each of the six discrete emotions (Happy, Sad, Anger, Surprise, Disgust, Fear). Each MLAttention module computes emotion-specific attention weights:

```
α_emotion = softmax(W_emotion · h + b_emotion)
context_emotion = Σ_t α_emotion,t · h_t
```

This approach enables the model to learn distinct attention patterns for different emotional categories, addressing the multi-label nature of the emotion recognition task.

### 3.3.4 Cross-Modal Fusion and Reconstruction

The architecture implements sophisticated cross-modal fusion through learned linear transformations that combine information from multiple modalities:

```
h_fused = MLLinear([h_text; h_visual; h_audio])
```

Reconstruction modules (tv2a, ta2v, va2t) enforce cross-modal consistency by requiring each modality to reconstruct representations of other modalities:

```
h_reconstructed_audio = tv2a([h_text; h_visual])
```

This reconstruction constraint promotes the learning of shared semantic representations across modalities.

## 3.4 Loss Function Composition

### 3.4.1 Primary Classification Loss

The primary classification objective employs weighted Binary Cross-Entropy loss to address class imbalance in emotion distribution:

```
L_BCE = -Σ_i [w_i · y_i · log(σ(ŷ_i)) + (1-y_i) · log(1-σ(ŷ_i))]
```

The positive weights are specifically tuned for emotion classes:
- Happy: 1.0, Sad: 1.1, Anger: 1.2, Surprise: 2.5, Disgust: 1.3, Fear: 3.0

These weights compensate for the natural class imbalance, providing stronger gradients for underrepresented emotions.

### 3.4.2 Reconstruction Loss Components

Multiple reconstruction losses enforce cross-modal consistency and feature quality:

```
L_recon = recon_mse_weight · MSE(h_original, h_reconstructed)
L_aug = aug_mse_weight · MSE(h_original, h_augmented)
```

The reconstruction MSE weight (1.0) and augmentation MSE weight (1.0) ensure balanced contributions from consistency constraints.

### 3.4.3 Supervised Contrastive Loss

The SupConLoss implements supervised contrastive learning to enhance discriminative feature learning:

```
L_contrastive = -log(exp(sim(z_i, z_p)/τ) / Σ_j exp(sim(z_i, z_j)/τ))
```

where τ represents the temperature parameter (0.07), z_i is the anchor sample, z_p is a positive sample (same emotion class), and the denominator sums over all samples in the batch.

### 3.4.4 Combined Loss Function

The total loss combines all components with learnable weights:

```
L_total = lsr_clf_weight · L_BCE + recon_clf_weight · L_recon_clf + 
          aug_clf_weight · L_aug_clf + cl_weight · L_contrastive +
          recon_mse_weight · L_recon_mse + aug_mse_weight · L_aug_mse
```

The training configuration specifies:
- lsr_clf_weight: 0.01
- recon_clf_weight: 0.0  
- aug_clf_weight: 0.1
- cl_weight: 1.0
- recon_mse_weight: 1.0
- aug_mse_weight: 1.0

## 3.5 Optimization Strategy

### 3.5.1 BertAdam Optimizer

The model employs the BertAdam optimizer, which implements the Adam optimization algorithm with specific modifications for transformer-based architectures. BertAdam incorporates:

- Weight decay for regularization (default: 0.01)
- Learning rate scheduling with linear warmup
- Gradient clipping for training stability (max_grad_norm: 1.0)

The optimization parameters follow:

```
β_1 = 0.9, β_2 = 0.999, ε = 1e-6
lr_scheduled = lr · min(step/warmup_steps, sqrt(warmup_steps/step))
```

### 3.5.2 Learning Rate Configuration

The training employs a learning rate of 3e-5 with linear warmup over 20% of training steps (warmup_proportion: 0.2). Learning rate decay of 0.98 per epoch ensures gradual convergence. The coefficient learning rate (coef_lr: 0.1) provides differential learning rates for different model components.

### 3.5.3 Training Dynamics

The training configuration implements:
- Batch size: 64 samples
- Epochs: 15
- Gradient accumulation steps: 1
- Binary classification threshold: 0.2

The binary threshold of 0.2 optimizes for recall sensitivity, enabling detection of subtle emotional expressions at the cost of some precision.

## 3.6 Model Configuration and Hyperparameters

### 3.6.1 Architectural Hyperparameters

The model architecture utilizes the following key hyperparameters:

- Hidden size: 512 dimensions
- Projection size: 64 dimensions (for contrastive learning)
- Number of emotion classes: 6
- Prototype momentum: 0.99 (for moving average of prototypes)
- MoCo queue size: 8192 (for contrastive learning memory bank)

### 3.6.2 Modality-Specific Configurations

Each modality employs specific architectural configurations:

**Text Modality:**
- Input dimension: 300 (GloVe embeddings)
- Hidden layers: 4
- Max position embeddings: 600

**Visual Modality:**
- Input dimension: 35 (FacetNet features)
- Hidden layers: 3
- Max position embeddings: 1200

**Audio Modality:**
- Input dimension: 74 (COVAREP features)
- Hidden layers: 3
- Max position embeddings: 1200

### 3.6.3 Implementation Details

The implementation incorporates several technical optimizations:

- Multi-GPU training support with distributed training capabilities
- Dynamic batch-level padding for variable-length sequences
- Attention mask handling for padded sequences
- Prototype-based memory mechanisms for contrastive learning

The model processes unaligned multimodal sequences directly, eliminating the need for preprocessing synchronization while maintaining computational efficiency through batch-level optimizations.

## 3.7 Custom Dataset Preparation

The custom CMU-MOSEI dataset preparation preserves natural temporal dynamics by maintaining variable sequence lengths across all modalities. This approach contrasts with traditional fixed-alignment methods that typically downsample to common temporal resolutions (often 50-100 timesteps).

The dataset characteristics include:
- Total test samples: 4,659
- Emotion distribution: Happy (53.7%), Sad (24.2%), Anger (23.0%), Surprise (9.5%), Disgust (17.3%), Fear (8.3%)
- Temporal preservation: Up to 217x more temporal information compared to fixed-alignment approaches

This methodology enables the model to learn from naturalistic temporal patterns while addressing the computational challenges associated with extreme sequence length variations through adaptive processing strategies.

## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

3. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. Proceedings of the 23rd international conference on Machine learning.

4. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. International conference on machine learning.

5. Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global vectors for word representation. Proceedings of the 2014 conference on empirical methods in natural language processing.

6. Degottex, G., Kane, J., Drugman, T., Raitio, T., & Scherer, S. (2014). COVAREP—A collaborative voice analysis repository for speech technologies. 2014 ieee international conference on acoustics, speech and signal processing.

7. Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). Openface 2.0: Facial behavior analysis toolkit. 2018 13th IEEE international conference on automatic face & gesture recognition.

8. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
