# H-RDT vs. Large Behavior Models (LBMs) Architecture Comparison

This document compares the H-RDT model architecture with the Large Behavior Models (LBMs) described in ["A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation"](https://arxiv.org/abs/2507.05331).

## Executive Summary

**Both H-RDT and LBMs use Diffusion Transformer (DiT) architectures**, but they make fundamentally different design choices that trade off between **spatial precision**, **training efficiency**, and **domain adaptation**.

### Key Differences at a Glance

| Aspect | H-RDT | LBMs |
|--------|-------|------|
| **Model Size** | 16 blocks, 2,176 dim (larger) | 8 blocks, 768 dim (smaller) |
| **Vision Features** | 196 patch tokens (spatial) | 1 CLS token (global) |
| **Vision Encoder** | Frozen DINO-SigLIP + adapter | Finetuned CLIP ViT |
| **Language Encoder** | Precomputed T5 + adapter | Frozen CLIP text + projection |
| **Conditioning** | Separate cross-attention | Feature concatenation |
| **Training Efficiency** | Higher (frozen encoders) | Lower (finetunes vision) |
| **Spatial Reasoning** | Strong (patch tokens) | Limited (CLS token) |
| **Domain Adaptation** | Limited (frozen vision) | Strong (finetuned vision) |

### Which Architecture is Better?

**Neither architecture is universally better**—the choice depends on your priorities:

**Choose H-RDT if you need:**
- ✅ **Spatial precision** for tasks requiring object localization
- ✅ **Training efficiency** with frozen encoders
- ✅ **Fine-grained visual reasoning** over image regions
- ✅ **Larger model capacity** (16 blocks, 2,176 dim)

**Choose LBMs if you need:**
- ✅ **Domain adaptation** via vision encoder finetuning
- ✅ **Simpler architecture** (easier to implement/debug)
- ✅ **Proven CLIP integration** (well-established baseline)
- ✅ **Global scene understanding** (CLS token)

**For dexterous manipulation tasks** (like those in EgoDex), H-RDT's patch token approach and larger scale are likely better suited, as these tasks require precise spatial understanding and object manipulation. However, LBMs' vision finetuning strategy provides better domain adaptation if compute allows.

## Overview

Both models are designed for multitask robot manipulation and use diffusion-based action prediction. However, they differ significantly in their architectural choices.

## H-RDT Architecture

### Core Architecture: **Transformer-Based Diffusion Model**

H-RDT uses a **Transformer architecture** inspired by DiT (Diffusion Transformer), specifically designed for robotics:

```
Input: [State token | Noisy Action tokens]
         ↓
    Position Embeddings
         ↓
   Transformer Blocks (with AdaLN)
         ├── Self-Attention
         ├── Image Cross-Attention
         ├── Language Cross-Attention
         └── Feed-Forward Network
         ↓
    Action Decoder
         ↓
   Predicted Denoised Actions
```

### Key Components

1. **HRDTBlock Structure** (per layer):
   - **Self-Attention**: Processes state-action sequence
   - **Image Cross-Attention**: Conditions on visual observations
   - **Language Cross-Attention**: Conditions on language instructions
   - **Feed-Forward Network**: SwiGLU activation (from LLaMA)
   - **AdaLN Modulation**: Adaptive Layer Normalization using timestep embeddings (9 parameters per block)

2. **Diffusion Process**:
   - **Noise Schedule**: Custom logit-normal distribution (`time_noise.py`)
   - **Timestep Embedding**: Sinusoidal embeddings (similar to DiT)
   - **Forward Process**: Linear interpolation between clean actions and noise
   - **Training**: Predicts `action_gt - noise` from noisy actions

3. **Conditioning**:
   - **Visual**: DINO-SigLIP vision encoder → MLP adapter → Cross-attention
   - **Language**: T5 encoder → MLP adapter → Cross-attention
   - **State**: Concatenated with noisy actions as input tokens

4. **Model Specifications** (from `hrdt_pretrain.yaml`):
   - Hidden size: 2176
   - Depth: 16 transformer blocks
   - Attention heads: 16 (with 8 KV heads for grouped query attention)
   - Action chunk size: 16 timesteps
   - Flash Attention: Enabled

## Large Behavior Models (LBMs) - Architecture Details

Based on the paper, LBMs use a **Diffusion Transformer (DiT)** architecture, similar to H-RDT but with key differences.

### Core Architecture: **Diffusion Transformer (DiT)**

```
Input: [Visual Features | Language Features | Proprioception | Diffusion Timestep]
         (Concatenated for 2 observation timesteps, size 6,732)
         ↓
    DiT Blocks (8 blocks, embedding size 768)
         ├── Adaptive Layer Norm (adaLN) conditioning
         └── Self-attention + Feed-forward
         ↓
    Action Prediction
         ↓
   16 timesteps × 20-dim actions = 320 output dimensions
```

### Key Components

1. **Feature Extraction**:
   - **Visual**: CLIP ViT backbone (CLS token output)
     - Finetuned during training (shared across cameras)
     - Images: 256x342 → random crop → 224x224
   - **Language**: CLIP text encoder (pooled EOS token)
     - Frozen encoder, trainable projection layer
   - **Proprioception**: Concatenated with visual/language features

2. **Input Conditioning**:
   - Concatenates visual + language + proprioception features
   - Conditions on **2 timesteps** of observation features
   - Total input size: **6,732** dimensions
   - Diffusion timestep encoded with sinusoidal embedding + 2-layer MLP

3. **DiT Architecture**:
   - **8 DiT blocks** (vs. H-RDT's 16 blocks)
   - **Embedding size: 768** (vs. H-RDT's 2,176)
   - **Adaptive Layer Norm (adaLN)**: Conditions on diffusion timestep
   - Similar DiT structure as H-RDT but smaller scale

4. **Output**:
   - **16 timesteps** of **20-dimensional actions** (total 320)
   - During deployment: Execute 8 timesteps before recomputing

5. **Training**:
   - Pretraining: 48k steps, batch size 2560, LR 3e-4
   - Vision encoder LR: 3e-5 (1/10 of main model)
   - Finetuning: 10k steps (sim) / 30k steps (real), batch size 320, LR 2e-5
   - Policy loop: 10 Hz execution rate

## Key Architectural Differences

| Aspect | H-RDT | LBMs |
|--------|-------|------|
| **Backbone** | Diffusion Transformer (DiT-style) | Diffusion Transformer (DiT) |
| **Architecture Scale** | 16 blocks, 2,176 hidden dim | 8 blocks, 768 hidden dim |
| **Attention Mechanism** | Self-attention + Separate cross-attention (image + language) | Self-attention only (features concatenated as input) |
| **Input Structure** | State + noisy actions as tokens; separate cross-attention for modalities | Visual + language + proprioception + timestep concatenated (2 timesteps, size 6,732) |
| **Feature Extraction** | DINO-SigLIP (vision), T5 (language) | CLIP ViT (vision), CLIP text encoder (language) |
| **Vision Features** | Patch tokens (196 patches per image) | CLS token (1 global vector per image) |
| **Conditioning** | Modalities via separate cross-attention layers | All features concatenated before DiT blocks |
| **Position Encoding** | Learned position embeddings for state-action sequence | Sinusoidal embedding for diffusion timestep |
| **Feed-Forward** | SwiGLU (LLaMA-style) | Standard MLP (DiT-style) |
| **Normalization** | Adaptive Layer Normalization (AdaLN) with timestep | Adaptive Layer Normalization (adaLN) with timestep |
| **Output Dimensions** | 16 timesteps × 48-dim (pretrain) or 14-dim (finetune) | 16 timesteps × 20-dim = 320 total |
| **Deployment** | Full 16-step predictions | Execute 8 steps, then recompute |

## Architectural Advantages

### H-RDT Advantages:
1. **Larger Scale**: 16 blocks vs. 8 blocks, 2,176 vs. 768 hidden dimensions (more capacity)
2. **Modular Conditioning**: Separate cross-attention layers allow better control over modality interactions
3. **Separate Modality Processing**: Images and language processed independently before cross-attention
4. **Efficiency**: Flash Attention, grouped query attention (8 KV heads), gradient checkpointing
5. **Flexibility**: Easy to add/remove conditioning modalities via cross-attention
6. **Advanced Features**: SwiGLU activation (LLaMA-style), more sophisticated FFN

### LBMs Advantages:
1. **Simpler Architecture**: Concatenation-based conditioning is more straightforward
2. **CLIP Features**: Uses well-established CLIP vision/text encoders
3. **Smaller Model**: 8 blocks, 768 dim may be more efficient for deployment
4. **Proven Recipe**: Follows standard foundation model pretraining + finetuning pattern
5. **Comprehensive Evaluation**: Rigorous evaluation methodology demonstrated

## Key Architectural Differences Explained

### 1. **Input Conditioning Strategy**

**H-RDT**: **Separate cross-attention**
- State and noisy actions as input tokens
- Visual features → separate cross-attention layer
- Language features → separate cross-attention layer
- Allows model to learn optimal interaction between modalities

**LBMs**: **Concatenation-based**
- All features (visual + language + proprioception + timestep) concatenated
- Single input stream to DiT blocks
- Simpler but less flexible modality interaction

### 2. **Model Scale**

**H-RDT**: **Larger model**
- 16 transformer blocks
- 2,176 hidden dimensions
- More parameters, more capacity

**LBMs**: **Smaller model**
- 8 transformer blocks  
- 768 hidden dimensions
- More efficient, potentially faster inference

### 3. **Feature Extractors**

**H-RDT**: **DINO-SigLIP + T5**
- DINO-SigLIP for vision (custom combination)
- T5 for language (separate encoder)

**LBMs**: **CLIP unified**
- CLIP ViT for vision
- CLIP text encoder for language
- Unified representation space

### 4. **Vision Feature Extraction: CLS Token vs. Patch Tokens**

**LBMs: CLS Token Approach** 🔍
- Uses **CLIP ViT CLS token** as vision representation
- CLS token = **single global vector** per image
- Extracted from the final layer of CLIP ViT
- Size: **1 token per image** (single vector)
- Represents: Global, aggregated image information
- Advantage: Compact representation, focuses on global semantics

**H-RDT: Patch Tokens Approach** 🖼️
- Uses **ALL patch tokens** from DINO-SigLIP (second-to-last layer)
- Returns **multiple tokens per image** (one per patch)
- For 224×224 images with patch size 14: **196 patch tokens per image**
- **Concatenates DINO + SigLIP features along feature dimension** (not sequence dimension)
  - Each of 196 tokens has `(DINO_embed_dim + SigLIP_embed_dim)` features
  - DINO: ~1,024 dim, SigLIP: ~1,152 dim → Combined: ~2,176 dim per token
- Represents: **Spatial, fine-grained image information**
- Advantage: Preserves spatial structure, allows attending to specific image regions
- Used as: Cross-attention keys/values (196 tokens × 2,176 dim)

**Key Difference**:
```
LBMs:  Image → CLIP ViT → [CLS token] → Single global vector (1 token × 768-dim)
                                                    ↓
                                        Concatenated with language + proprioception
                                        (Total: 6,732 dim for 2 timesteps)

H-RDT: Image → DINO-SigLIP → [196 patch tokens] → Spatial tokens (196 tokens × 2,176-dim)
                                                    ↓
                                        Cross-attention keys/values in transformer blocks
                                        (196 tokens available for selective attention)
```

**Implications**:

**LBMs (CLS Token)**:
- ✅ **Compact**: Single vector (768-dim) per image, efficient
- ✅ **Global semantics**: CLIP CLS token captures high-level scene understanding
- ❌ **Loses spatial detail**: No information about where objects are in the image
- ❌ **Fixed aggregation**: All spatial information collapsed into one vector
- **Use case**: Good when global scene understanding is sufficient

**H-RDT (Patch Tokens)**:
- ✅ **Spatial preservation**: 196 tokens maintain spatial relationships
- ✅ **Selective attention**: Model can focus on relevant image regions (e.g., object being manipulated)
- ✅ **Fine-grained**: Preserves local visual features at patch level
- ❌ **Higher memory**: 196 tokens × 2,176 dim = ~427K parameters per image vs. 768 for CLS
- ❌ **More computation**: Cross-attention over 196 tokens vs. simple concatenation
- **Use case**: Better for tasks requiring spatial reasoning and object localization

### 5. **Encoder Training Strategy: Finetuning vs. Freezing**

**LBMs Training Strategy** 🔧
- **Vision Encoder (CLIP ViT)**: **FINETUNED** during training
  - Shared across all camera inputs
  - Learning rate: 3e-5 (1/10 of main model LR 3e-4)
  - All CLIP ViT parameters updated
- **Language Encoder (CLIP text)**: **FROZEN** during training
  - Encoder weights not updated
  - **Trainable projection layer** on top of frozen features
  - Allows adapting language features to task without changing pretrained CLIP text encoder

**H-RDT Training Strategy** ❄️
- **Vision Encoder (DINO-SigLIP)**: **FROZEN** during training
  - All parameters: `requires_grad = False`
  - Features extracted with `.detach()` (no gradient flow)
  - **Trainable `img_adapter`** projection layer (MLP)
    - Maps: Vision features (2,176 dim) → Hidden size (2,176 dim)
    - Allows adapting frozen vision features to task
- **Language Encoder (T5)**: **NOT USED** during training
  - Language embeddings are **precomputed offline** (saved as `.pt` files)
  - T5 model never loaded during training
  - Precomputation saves compute and ensures reproducibility
  - **Trainable `lang_adapter`** projection layer (MLP)
    - Maps: T5 features (4,096 dim) → Hidden size (2,176 dim)
    - Adapts frozen precomputed embeddings to task

**Key Differences**:

| Aspect | LBMs | H-RDT |
|-------|------|-------|
| **Vision Encoder** | ✅ Finetuned (LR: 3e-5) | ❌ Frozen + trainable adapter |
| **Language Encoder** | ❌ Frozen + trainable projection | ❌ Precomputed (not used in training) |
| **Vision Projection** | Not needed (encoder trained) | ✅ `img_adapter` (MLP) |
| **Language Projection** | ✅ Trainable projection layer | ✅ `lang_adapter` (MLP) |
| **Training Compute** | Higher (backprop through CLIP ViT) | Lower (no encoder backprop) |

**Implications**:

**LBMs Approach**:
- ✅ Vision features adapt to robot manipulation domain during training
- ✅ Can learn task-specific visual representations
- ❌ Higher memory/compute (gradients through large CLIP ViT)
- ❌ Requires careful learning rate scheduling (vision LR < main LR)

**H-RDT Approach**:
- ✅ More efficient training (no encoder backprop)
- ✅ Deterministic language features (precomputed once)
- ✅ Adapters provide flexibility without encoder training
- ❌ Vision features don't adapt to domain (frozen DINO-SigLIP)
- ❌ Cannot fine-tune vision representations for robot tasks

**Trade-off Summary**:
- **LBMs**: Finetune vision for domain adaptation; requires more compute
- **H-RDT**: Freeze both encoders; use trainable adapters; more efficient but less domain adaptation

### 6. **Temporal Modeling**

**H-RDT**: **Single timestep conditioning**
- Processes current observation + action sequence
- Temporal dependencies via self-attention in transformer blocks
- Image history handled via `img_history_size` parameter (typically 1)

**LBMs**: **Two timestep conditioning**
- Conditions on 2 observation timesteps
- Concatenates features from both timesteps (6,732 total size)

## Similarities

1. **Both use Diffusion Transformer (DiT)** architecture for action prediction
2. **Both use Adaptive Layer Normalization (AdaLN/adaLN)** for timestep conditioning
3. **Both support multitask learning** across diverse robot manipulation tasks
4. **Both condition on visual observations and language instructions**
5. **Both demonstrate improved performance with scale** (data and model size)
6. **Both show benefits over single-task baselines**
7. **Both predict 16 timesteps of actions** during training
8. **Both use pretraining + finetuning** strategy
9. **Both use projection/adaptation layers** for language features (though with different strategies)

## Implementation Details

### H-RDT Loss Function:
```python
noisy_action = action_gt * timestep + noise * (1 - timestep)
pred = model(state_action_tokens, timestep, img_c, lang_c)
target = action_gt - noise
loss = MSE(pred, target)
```

### H-RDT Noise Schedule:
- Custom logit-normal distribution
- Parameters: `a=5`, `beta_m=100`
- Timestep range: `[0, 0.999]`

## References

- **H-RDT Paper**: [H-RDT paper reference]
- **LBMs Paper**: [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/abs/2507.05331)
- **Diffusion Policy**: Original Diffusion Policy work (likely referenced in LBMs paper)
- **DiT**: Diffusion Transformer architecture inspiration for H-RDT

## Summary

Both H-RDT and LBMs use **Diffusion Transformer (DiT)** architectures, making them more similar than initially apparent. The key differences are:

1. **Scale**: H-RDT is larger (16 blocks, 2,176 dim) vs. LBMs (8 blocks, 768 dim)
2. **Conditioning**: H-RDT uses separate cross-attention, LBMs use concatenation
3. **Features**: H-RDT uses DINO-SigLIP+T5, LBMs use CLIP (vision+text)
4. **Vision Features**: H-RDT uses patch tokens (196 spatial tokens), LBMs use CLS token (1 global vector)
5. **Encoder Training**: LBMs finetune CLIP ViT, H-RDT freezes DINO-SigLIP + uses trainable adapters
6. **Language Handling**: LBMs freeze CLIP text + trainable projection, H-RDT uses precomputed T5 embeddings + trainable adapter
7. **Input**: H-RDT processes state+actions as tokens, LBMs concatenate all features
8. **Temporal**: LBMs condition on 2 timesteps, H-RDT on single timestep

**Training Strategy Trade-offs**:
- **LBMs**: Finetune vision encoder → Better domain adaptation, higher compute cost
- **H-RDT**: Freeze both encoders + trainable adapters → More efficient, less domain adaptation

Both are valid approaches, with H-RDT offering more architectural sophistication and efficiency, while LBMs offer proven CLIP integration and vision domain adaptation.

## References

- **H-RDT**: This repository (H-RDT codebase)
- **LBMs Paper**: [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/abs/2507.05331)
- **DiT**: Diffusion Transformer (Peebles & Xie, 2023) - architecture inspiration for both
- **CLIP**: Vision-Language model used by LBMs (Radford et al., 2021)
- **DINO-SigLIP**: Vision encoder used by H-RDT

