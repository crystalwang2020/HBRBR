# HBRBR

**HBRBR** (Hypergraph Bipartite Recommendation with Bilateral Representation) is a deep learning model for person-job fit built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole).

## Overview

HBRBR combines:
- **TextCNN** for encoding user resumes and job descriptions
- **BGNN (Bipartite Graph Neural Network)** for learning bilateral graph representations via adversarial training
- **Pluggable prediction heads** (dot product, bilinear, gated fusion, MLP variants)

The model evaluates recommendations from **two perspectives** — for both job seekers and employers — which is essential for bilateral recommendation scenarios.

## Architecture

```
User/Job Text → TextCNN → Feature Embeddings
                              ↓
                    BGNN (Adversarial Learning)
                              ↓
                  Enhanced Bilateral Representations
                              ↓
                      Prediction Head → Matching Score
```

**Key Components:**
- **TextCNN**: Multi-channel CNN for text feature extraction
- **BGNN**: Bipartite graph neural network with adversarial learning for bilateral representation enhancement
- **Prediction Heads**: Five variants for ablation studies (see `recbole_pjf/model/prediction_heads.py`)

## Requirements

```
recbole>=1.0.0
pytorch>=1.7.0
python>=3.7.0
networkx>=2.5.0
numpy>=1.19.0
pandas>=1.1.0
```

## Installation

```bash
pip install recbole
pip install torch torchvision
git clone https://github.com/your-repo/HBRBR.git
cd HBRBR
```

## Quick Start

### 1. Prepare Dataset

Download the **zhilian** dataset from [TIANCHI](https://tianchi.aliyun.com/dataset/dataDetail?dataId=31623) and process it:

```bash
cd dataset/zhilian
python prepare_zhilian.py
```

### 2. Train HBRBR

```bash
python run_hbrbr.py
```

Or specify model and dataset:

```bash
python run_hbrbr.py --model HBRBR --dataset zhilian
```

### 3. Ablation Studies

Run ablation variants:

```bash
# Without BGNN
python run_hbrbr.py -m HBRBR_NoBGNN

# With different prediction heads
python run_hbrbr.py -m HBRBR -p dot
python run_hbrbr.py -m HBRBR -p bilinear
python run_hbrbr.py -m HBRBR -p concat_mlp
```

Available ablation models:
- `HBRBR_NoBGNN`: Remove BGNN module
- `HBRBR_GCNOnly`: Use only GCN without adversarial learning
- `HBRBR_SharedGCN`: Shared GCN for both sides
- `HBRBR_Contrastive`: Add contrastive learning
- `HBRBR_MultiHopGCN`: Multi-hop GCN aggregation
- `HBRBR_ResidualGCN`: Residual connections in GCN
- `HBRBR_JKNetGCN`: Jumping Knowledge Networks

## Configuration

Key hyperparameters in `recbole_pjf/properties/model/HBRBR.yaml`:

```yaml
embedding_size: 64
max_sent_num: 20
max_sent_len: 30
prediction_head: elem_mlp

# BGNN parameters
bgnn_weight_decay: 0.1
bgnn_dropout: 0.3
bgnn_layer_depth: 2
bgnn_u_num: 256
bgnn_learning_rate: 0.001
```

## Data Format

| **Suffix** |           **Content**            |            **Example**            |
| :--------: | :------------------------------: | :-------------------------------: |
| **.inter** |       User-job interaction       | user_id, job_id, direct, label    |
| **.user**  |           User feature           |       user_id, age, gender        |
| **.item**  |           Job feature            |         job_id, category          |
| **.udoc**  |     Text description of user     |        user_id, user_doc          |
| **.idoc**  |     Text description of job      |         job_id, job_doc           |

Each row in `.udoc` and `.idoc` represents a sentence. Words are space-separated, and preprocessing (segmentation) is required.

## Model Details

### BGNN Module

The Bipartite Graph Neural Network (BGNN) enhances user and job representations through:
1. **Graph Convolution**: Propagates features across the bipartite graph
2. **Adversarial Learning**: Discriminator distinguishes real vs. GCN-generated features
3. **Layer-wise Refinement**: Alternates between user-side and job-side updates

### Prediction Heads

Five prediction head variants for ablation:
- `dot`: Pure dot product (no parameters)
- `bilinear`: Bilinear scoring u^T W v
- `gated`: Gated fusion with learnable weights
- `elem_mlp`: Element-wise product + MLP (current default)
- `concat_mlp`: Concatenate (product, sum) + MLP

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{hbrbr2024,
  title={HBRBR: Hypergraph Bipartite Recommendation with Bilateral Representation},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## Acknowledgments

Built upon:
- [RecBole](https://github.com/RUCAIBox/RecBole) - Unified recommendation framework
- [RecBole-PJF](https://github.com/RUCAIBox/RecBole-PJF) - Person-job fit extensions

## License

MIT License
