# BrainFIBRE

Official repository of the paper  
**BrainFIBRE: A Foundation Model via Information Decomposition for Brain Microstructure (*ECCV* 2026)**

## Overview

<!-- Overview figure will be added here -->

## 🏗️ Project Architecture

```
BrainFIBRE/
├── src/                            # Core source code
│   ├── models/                     # Model architecture and definitions
│   │   ├── vit_model.py            # Multi-modal ViT + experts
│   │   ├── pos_emb.py              # Positional encodings
│   │   └── masks.py                # Block masking and brain mask utilities
│   ├── attn_utils/                 # Attention utilities
│   │   ├── fa2_utils.py            
│   │   ├── expand_mask.py          
│   │   └── modeling_flash_attention_utils.py
│   └── utils/                      # General utilities
│       ├── misc.py
│       └── util.py
├── datasets/                       # Dataset and dataloader definitions
│   ├── dataloader_pretrain.py      # Self-supervised pretraining loader
│   ├── dataloader_internal.py      # Fine-tuning dataloader for internal corhot
│   ├── dataloader_external.py      # Fine-tuning dataloader for external corhot
│   └── util.py                     # NODDI file mapping

├── configs/                        # Configuration files
│   └── ukb_full_tuning.json        # Example fine-tuning config
├── scripts/                        # Training launch scripts
│   ├── run_pretrain.sh             # Pretraining launcher
│   └── run_finetune.sh             # Fine-tuning launcher
├── checkpoints/                    # Pretrained model weights
├── train.py                        # Self-supervised PID pretraining script
└── finetune.py                     # Downstream task fine-tuning script
```

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n brainfibre python=3.10
conda activate brainfibre

# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

pip install -e .
```

### Download Pretrained Checkpoints

<!-- Checkpoint download link will be added here -->

Place the downloaded checkpoints in:

- `checkpoints/` — BrainFIBRE pretrained model weights

### Training Pipeline

#### Self-supervised PID Pretraining

```bash
# Set your paths and run pretraining with torchrun (multi-GPU)
bash scripts/run_pretrain.sh
```

#### Downstream Task Finetuning

Fine-tune the pretrained model on downstream prediction tasks. The fine-tuning script supports both classification and regression tasks across multiple cohorts.

```bash
# Usage: run_finetune.sh <dataset> <task> <splits> <tuning_mode> [pretrained_path] [params_json] [resume_path]

# Example: full fine-tuning on HCP age prediction (split 1)
bash scripts/run_finetune.sh HCP age 1 full checkpoints/pretrain/ckpt_latest.pt hcp_finetune.json
```

Supported tuning modes: `full` (fine-tune from pretrained weights), `tfs` (train from scratch).

## Citation

If you find this repository useful, please cite our *ECCV* 2026 paper:

```bibtex
@inproceedings{brainfibre2026,
  title={BrainFIBRE: A Foundation Model via Information Decomposition for Brain Microstructure},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

*BrainFIBRE - Advancing Brain Microstructure Analysis with AI*
