# Training of Large Language Models (LLMs) Across Different Compute Platforms

This repository contains materials for the tutorial: Supercomputing India (SCI) 2025

## Repository Structure
```
01_Basics/                 SLURM, modules, Miniconda environment setup
02_Notebook_Parallelism/   DP, TP, PP concept notebooks
03_LLM_Training_on_GPU/    DDP, Tensor, Pipeline training
04_Finetuning/             FSDP, TorchTune, Unsloth
```


## Prerequisites

* Python ≥ 3.9
* PyTorch ≥ 2.1
* CUDA ≥ 11.8
* NVIDIA GPU
* Conda / Miniconda


## Learning Flow

1. Basics
2. Parallelism notebooks
3. Distributed training
4. Fine-tuning approaches

## Supported Topics

* Distributed Data Parallel (DDP)
* Tensor Parallelism
* Pipeline Parallelism
* Fully Sharded Data Parallel (FSDP)
* Fine-tuning workflows

## References

### Documentation

* PyTorch: [https://pytorch.org](https://pytorch.org)
* TorchTune: [https://pytorch.org/torchtune](https://pytorch.org/torchtune)
* Hugging Face: [https://huggingface.co/docs](https://huggingface.co/docs)
* Slurm: [https://slurm.schedmd.com](https://slurm.schedmd.com)

### Research

* ZeRO: [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
* GPipe: [https://arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965)
* FSDP: [https://arxiv.org/abs/2304.11277](https://arxiv.org/abs/2304.11277)
