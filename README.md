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

Here is the **clean `References` section** updated with your requested links, formatted for GitHub README.md.
You can paste this directly under your `## References` section.

## References

### Software & Frameworks

* Miniconda – [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Unsloth – [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
* TorchTune – [https://pytorch.org/torchtune](https://pytorch.org/torchtune)
* PyTorch – [https://pytorch.org](https://pytorch.org)
* Hugging Face – [https://huggingface.co/docs](https://huggingface.co/docs)
* Slurm – [https://slurm.schedmd.com](https://slurm.schedmd.com)

### Parallelism Documentation

* Distributed Data Parallel (DDP)
  [https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

* Tensor Parallelism
  [https://pytorch.org/docs/stable/distributed.tensor.parallel.html](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)

* Pipeline Parallelism
  [https://pytorch.org/docs/stable/distributed.pipeline.sync.html](https://pytorch.org/docs/stable/distributed.pipeline.sync.html)

* Fully Sharded Data Parallel (FSDP)
  [https://pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html)

