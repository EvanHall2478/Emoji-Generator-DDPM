# Emoji-Generator-DDPM

A from-scratch training pipeline that fine-tunes a **GPT-2 language model to complete Python source code**, built on top of the Hugging Face `transformers`/`datasets` stack. The project implements the full loop end-to-end: a streaming tokenized dataset over raw `.py` files, a custom `Trainer` integration with live generation previews during evaluation, and multiple decoding strategies for inference.

> **Note:** this repository's name/description reference DDPM and emoji generation, but the current codebase implements GPT-2 fine-tuning for code completion (see [Repo Naming](#repo-naming) below).

## Highlights

- **Custom streaming dataset pipeline** (`dataset.py`) — discovers Python source files recursively, tokenizes and concatenates them into a single token stream per category, then slides a fixed-length, fixed-stride window across the stream to produce training segments. Categories are shuffled with a seeded RNG and split into train/validation sets so no category leaks across the split.
- **Custom model architecture** (`network.py`) — a `ReGPT2LMHeadModel` built on top of Hugging Face's GPT-2 implementation, loaded and fine-tuned via `from_pretrained`.
- **Hugging Face `Trainer` integration** (`train.py`) — wraps a standard fine-tuning loop with:
  - a custom `preprocess_logits_for_metrics` step to keep evaluation memory-efficient
  - a `compute_metrics` function that reports token-level accuracy
  - a custom `TrainerCallback` (`CodeGenerationCallback`) that samples held-out prompts after every evaluation phase, generates completions with the in-training model, and writes them to disk alongside a console preview — useful for watching qualitative model behavior evolve during a run
- **Multiple decoding strategies** (`generate.py`) — greedy, multinomial (temperature) sampling, and top-k sampling, run side-by-side for the same prompt at inference time.
- **Reproducible, run-scoped configuration** (`configs.py`) — a `dataclasses`-based `Config` object with CLI argument parsing (`argparse`), auto-generated run IDs, and path templating for outputs, generations, and saved configs — plus a `--quick_test` flag that shrinks the run to a fast smoke-test configuration.
- **Cross-platform device handling** (`utils.py`) — automatic device selection across CUDA, Apple Silicon (`mps`), and CPU, with a full reproducibility routine (Python/NumPy/PyTorch seeding, deterministic cuDNN settings) and cluster-node detection for HPC environments.
- **AMD ROCm support** — `requirements.txt` ships with a ROCm 6.4 PyTorch build by default (with clear instructions to swap to a standard CUDA/CPU build), reflecting development/testing on AMD GPU hardware.

## Tech stack

| Area | Tools |
|---|---|
| Modeling | PyTorch, Hugging Face `transformers` (GPT-2) |
| Training | Hugging Face `Trainer`, `accelerate` |
| Data | Hugging Face `datasets`, custom streaming `torch.utils.data.Dataset` |
| Utilities | `einops`, `numpy`, `tqdm` |

## Project structure

```
.
├── configs.py                    # Dataclass-based experiment configuration + CLI args
├── dataset.py                    # Streaming, windowed Python-code dataset
├── network.py                    # Custom GPT-2-based language model (ReGPT2LMHeadModel)
├── network_utils.py              # Supporting model components
├── train.py                      # Fine-tuning entry point (HF Trainer + custom callback)
├── generate.py                   # Inference entry point (greedy / multinomial / top-k)
├── script_download_resources.py  # Dataset/resource download script
├── utils.py                      # Device selection, seeding, misc helpers
├── submission.yml
├── requirements.txt
└── LICENSE                       # MIT
```

## Getting started

```bash
# Install dependencies (defaults to an AMD/ROCm PyTorch build — see requirements.txt
# for the one-line change needed for NVIDIA/CPU)
pip install -r requirements.txt

# Download the training resources
python script_download_resources.py

# Fine-tune (add --quick_test for a fast smoke test)
python train.py

# Generate completions with a trained checkpoint
python generate.py --run_id <your_run_id>
```

## Repo naming

This repository is currently named and described as an emoji-generating DDPM (denoising diffusion probabilistic model), but the code implements a GPT-2 fine-tuning pipeline for Python code completion instead. If a DDPM/emoji project is still in progress, consider renaming this repo (or splitting the two projects apart) so the name matches what's actually here — that consistency matters when the repo is being reviewed as part of a portfolio.

## License

MIT — see [LICENSE](LICENSE).
