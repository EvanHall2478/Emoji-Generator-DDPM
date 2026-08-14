# Emoji-Generator-DDPM: A From-Scratch Text-Conditional Diffusion Model

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-ROCm%20%7C%20CUDA%20%7C%20MPS-76b900.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
This repository contains a complete, from-scratch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** using Python and PyTorch, applied to the generation of 48×48 emoji images. Rather than relying on high-level wrapper libraries (such as HuggingFace `diffusers`), this project manually implements the core diffusion mechanics — the noise schedule, the closed-form forward process, and the ancestral reverse sampling loop.

The primary objective of this project was to build an intimate, tensor-level understanding of diffusion-based generative models: a U-Net learns to predict the noise injected into an emoji image at a given timestep, and — optionally — is steered toward a specific emoji by a natural-language caption (e.g. *"pouting cat face"*, *"mountain bicyclist"*, *"adult fairy"*) encoded by a frozen sentence-transformer.

## 🧠 Architecture & Implementation Details

The model follows a conditional U-Net design, emphasizing modularity across resolution levels:

*   **Sinusoidal Timestep Embeddings:** Implemented from scratch to encode the current diffusion step into a continuous vector, later projected through a small MLP.
    *   Embedding formula applied:

$$
\text{emb}(t) = \left[\sin\!\left(t \cdot e^{-i \cdot c}\right),\ \cos\!\left(t \cdot e^{-i \cdot c}\right)\right], \quad c = \frac{\ln(10000)}{d/2 - 1}
$$

*   **Conditional Residual Blocks:** `GroupNorm → SiLU → Conv2d` blocks with the timestep embedding and (optional) text embedding each projected and **added** into the feature map, so every block is aware of both "when" and "what" is being generated.
*   **U-Net Encoder / Bottleneck / Decoder:**
    *   `DownBlock`s max-pool between configurable numbers of residual layers per resolution.
    *   A `MidBlock` bottleneck processes the lowest-resolution features.
    *   `UpBlock`s concatenate encoder skip connections and bilinearly upsample back to full resolution.
*   **Configurable Capacity:** Channel widths per resolution level (`block_out_ch`) and residual depth (`num_layers_per_block`) are both exposed as hyperparameters.

## ⚙️ Diffusion Process: Forward Noising & Reverse Sampling

The generative pipeline is built directly on the DDPM formulation, with both directions implemented manually:

*   **Forward Process (closed-form, no learning):** Given a linear beta schedule over `timesteps=500` steps, any noise level can be sampled directly without simulating every intermediate step:

$$
z_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

*   **Reverse Process (learned denoising):** At each step the U-Net predicts the noise $\epsilon_\theta(z_t, t)$, which is used to estimate the mean of $p(z_{t-1} \mid z_t)$; posterior-variance noise is re-injected at every step except the last:

$$
z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(z_t, t)\right) + \sigma_t \epsilon
$$

*   **Ancestral Sampling Loop:** Generation starts from pure Gaussian noise and iterates `t = T-1 → 0`, applying one reverse step per timestep until a clean image emerges.

## 🎨 Text Conditioning

To make the model steerable, prompts are embedded and injected throughout the network rather than concatenated at the input:

*   **Frozen Sentence-Transformer:** Captions are encoded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim); all encoder parameters are frozen (`requires_grad=False`) so only the U-Net is trained.
*   **Additive Conditioning:** The text embedding is projected per-block and added into the residual stream alongside the timestep embedding — no cross-attention is used, keeping the conditioning pathway lightweight.
*   **Embedding-Space Sanity Checks:** A cosine-similarity utility (`find_most_similar`) is included to verify that semantically similar captions land close together in embedding space before spending compute on training.
*   **Switchable Conditioning:** Setting `text_emb_dim = None` in the config disables the text pathway entirely, producing a fully unconditional model from the same codebase.

## 🚀 Training & Sampling Infrastructure

The training and inference pipeline includes:

*   **Hardware Acceleration:** Automatic device selection across ROCm/CUDA GPUs and Apple Silicon (MPS), with CPU fallback.
*   **Checkpointing & Sample Logging:** Every `save_interval` epochs, model weights are saved and a 4×4 grid of samples (captioned, if conditional) is rendered to disk for qualitative tracking.
*   **Quick-Test Mode:** A `--quick_test` flag shrinks the model, dataset, and timestep count for fast end-to-end pipeline verification before committing to a full run.
*   **Run Tracking:** `submission.yml` maps named experiment configurations (`unconditional`, `conditional`) to their corresponding run IDs and saved checkpoints under `runs/`.

## 🔬 Relevance to Diffusion Research

Because the noise schedule, forward process, and sampling loop are fully self-contained (no `diffusers` abstraction), this codebase is structured to support:

1.  **Schedule Ablations:** Direct access to `betas`, `alphas_cumprod`, and `posterior_variance` tensors for experimenting with alternative noise schedules.
2.  **Conditioning Experiments:** The additive (rather than attention-based) text-injection mechanism is easy to swap out or extend to other conditioning signals.
3.  **Step-Level Inspection:** `ResidualBlock.forward` can optionally return intermediate activations (`return_intermediates=True`) for tracing how time and text signals propagate through the network.

## 📂 Project Structure

```text
├── network.py                          # U-Net: TimestepEmbedding, ResidualBlock, Down/Up/MidBlock
├── dataset.py                          # EmojiDataset — loads & preprocesses valhalla/emoji-dataset
├── text_encoder.py                     # Frozen sentence-transformer text encoder for conditioning
├── train.py                            # DDPM noise schedule, q_sample/p_sample/sample, training loop
├── script_infer.py                     # Load a checkpoint and generate a sample grid
├── script_download_model.py            # Pre-download the text encoder + dataset from Hugging Face
├── script_plot_timestep_embeddings.py  # Visualize the sinusoidal timestep embeddings
├── configs.py                          # Dataclass with all hyperparameters, paths, and CLI flags
├── utils.py                            # Seeding, device selection, sample-grid plotting
├── runs/                               # Saved checkpoints, configs, and generated sample grids
├── submission.yml                      # Maps named runs (unconditional/conditional) to run IDs
└── requirements.txt                    # Python dependencies
```

## 💻 Usage

**Generation Example:**
```python
import torch
from configs import Config
from train import DDPM
import network
import text_encoder

cfg = Config()
cfg.run_id = "1772513939928"  # a trained conditional run
device = "cuda"  # or "mps" / "cpu"

# Rebuild the model and DDPM scheduler
ddpm = DDPM(cfg)
model = network.UNet(cfg).to(device)
model.load_state_dict(torch.load(cfg.get_path("checkpoint_path"), map_location=device))
model.eval()
ddpm.set_model(model)
ddpm.set_text_encoder(text_encoder.TextEncoder(cfg))

# Generate a batch of emoji conditioned on text prompts
prompts = ["ferris wheel", "female mechanic", "shrugging fairy"]
with torch.no_grad():
    samples = ddpm.sample((len(prompts), 3, cfg.img_size, cfg.img_size), prompts)
```

**Command-line equivalents:**
```bash
python train.py --quick_test          # fast sanity-check training run
python train.py                       # full training run
python script_infer.py --run_id <RUN_ID>   # sample from a trained checkpoint
```

## License
MIT — see [`LICENSE`](LICENSE).
