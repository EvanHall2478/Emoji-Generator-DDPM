from typing import Annotated, List, Optional, Tuple

from configs import Config
import dataset as dataset_lib
import einops
import matplotlib.pyplot as plt
import network
import numpy as np
import text_encoder
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from utils import not_implemented

Tensor4D = Annotated[torch.Tensor, ("batch", "channels", "height", "width")]
Tensor1D = Annotated[torch.Tensor, ("batch",)]


class DDPM:
    """Denoising Diffusion Probabilistic Model with Text Conditioning"""

    def __init__(self, cfg: Config):
        """Initialize DDPM with configuration object"""
        self.cfg = cfg
        self.device = utils.get_device()

        # Initialize variables for model and text encoder which will be populated
        # later.
        self.model = None
        self.text_encoder = None

        # Compute all noise schedule parameters at once
        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        """Setup linear noise schedule and precompute all necessary values"""
        # Linear beta schedule
        self.betas = torch.linspace(
            start=self.cfg.beta_start,
            end=self.cfg.beta_end,
            steps=self.cfg.timesteps,
        ).to(self.device)

        # ---------------------------------------- Part 2(a) [your code here]
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # posterior_variance = Beta_t * (1 - alpha_(t-1)) / (1 - alpha_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # ----------------------------------------

    def set_model(self, model):
        self.model = model
        print(f"UNet parameter count: {utils.count_trainable_parameters(model)}")

    def set_text_encoder(self, text_encoder):
        self.text_encoder = text_encoder

    @staticmethod
    def expand(tensor: Tensor1D) -> Tensor4D:
        """Expand a 1D batch tensor into a 4D tensor for broadcasting."""
        return einops.rearrange(tensor, 'b -> b 1 1 1')

    def q_sample(self, x: Tensor4D, t: Tensor1D, noise: Tensor4D) -> Tensor4D:
        """Sample from q(z_t | x)."""
        # ---------------------------------------- Part 2(b) [your code here]
        # implement z_t = sqrt(alphas_t) * x + sqrt(1-alphas_t) * epsilon
        sqrt_alphas_cumprod_t = self.expand(self.sqrt_alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = self.expand(self.sqrt_one_minus_alphas_cumprod[t])

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        # ----------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        z_t: Tensor4D,
        t: Tensor1D,
        t_index: int,
        text_embeds: Optional[Tensor4D] = None,
    ) -> Tensor4D:
        """Sample from p(z_{t-1} | z_t) with text conditioning"""
        # ---------------------------------------- Part 2(d) [your code here]
        # z_(t-1) = 1/sqrt(alpha_t) * (z_t - (betas_t) / (sqrt(1 - alphas_t)) * epsilon(z_t, t)) + sigma_t * epsilon
        # sigma_t is the sqrt posterior variance, epsilon(z_t, t) is the models predictd noise, epsilon is the noise
        betas_t = self.expand(self.betas[t])
        sqrt_one_minus_alphas_cumprod_t = self.expand(self.sqrt_one_minus_alphas_cumprod[t])
        alphas = 1.0 - self.betas
        sqrt_recip_alphas_t = self.expand(1.0 / torch.sqrt(alphas[t]))

        pred_noise = self.model(z_t, t, text_embeds)

        model_mean = sqrt_recip_alphas_t * (z_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        # posterior variance ~= 0 at t_index == 0
        if t_index == 0:
            return model_mean
        else:
            self.posterior_variance_t = self.expand(self.posterior_variance[t])
            noise = torch.randn_like(z_t)
            return model_mean + torch.sqrt(self.posterior_variance_t) * noise
        # ----------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        prompts: Optional[List[str]] = None,
    ) -> Tensor4D:
        """Generate samples conditioned on text prompts"""
        # Encode text prompts
        if prompts:
            text_embeds = self.text_encoder.encode(prompts)
        else:
            text_embeds = None

        # 1. Start with Gaussian noise tensor of given `shape`.
        # 2. Iterate from T - 1 to 0 to call `self.p_sample` with appropriate
        #   parameters.
        # 3. Return the final denoised image.
        # ---------------------------------------- Part 2(e) [your code here]
        z_T = torch.randn(shape, device=self.device)

        # Iterate from T-1 to 0
        for t_index in reversed(range(self.cfg.timesteps)):
            t = torch.full((shape[0],), t_index, device=self.device, dtype=torch.long)
            z_T = self.p_sample(z_T, t, t_index, text_embeds)
        return z_T
        # ----------------------------------------


def train_ddpm(cfg: Config):
    """Train the DDPM model using configuration"""
    print(f"Experiment: {cfg.run_id}")

    # Setup device
    device = utils.get_device()
    print(f"Using device: {device}")


    # Dataset and dataloader
    dataset = dataset_lib.EmojiDataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Initialize DDPM first.
    ddpm_scheduler = DDPM(cfg)

    # Initialize the model
    model = network.UNet(cfg).to(device)
    ddpm_scheduler.set_model(model)

    text_enc = None
    if cfg.text_emb_dim:
        text_enc = text_encoder.TextEncoder(cfg)
        ddpm_scheduler.set_text_encoder(text_enc)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    # Training loop
    model.train()
    text_embeds = None
    for epoch in range(cfg.epochs):
        total_loss = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{cfg.epochs}')
        for images, texts in pbar:
            images = images.to(device)

            # Use the text encoder to get embeddings for the `texts` batch.
            # This operation must be wrapped in a `torch.no_grad()` context manager
            # because the text encoder is frozen and we don't need to compute its gradients.
            if text_enc is not None:
                # ---------------------------------------- Part 3(b) [your code here]
                with torch.no_grad():
                    text_embeds = text_enc.encode(texts)
                # ----------------------------------------

            # Create time steps tensor
            # ---------------------------------------- Part 2(c)(i) [your code here]
            t = torch.randint(0, cfg.timesteps, (images.shape[0],), device=device)
            # ----------------------------------------

            # Calculate the loss for the model.
            # ---------------------------------------- Part 2(c)(ii-v) [your code here]
            # Initialize noise used within the model
            noise = torch.randn_like(images)
            x_noisy = ddpm_scheduler.q_sample(images, t, noise)

            predicted_noise = model(x_noisy, t, text_embeds)
            loss = F.mse_loss(predicted_noise, noise)
            # ----------------------------------------

            # The standard PyTorch backward pass and optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        pbar.close()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')

        # Save model and generate samples
        if (epoch + 1) % cfg.save_interval == 0 or (epoch + 1) == cfg.epochs:
            # Save model
            model_path = cfg.get_path("checkpoint_path", epoch=epoch + 1)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
            # Generate and save samples
            save_samples(ddpm_scheduler, cfg, epoch + 1)

    return model_path


def save_samples(ddpm_scheduler: DDPM, cfg: Config, epoch: int):
    """Generate and save sample images with text conditioning"""
    prompts = cfg.sample_prompts if cfg.text_emb_dim else None

    # Generate samples
    ddpm_scheduler.model.eval()
    img_size = cfg.img_size
    batch_size = cfg.num_samples
    with torch.no_grad():
        samples = ddpm_scheduler.sample((batch_size, 3, img_size, img_size), prompts)
    ddpm_scheduler.model.train()

    # Save as grid
    samples_path = cfg.get_path("samples_path", epoch=epoch)
    utils.plot_sample_grid(samples, prompts, samples_path)


def main():
    cfg = Config()
    cfg.apply_quick_test_config()
    cfg.save()

    print("Starting training...")
    train_ddpm(cfg)


# Example usage with configurations
if __name__ == "__main__":
    main()
