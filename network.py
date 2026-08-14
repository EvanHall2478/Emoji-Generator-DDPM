"""This module contains classes for creating a UNet architecture for conditional DDPM."""

import math
from typing import Optional, Tuple, Union

from configs import Config
import torch
import torch.nn as nn
from torchinfo import summary
from utils import not_implemented


class TimestepEmbedding(nn.Module):
    """Sinusoidal position embeddings for time steps.
    
    Attributes:
        dim: Dimension of the embedding vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for input timesteps.
        
        Args:
            time: Input timesteps of shape (batch_size,).
            
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, dim).
        """
        half_dim = self.dim // 2
        constant = math.log(10000) / (half_dim - 1)
        # ---------------------------------------- Part 1(b)(i) [your code here]
        i = torch.arange(half_dim, device=time.device) # tensor of indices
        freq_vec = torch.exp(-i * constant) # frequency vector

        scaled_t = time.unsqueeze(1) * freq_vec

        return torch.cat([torch.sin(scaled_t), torch.cos(scaled_t)], dim=1)
        # ----------------------------------------


class ResidualBlock(nn.Module):
    """Residual block with time embedding and optional text conditioning"""

    def __init__(self, in_ch: int, out_ch: int, cfg: Config):
        """Initialize the ResidualBlock.
        
        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            cfg: Configuration object containing model hyperparameters.
        """
        super().__init__()

        self.time_mlp = None
        time_emb_dim = cfg.time_emb_dim
        if time_emb_dim:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.text_mlp = None
        text_emb_dim = cfg.text_emb_dim
        if text_emb_dim:
            self.text_mlp = nn.Sequential(nn.SiLU(), nn.Linear(text_emb_dim, out_ch))

        self.block1 = nn.Sequential(
            nn.GroupNorm(cfg.num_groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(cfg.num_groups, out_ch),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ):
        """Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, in_ch, height, width).
            time_emb: Time embedding tensor of shape (batch_size, time_emb_dim).
            text_context: Text context tensor of shape (batch_size, text_emb_dim).
            return_intermediates: If True, returns intermediate activations for analysis.
            
        Returns:
            If return_intermediates is False:
                torch.Tensor: Output tensor of shape (batch_size, out_ch, height, width).
            If return_intermediates is True:
                tuple: (output, h_after_block1, h_after_time_emb, h_after_text_emb).
        """
        h = self.block1(x)
        h_after_block1 = h.clone() if return_intermediates else None

        # Add time embedding
        if time_emb is not None and self.time_mlp is not None:
            # ---------------------------------------- Part 1(b)(ii) [your code here]
            # format the projected time to add into h
            projected_time_emb = self.time_mlp(time_emb) # [batch, out_ch]
            projected_time_emb = projected_time_emb.unsqueeze(-1).unsqueeze(-1) # [batch, out_ch, 1, 1]

            h = h + projected_time_emb
            # ----------------------------------------
        h_after_time_emb = h.clone() if return_intermediates else None

        # Add text conditioning if available
        if text_context is not None and self.text_mlp is not None:
            # ---------------------------------------- Part 3(c) [your code here]
            proj_text = self.text_mlp(text_context)
            proj_text = proj_text.unsqueeze(-1).unsqueeze(-1) # (B, out_ch, 1, 1)
            h = h + proj_text
            # ----------------------------------------
        h_after_text_emb = h.clone() if return_intermediates else None

        h = self.block2(h)
        if return_intermediates:
            return (
                h + self.shortcut(x),
                h_after_block1,
                h_after_time_emb,
                h_after_text_emb,
            )
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Downsampling block with residual layers"""

    def __init__(self, in_ch: int, out_ch: int, add_downsample: bool, cfg: Config):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, cfg)
            for i in range(cfg.num_layers_per_block)
        ])
        self.downsample = nn.MaxPool2d(2) if add_downsample else None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the down block."""
        for resnet in self.resnets:
            x = resnet(x, time_emb, text_embeds)
        output_before_downsample = x
        if self.downsample is not None:
            x = self.downsample(x)
        # The second output is used for skip connections in the decoder.
        return x, output_before_downsample


class UpBlock(nn.Module):
    """Upsampling block with residual layers"""

    def __init__(self, in_ch: int, out_ch: int, prev_out_ch: int, add_upsample: bool,
                 cfg: Config):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResidualBlock(in_ch + prev_out_ch if i == 0 else out_ch, out_ch, cfg)
            for i in range(cfg.num_layers_per_block)
        ])
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        ) if add_upsample else None

    def forward(
        self,
        x: torch.Tensor,
        skip_connection: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the up block."""
        x = torch.cat([x, skip_connection], dim=1)
        for resnet in self.resnets:
            x = resnet(x, time_emb, text_embeds)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    """Middle block with residual layers"""

    def __init__(self, in_ch: int, cfg: Config):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResidualBlock(in_ch=in_ch, out_ch=in_ch, cfg=cfg)
            for _ in range(cfg.num_layers_per_block)
        ])

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the middle block."""
        for resnet in self.resnets:
            x = resnet(x, time_emb, text_embeds)
        return x


class UNet(nn.Module):
    """U-Net architecture for Conditional DDPM.
    
    This U-Net implementation supports time conditioning and optional text conditioning
    for DDPMs. It consists of an encoder (down blocks), bottleneck (middle block),
    and decoder (up blocks) with skip connections.
    
    Attributes:
        time_embedding: Sequential module for processing timestep embeddings.
        conv_in: Initial convolutional layer.
        down_blocks: ModuleList of DownBlock modules for the encoder path.
        mid_block: MidBlock module for the bottleneck.
        up_blocks: ModuleList of UpBlock modules for the decoder path.
        conv_norm_out: Output GroupNorm layer.
        conv_act: Output activation function (SiLU).
        conv_out: Final convolutional layer to produce output.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        time_emb_dim = cfg.time_emb_dim
        self.time_embedding = None
        if time_emb_dim:
            self.time_embedding = nn.Sequential()
            # ---------------------------------------- Part 1(b)(i) [your code here]
            self.time_embedding.add_module("TimestepEmbedding", TimestepEmbedding(time_emb_dim))
            # ----------------------------------------
            self.time_embedding.add_module("linear_1",
                                           nn.Linear(time_emb_dim, time_emb_dim * 2))
            self.time_embedding.add_module("silu", nn.SiLU())
            self.time_embedding.add_module("linear_2",
                                           nn.Linear(time_emb_dim * 2, time_emb_dim))

        num_ch = cfg.num_ch
        block_out_ch = cfg.block_out_ch

        # Initial convolution
        self.conv_in = nn.Conv2d(num_ch, block_out_ch[0], 3, padding=1)

        # Encoder (Down blocks)
        self.down_blocks = nn.ModuleList([])
        for i, block_ch in enumerate(block_out_ch):
            is_final_block = i == len(block_out_ch) - 1
            down_block = DownBlock(
                in_ch=block_out_ch[i - 1] if i > 0 else block_out_ch[0],
                out_ch=block_ch,
                add_downsample=not is_final_block,
                cfg=cfg,
            )
            self.down_blocks.append(down_block)

        # Middle block
        self.mid_block = MidBlock(in_ch=block_out_ch[-1], cfg=cfg)

        # Decoder (Up blocks)
        self.up_blocks = nn.ModuleList([])
        rev_block_out_ch = list(reversed(block_out_ch))

        for i, block_ch in enumerate(rev_block_out_ch):
            is_final_block = i == len(block_out_ch) - 1
            skip_ch = block_ch
            up_block = UpBlock(
                in_ch=block_ch,
                out_ch=(rev_block_out_ch[i + 1] if i < len(rev_block_out_ch) -
                        1 else block_out_ch[0]),
                prev_out_ch=skip_ch,
                add_upsample=not is_final_block,
                cfg=cfg,
            )
            self.up_blocks.append(up_block)

        # Output
        self.conv_norm_out = nn.GroupNorm(cfg.num_groups, block_out_ch[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_ch[0], num_ch, 3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Optional[Union[torch.Tensor, float, int]] = None,
        txt_enc_hid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        Args:
            sample: Noisy input tensor of shape (batch_size, num_ch, height, width).
            timestep: Current diffusion timestep, can be a tensor, float, or int.
            txt_enc_hid: Optional text encoder hidden states for conditioning,
                shape (batch_size, seq_len, text_emb_dim).
                
        Returns:
            torch.Tensor: Predicted noise or denoised output of shape
                (batch_size, num_ch, height, width).
        """
        # Convert timestep to tensor if needed
        time_emb = None
        if timestep is not None and self.time_embedding is not None:
            if isinstance(timestep, (float, int)):
                timestep = torch.tensor([timestep], dtype=torch.long).to(sample.device)
            time_emb = self.time_embedding(timestep)

        # Initial convolution
        sample = self.conv_in(sample)

        # Encoder
        down_block_res_samples = [sample]
        for down_block in self.down_blocks:
            sample, skip_sample = down_block(sample, time_emb, txt_enc_hid)
            down_block_res_samples.append(skip_sample)

        # Middle
        sample = self.mid_block(sample, time_emb, txt_enc_hid)

        # Decoder
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples.pop()
            sample = up_block(sample, res_samples, time_emb, txt_enc_hid)

        # Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


if __name__ == "__main__":
    # Small model configuration
    cfg = Config()
    cfg.block_out_ch = (32, 64, 128)
    model = UNet(cfg)

    # ---------------------------------------- Part 1(a) [your code here]
    # Summary of the UNet with an input shape of (batch_size=2, numer_of_channels=3, height=48, width=48) 
    summary(model, input_shape = (2, 3, 48, 48))
    # ----------------------------------------
