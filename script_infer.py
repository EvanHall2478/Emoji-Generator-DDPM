from typing import List

from configs import Config
import network
import text_encoder
import torch
import train
import utils


def infer(cfg: Config):
    """Generate emoji samples from trained model with text prompts"""
    device = utils.get_device()

    ddpm_scheduler = train.DDPM(cfg)
    if cfg.text_emb_dim:
        text_enc = text_encoder.TextEncoder(cfg)
        ddpm_scheduler.set_text_encoder(text_enc)

    # Load model with correct dimensions
    model = network.UNet(cfg).to(device)
    model.load_state_dict(
        torch.load(cfg.get_path("checkpoint_path"), map_location=device))
    model.eval()

    # Set model
    ddpm_scheduler.set_model(model)

    # Generate samples
    if cfg.text_emb_dim:
        prompts = cfg.sample_prompts
        batch_size = len(prompts)
    else:
        prompts = None
        batch_size = cfg.num_samples
    img_size = cfg.img_size
    with torch.no_grad():
        samples = ddpm_scheduler.sample((batch_size, 3, img_size, img_size), prompts)
    return samples, prompts


def main():
    cfg = Config()
    cfg.apply_infer_config()
    cfg.apply_quick_test_config()

    samples, prompts = infer(cfg)
    samples_path = cfg.get_path("samples_path", epoch="infer")
    utils.plot_sample_grid(samples, prompts, samples_path)


if __name__ == "__main__":
    main()
