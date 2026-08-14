import inspect
import os
import random
import socket
import sys
import textwrap
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    transformers.set_seed(seed)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_sample_grid(samples, prompts, samples_path):
    """Create and save a grid of sample images"""
    # Create grid
    rows, cols = (4, 4)
    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < len(samples):
            img = (samples[i].cpu().permute(1, 2, 0) + 1) / 2
            img = torch.clamp(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            title = prompts[i] if prompts else f"Sample {i+1}"
            wrapped_title = "\n".join(textwrap.wrap(title, 22))
            ax.set_title(wrapped_title, fontsize=7)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Samples saved to: {samples_path}")


def is_cluster_node():
    return socket.gethostname().startswith("ece-")


def not_implemented(final_name: str = None):
    """Helper function to raise NotImplementedError with appropriate message."""
    stack = traceback.extract_stack()
    caller = stack[-2]
    if not final_name:
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        class_name = None
        for name, obj in caller_frame.f_globals.items():
            if inspect.isclass(obj) and hasattr(obj, func_name):
                class_name = name
                break
        final_name = f"{class_name}.{func_name}" if class_name else func_name
    msg = f"{final_name} not implemented (in {caller.filename}:{caller.lineno})"
    raise NotImplementedError(msg)


if __name__ == "__main__":
    sys.exit("Intended for import.")
