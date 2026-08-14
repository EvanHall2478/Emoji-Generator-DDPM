"""Download required models and datasets from Hugging Face.

No need to run this on ECE cluster nodes.
"""

import os

import configs
from huggingface_hub import snapshot_download


def main():
    """Download all required models and datasets."""
    cfg = configs.Config()
    snapshot_download(
        repo_id="/".join(os.path.normpath(cfg.model_name).split(os.sep)[-2:]),
        local_dir=cfg.model_name,
    )
    snapshot_download(
        repo_id="/".join(os.path.normpath(cfg.dataset_name).split(os.sep)[-2:]),
        local_dir=cfg.dataset_name,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
