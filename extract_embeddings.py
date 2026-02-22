"""
Embedding extraction script for satellite image retrieval.

Loads a trained embedder checkpoint produced by ``run.py`` (with
``task: embedding``) and runs inference over a chosen dataset split,
saving every image's L2-normalised embedding together with its filename
and label to disk.  The resulting file can be used directly for
nearest-neighbour retrieval evaluation or database construction.

Output formats
--------------
``--output_format pt``  (default)
    A single ``<dataset>_<split>_embeddings.pt`` file loadable via
    ``torch.load()``.  Contains a dict with keys:

    * ``embeddings``  – ``torch.Tensor``  (N, D)  float32
    * ``labels``      – ``torch.Tensor``  (N,) or (N, C)
    * ``filenames``   – list[str]         length N

``--output_format npz``
    A NumPy compressed archive ``.npz`` with the same three arrays.

Usage
-----
.. code-block:: bash

    python extract_embeddings.py \\
        --config  configs/embedding_exp.yaml \\
        --checkpoint experiments/<run>/best_ckpt_step_<N>.pth \\
        --split   test \\
        --output_dir embeddings/
"""

import argparse
import os
import pprint
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.utils import get_datasets
from diffusers import AutoencoderKL
from diffusionsat import DiffusionSatPipeline, SatUNet
from archs.ldm_extractor import LDMExtractor
from utils.logger import init_logger
from utils.tasks import create_decoder
from utils.utils import combine_configs, fix_seed, load_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from satellite images using a trained SatDiFuser embedder."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config YAML (e.g. configs/embedding_exp.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth) saved by run.py.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to extract embeddings from (default: test).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Directory where the embedding file will be saved.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="pt",
        choices=["pt", "npz"],
        help="Output format: PyTorch tensor file (.pt) or NumPy archive (.npz).",
    )
    # Allow selected config overrides on the CLI (mirrors run.py).
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = combine_configs(args, cfg)

    # Use val_batch_size for extraction (no gradient needed → can be larger).
    batch_size = cfg.get("val_batch_size", cfg.get("batch_size", 32))

    fix_seed(cfg.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = init_logger(os.path.join(args.output_dir, "extract.log"))
    logger.info("=" * 60)
    logger.info("Embedding Extraction — SatDiFuser")
    logger.info("=" * 60)
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info(
        f"Split: {args.split} | Output: {args.output_dir} | Format: {args.output_format}"
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset '{cfg.dataset_name}' — split '{args.split}'")
    dataset_train, dataset_val, dataset_test = get_datasets(cfg)
    split_map = {"train": dataset_train, "val": dataset_val, "test": dataset_test}
    dataset = split_map[args.split]

    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    logger.info(f"  {len(dataset)} samples  |  {len(loader)} batches")

    # ── DiffusionSat backbone ─────────────────────────────────────────────────
    logger.info("Initialising DiffusionSat backbone …")
    pretrained_path = cfg.pretrained_model_name_or_path
    if f"/resolution{cfg.img_size}" not in pretrained_path:
        pretrained_path = pretrained_path + f"/resolution{cfg.img_size}"

    unet = SatUNet.from_pretrained(
        pretrained_path,
        subfolder="checkpoint-150000/unet",
        revision=cfg.revision,
        num_metadata=cfg.num_metadata,
        use_metadata=cfg.use_metadata,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
    )
    unet.requires_grad_(False)
    unet.to(cfg.device)

    pipe = DiffusionSatPipeline.from_pretrained(pretrained_path, unet=unet)
    pipe.to(cfg.device)

    vae = AutoencoderKL.from_pretrained(
        pretrained_path, subfolder="vae", revision=cfg.revision
    )
    vae.requires_grad_(False)
    vae.to(cfg.device)

    # ── Feature extractor + embedder ─────────────────────────────────────────
    logger.info("Initialising LDM feature extractor …")
    ldm_extractor = LDMExtractor(cfg, pipe)

    logger.info("Initialising embedding head …")
    extraction_dims = ldm_extractor.collected_dims
    decoder = create_decoder(cfg, extraction_dims)
    decoder.to(cfg.device)
    decoder.eval()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=cfg.device)
    decoder.load_state_dict(checkpoint["decoder_net"])
    ckpt_epoch = checkpoint.get("epoch", "?")
    ckpt_step = checkpoint.get("global_step", "?")
    logger.info(f"  Checkpoint epoch={ckpt_epoch}  global_step={ckpt_step}")

    # ── Extraction loop ───────────────────────────────────────────────────────
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    all_embeddings: list = []
    all_labels: list = []
    all_lats: list = []
    all_lons: list = []
    all_filenames: list = []

    logger.info(f"Extracting embeddings from {len(dataset)} samples …")
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting [{args.split}]"):
            images = batch["rgb"].to(cfg.device)
            labels = batch["label"]  # stays on CPU
            filenames = batch["filename"]
            meta = batch.get("metadata", {})

            latents = (vae.encode(images).latent_dist.sample(generator=g) * 0.18215).to(
                cfg.device
            )

            with torch.inference_mode():
                feats, _ = ldm_extractor.forward(latents)
                embeddings, _ = decoder(feats, output_shape=None)  # (B, D)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_lats.extend(meta.get("lat", [None] * len(filenames)))
            all_lons.extend(meta.get("lon", [None] * len(filenames)))
            all_filenames.extend(filenames)

    elapsed = time.time() - t0
    emb_tensor = torch.cat(all_embeddings, dim=0)  # (N, D)
    label_tensor = torch.cat(all_labels, dim=0)  # (N,) or (N, C)

    logger.info(
        f"Done in {elapsed:.1f}s — "
        f"{len(all_filenames)} embeddings of dim {emb_tensor.shape[1]}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    stem = os.path.join(args.output_dir, f"{cfg.dataset_name}_{args.split}_embeddings")

    lats_out = [v if v is not None else float("nan") for v in all_lats]
    lons_out = [v if v is not None else float("nan") for v in all_lons]

    if args.output_format == "pt":
        out_path = stem + ".pt"
        torch.save(
            {
                "embeddings": emb_tensor,
                "labels": label_tensor,
                "lats": lats_out,
                "lons": lons_out,
                "filenames": all_filenames,
            },
            out_path,
        )
    else:
        out_path = stem + ".npz"
        np.savez_compressed(
            out_path,
            embeddings=emb_tensor.numpy(),
            labels=label_tensor.numpy(),
            lats=np.array(lats_out, dtype=np.float64),
            lons=np.array(lons_out, dtype=np.float64),
            filenames=np.array(all_filenames),
        )

    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
