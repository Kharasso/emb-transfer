#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_lab_manual_to_sae_qwen25.py
-----------------------------------

Reads multiple Excel files with columns:
  - text column (default: 'sentence')
  - label column (default: 'label', values in {0,1,2})

Encodes with Qwen2.5-7B-Instruct + SAELens SAEs:
  release: qwen2.5-7b-instruct-andyrdt
  sae_id : resid_post_layer_{layer}_trainer_1  (default)

Produces:
  - <out_root>/<split_name>_features.npz
  - <out_root>/<split_name>_meta.csv

Each .npz contains:
  - X_sum        : (N, D) float32
  - X_mean       : (N, D) float32
  - X_max        : (N, D) float32
  - token_counts : (N,)   int32
  - labels       : (N,)   int64
  - row_ids      : (N,)   object (file__row_index)
"""

import os
import argparse
import logging
import glob
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_text_cleaner import clean_text as external_clean_text

try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except Exception:
    SAE_LENS_AVAILABLE = False


# ----------------- Logging -----------------
def make_logger(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(
        filename=path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("LabManual_SAE_QWEN25")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger


# ----------------- Cleaning -----------------
def clean_text_generic(text: str) -> str:
    return external_clean_text(str(text))


# ----------------- Qwen2.5 model & SAE -----------------
def load_models(
    hf_model: str,
    sae_release: str,
    sae_id: str,
    device: str,
    hf_token: Optional[str],
):
    if not SAE_LENS_AVAILABLE:
        raise RuntimeError("sae_lens not available. Install with `pip install sae-lens`.")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        use_auth_token=hf_token,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        use_auth_token=hf_token,
    )

    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device if device.startswith("cuda") else "cpu",
    )

    model.eval()
    sae.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer, sae


def get_block_module(model, layer_idx: int):
    """
    Try to find the correct block module for Qwen2.5-style architectures.
    Handles common HF layouts: model.model.layers, model.transformer.blocks, etc.
    """
    m = getattr(model, "model", None)
    if m is None:
        m = getattr(model, "transformer", None)
    if m is None:
        raise AttributeError("Could not find .model or .transformer on the Qwen model.")

    if hasattr(m, "layers"):
        return m.layers[layer_idx]
    if hasattr(m, "blocks"):
        return m.blocks[layer_idx]
    if hasattr(m, "h"):
        return m.h[layer_idx]

    raise AttributeError("Backbone has no .layers/.blocks/.h; please inspect Qwen architecture.")


# ----------------- Hook into Qwen -----------------
def gather_residual_activations(model, target_layer: int, inputs: torch.Tensor):
    """
    Grab block output (residual post) at a given layer index.
    This matches the SAEs trained on blocks.{layer}.hook_resid_post.
    """
    target_act = None

    def hook(mod, mod_in, mod_out):
        nonlocal target_act
        target_act = mod_out[0] if isinstance(mod_out, (tuple, list)) else mod_out
        return mod_out

    layer_mod = get_block_module(model, target_layer)
    h = layer_mod.register_forward_hook(hook)
    _ = model(inputs)
    h.remove()
    return target_act


# ----------------- Chunking + featurization -----------------
def chunk_ids(input_ids: torch.Tensor, window: int, overlap: int) -> List[torch.Tensor]:
    T = input_ids.size(1)
    if T <= window:
        return [input_ids]
    chunks = []
    start = 0
    while start < T:
        end = min(T, start + window)
        chunks.append(input_ids[:, start:end])
        if end == T:
            break
        start = max(0, end - overlap)
    return chunks


def featurize_text(
    model,
    sae,
    tokenizer,
    text: str,
    device: str,
    layer: int,
    window: int,
    overlap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    ids = enc.input_ids.to(device)

    chunks = chunk_ids(ids, window, overlap)

    sum_vec = None
    max_vec = None
    n_total = 0

    for ch in chunks:
        with torch.no_grad():
            res = gather_residual_activations(model, layer, ch)  # [B, T, H]
            acts = sae.encode(res.float())                       # [B, T, D_sae]
            arr = acts.detach().cpu().numpy().squeeze(0)         # [T, D]

        if arr.size == 0:
            continue

        n_total += arr.shape[0]
        block_sum = arr.sum(axis=0)
        block_max = arr.max(axis=0)

        if sum_vec is None:
            sum_vec = block_sum
            max_vec = block_max
        else:
            sum_vec += block_sum
            max_vec = np.maximum(max_vec, block_max)

        del res, acts
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if n_total == 0:
        D = int(
            getattr(sae, "d_sae", 0)
            or (sae.W_dec.weight.shape[0] if hasattr(sae, "W_dec") else 0)
        )
        z = np.zeros((D,), np.float32)
        return z, z.copy(), z.copy(), 0

    mean_vec = (sum_vec / max(n_total, 1)).astype(np.float32)
    return sum_vec.astype(np.float32), mean_vec, max_vec.astype(np.float32), int(n_total)


# ----------------- Processing one split -----------------
def process_split(
    split_name: str,
    data_dir: str,
    out_root: str,
    model,
    tokenizer,
    sae,
    device: str,
    layer: int,
    window: int,
    overlap: int,
    text_col: str,
    label_col: str,
    hf_model: str,
    sae_release: str,
    sae_id: str,
    logger,
):
    files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not files:
        logger.warning(f"[{split_name}] No .xlsx files found in {data_dir}")
        return

    logger.info(f"[{split_name}] Found {len(files)} Excel file(s) under {data_dir}")

    X_sum, X_mean, X_max = [], [], []
    token_counts, labels, row_ids = [], [], []
    meta_rows = []

    total_rows = 0
    used_rows = 0

    for fp in files:
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            logger.exception(f"[{split_name}] Failed to read {fp}: {e}")
            continue

        if text_col not in df.columns or label_col not in df.columns:
            logger.warning(
                f"[{split_name}] File {fp} missing required columns "
                f"({text_col}, {label_col}). Skipping."
            )
            continue

        logger.info(f"[{split_name}] Processing file {fp} with {len(df)} rows")
        fname = os.path.basename(fp)

        for idx, row in df.iterrows():
            total_rows += 1

            txt = row[text_col]
            lab = row[label_col]

            if pd.isna(txt) or pd.isna(lab):
                continue

            try:
                lab_int = int(lab)
            except Exception:
                continue

            if lab_int not in (0, 1, 2):
                continue

            cleaned = clean_text_generic(txt)

            try:
                sum_vec, mean_vec, max_vec, ntok = featurize_text(
                    model,
                    sae,
                    tokenizer,
                    cleaned,
                    device,
                    layer,
                    window,
                    overlap,
                )
            except Exception as e:
                logger.exception(
                    f"[{split_name}] Encoding failed for {fname} row {idx}: {e}"
                )
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

            X_sum.append(sum_vec)
            X_mean.append(mean_vec)
            X_max.append(max_vec)
            token_counts.append(ntok)
            labels.append(lab_int)

            rid = f"{fname}__{idx}"
            row_ids.append(rid)

            meta_rows.append(
                {
                    "row_id": rid,
                    "file": fname,
                    "row_index": idx,
                    text_col: txt,
                    label_col: lab_int,
                    "ntokens": ntok,
                    "split": split_name,
                    "hf_model": hf_model,
                    "sae_release": sae_release,
                    "sae_id": sae_id,
                    "layer": layer,
                    "window": window,
                    "overlap": overlap,
                }
            )

            used_rows += 1
            if used_rows % 100 == 0:
                logger.info(f"[{split_name}] Encoded {used_rows} rows so far")

    if used_rows == 0:
        logger.warning(f"[{split_name}] No valid rows encoded.")
        return

    logger.info(
        f"[{split_name}] Finished: total rows={total_rows}, encoded={used_rows}"
    )

    X_sum = np.vstack(X_sum)
    X_mean = np.vstack(X_mean)
    X_max = np.vstack(X_max)
    token_counts = np.array(token_counts, np.int32)
    labels = np.array(labels, np.int64)
    row_ids = np.array(row_ids, dtype=object)

    os.makedirs(out_root, exist_ok=True)
    npz_path = os.path.join(out_root, f"{split_name}_features.npz")
    np.savez(
        npz_path,
        X_sum=X_sum,
        X_mean=X_mean,
        X_max=X_max,
        token_counts=token_counts,
        labels=labels,
        row_ids=row_ids,
    )

    meta_df = pd.DataFrame(meta_rows)
    meta_path = os.path.join(out_root, f"{split_name}_meta.csv")
    meta_df.to_csv(meta_path, index=False)

    logger.info(
        f"[{split_name}] Saved features to {npz_path} and metadata to {meta_path}"
    )


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Directory with .xlsx files for this split (train or test)",
    )
    ap.add_argument(
        "--split-name",
        default="train",
        help="Name of this split (e.g. 'train' or 'test')",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Output directory for features",
    )

    # Column names in Excel
    ap.add_argument("--text-col", default="sentence", help="Text column name")
    ap.add_argument("--label-col", default="label", help="Label column name")

    # Qwen2.5-7B-Instruct + SAELens defaults
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--sae-release",
        default="qwen2.5-7b-instruct-andyrdt",
        help="SAELens release name for Qwen2.5-7B-Instruct",
    )
    ap.add_argument(
        "--sae-id",
        default=None,
        help=(
            "SAE id; if omitted, auto-derived as "
            "resid_post_layer_{layer}_trainer_1 "
            "(valid layers in this release: 3,7,11,15,19,23,27)"
        ),
    )

    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument(
        "--window",
        type=int,
        default=4096,
        help="Token window for model/SAE",
    )
    ap.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Token overlap between chunks",
    )
    ap.add_argument("--device", default=None, help="e.g., cuda:0 or cpu (default: auto)")

    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)

    logger = make_logger(os.path.join(args.out_root, "process_lab_manual_qwen25.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Split name: {args.split_name}")
    logger.info(f"Data dir: {args.data_dir}")

    hf_token = os.getenv("HF_HUB_TOKEN", None)
    sae_id = args.sae_id or f"resid_post_layer_{args.layer}_trainer_1"

    model, tokenizer, sae = load_models(
        args.hf_model,
        args.sae_release,
        sae_id,
        device,
        hf_token,
    )

    logger.info(
        f"Loaded SAE release='{args.sae_release}', sae_id='{sae_id}' "
        f"(d_sae≈{getattr(sae, 'd_sae', 'unknown')})"
    )

    process_split(
        args.split_name,
        args.data_dir,
        args.out_root,
        model,
        tokenizer,
        sae,
        device,
        args.layer,
        args.window,
        args.overlap,
        args.text_col,
        args.label_col,
        args.hf_model,
        args.sae_release,
        sae_id,
        logger,
    )

    logger.info("All done.")


if __name__ == "__main__":
    main()
