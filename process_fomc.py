#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_lab_manual_to_sae.py
----------------------------

Reads multiple Excel files with columns:
  - text column (default: 'sentence')
  - label column (default: 'label', values in {0,1,2})

Produces SAE encodings for each row and saves:
  - <out_root>/<split_name>_features.npz
  - <out_root>/<split_name>_meta.csv

Each .npz contains:
  - X_sum  : (N, D) float32
  - X_mean : (N, D) float32
  - X_max  : (N, D) float32
  - token_counts : (N,) int32
  - labels : (N,) int64
  - row_ids: (N,) object (file__row_index)
"""

import os, argparse, logging, glob
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
    logger = logging.getLogger("LabManual_SAE")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger


# ----------------- Cleaning -----------------
def clean_text_generic(text: str) -> str:
    return external_clean_text(str(text))


# ----------------- Model loading -----------------
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


# ----------------- Hook to grab residuals -----------------
def gather_residual_activations(model, target_layer: int, inputs: torch.Tensor):
    target_act = None

    def hook(mod, mod_in, mod_out):
        nonlocal target_act
        target_act = mod_out[0] if isinstance(mod_out, (tuple, list)) else mod_out
        return mod_out

    h = model.model.layers[target_layer].register_forward_hook(hook)
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
        start = end - overlap
        if start < 0:
            start = 0
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
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    chunks = chunk_ids(ids, window, overlap)

    sum_vec = None
    max_vec = None
    n_total = 0

    for ch in chunks:
        with torch.no_grad():
            res = gather_residual_activations(model, layer, ch)
            acts = sae.encode(res.float())
            arr = acts.detach().cpu().numpy().squeeze(0)

        if arr.size == 0:
            continue

        n_total += arr.shape[0]
        if sum_vec is None:
            sum_vec = arr.sum(axis=0)
            max_vec = arr.max(axis=0)
        else:
            sum_vec += arr.sum(axis=0)
            max_vec = np.maximum(max_vec, arr.max(axis=0))

        del res, acts
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if n_total == 0:
        D = sae.W_dec.weight.shape[0] if hasattr(sae, "W_dec") else 0
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
                    model, sae, tokenizer, cleaned, device, layer, window, overlap
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
    ap.add_argument("--data-dir", required=True,
                    help="Directory with .xlsx files for this split (train or test)")
    ap.add_argument("--split-name", default="train",
                    help="Name of this split (e.g. 'train' or 'test')")
    ap.add_argument("--out-root", required=True,
                    help="Output directory for features")

    # Column names in Excel
    ap.add_argument("--text-col", default="sentence", help="Text column name")
    ap.add_argument("--label-col", default="label", help="Label column name")

    # Model/SAE settings (same defaults as before)
    # ap.add_argument("--hf-model", default="google/gemma-2-9b")
    # ap.add_argument("--sae-release", default="gemma-scope-9b-pt-res-canonical")
    # ap.add_argument("--sae-id", default="layer_20/width_131k/canonical")
    ap.add_argument("--hf-model", default="google/gemma-2-2b")
    ap.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    hf_token = os.getenv("HF_HUB_TOKEN", None)

    logger = make_logger(os.path.join(args.out_root, "process_lab_manual.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Split name: {args.split_name}")
    logger.info(f"Data dir: {args.data_dir}")

    model, tokenizer, sae = load_models(
        args.hf_model,
        args.sae_release,
        args.sae_id,
        device,
        hf_token,
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
        logger,
    )

    logger.info("All done.")


if __name__ == "__main__":
    main()
