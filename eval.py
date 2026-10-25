"""Evaluate a trained action tokenizer on the DROID test split: reconstruction MSE and codebook utilization.

Usage:
    python eval.py output/<run_name>/model_e<epoch>.pt
"""

import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Trajectory
from main import CHUNK_SIZE, IN_DIM, K_DIM, MODEL_KWARGS, Z_DIM
from transformer import TransformerVQVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", help="Model checkpoint saved by main.py")
    parser.add_argument("--data_dir", default="./data/droid")
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # Checkpoints from main.py (FlowVQ and the k-means baseline) use the FlowQuantizer module.
    quantizer_type = "flow" if "codebook.prior_codebook" in ckpt else "vqvae"
    model = TransformerVQVAE(
        k_dim=K_DIM,
        z_dim=Z_DIM,
        in_dim=IN_DIM,
        n_reg=ckpt["reg_emb"].shape[1],
        max_seq_len=CHUNK_SIZE + 1,
        quantizer_type=quantizer_type,
        **MODEL_KWARGS,
    )
    model.load_state_dict(ckpt)
    model.codebook.initialized = True
    print(f"==> Loaded {args.ckpt} ({quantizer_type} codebook)")
    model = model.to(device).eval()

    test_ds = Trajectory(train_mode="test", data_dir=args.data_dir, chunk_size=CHUNK_SIZE)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    n_data = 0
    sum_sq_err = 0.0
    used = torch.empty(0, dtype=torch.long)
    with torch.no_grad():
        for batch_tau in tqdm(test_dl):
            tau_input = batch_tau.float().to(device)
            (_, _, zq_inds), pred_tau = model(tau_input)
            used = torch.cat([used, zq_inds.reshape(-1).cpu().unique()]).unique()
            sum_sq_err += F.mse_loss(pred_tau, tau_input).item() * batch_tau.shape[0]
            n_data += batch_tau.shape[0]

    print(f"Test chunks:         {n_data}")
    print(f"Reconstruction MSE:  {sum_sq_err / n_data:.6e}")
    print(f"Codebook usage:      {used.numel()} / {K_DIM} ({100.0 * used.numel() / K_DIM:.2f}%)")


if __name__ == "__main__":
    main()
