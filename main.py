"""Train a FlowVQ (or k-means initialized VQ-VAE) action tokenizer on DROID action chunks.

Usage:
    python main.py                                    # FlowVQ, as in the paper
    python main.py --quantizer kmeans                 # k-means initialized VQ-VAE baseline
    python main.py --ckpt output/<run>/model_ae_biattn_1000.pt   # restart from a saved auto-encoder
"""

import argparse
import os
import re

import torch
from sklearn.cluster import KMeans
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Trajectory
from logger import init_logger
from transformer import TransformerVQVAE
from vis_utils import vis_3d_codebook

# ----------------------------------------------------------------------------------------------------
# Hyper-parameters used in the paper.
# ----------------------------------------------------------------------------------------------------
IN_DIM = 7  # DROID action: 6-DoF end-effector + gripper
N_REG = 0  # one latent token per timestep
CHUNK_SIZE = 50
K_DIM = 2048  # codebook size
Z_DIM = 8  # latent dimension
LR = 1e-4
BATCH_SIZE = 512
TOTAL_STEPS = 2_000_000
MODEL_KWARGS = dict(
    h_dim=512, n_heads=8, n_blocks=8, dropout=0.1, codebook_h_dim=256, codebook_n_hiddens=3
)

FM_START_STEP = 1000  # auto-encoder warm-up steps before the codebook is initialized
FM_TRAIN_STEPS = 2000  # flow-matching steps between two codebook evaluations during initialization
FM_EVAL_STEPS = 1000  # approx. number of test batches used to evaluate the quantization error
FM_PATIENCE = 3
FM_UPDATE_EVERY = 250  # online codebook update interval (steps)
FM_UPDATE_TIMES = 1000 // FM_UPDATE_EVERY  # flow-matching steps per training step after initialization
ONLINE_UPDATE_MAX_RECON = 0.02  # online codebook updates only happen once the reconstruction is good
VIS_EVERY = 5000
LOG_EVERY = 200
TEST_EVERY = 10000


def chamfer_distance(A, B):
    dist_sq = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)) ** 2, dim=2)
    return dist_sq.min(dim=1)[0].mean() + dist_sq.min(dim=0)[0].mean()


def next_batch(data_iter, data_loader):
    """Next batch from ``data_iter``, restarting from ``data_loader`` once it is exhausted."""
    try:
        return data_iter, next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        return data_iter, next(data_iter)


@torch.no_grad()
def encode(model, batch, device):
    return model(batch.float().to(device), decode=False)[0]


@torch.no_grad()
def sample_test_latents(model, test_dl, rand_th, device):
    """Encode a random subset (~``rand_th`` of the batches) of the test set."""
    zes = None
    for batch in test_dl:
        if torch.rand(1).item() < rand_th:
            z_e = encode(model, batch, device)
            zes = z_e if zes is None else torch.cat([zes, z_e], 0)
    return zes


def init_codebook_with_flow(model, optimizer, train_dl, zes, logger, step, device):
    """FlowVQ codebook initialization.

    1. Fit the flow on encoder latents until the push-forward codebook stops improving the test
       quantization error.
    2. Refine the codebook vectors directly with the codebook loss (same stopping rule, max 3 rounds).
    3. Store the flow inverse of the refined codebook as the new prior codebook.

    NOTE: the main ``optimizer`` (over all model parameters) is used here, as in the paper experiments.
    Only the codebook receives gradients, but the Adam state of these parameters is shared with the
    auto-encoder training afterwards.

    Returns the training data iterator, which the main training loop continues from (as in the paper runs).
    """
    print("==> Initializing the codebook by training the flow matching")
    fm_i = 0
    last_vq_loss = None
    train_iter = iter(train_dl)
    while True:
        for _ in range(FM_TRAIN_STEPS):
            train_iter, fm_data = next_batch(train_iter, train_dl)
            optimizer.zero_grad()
            fm_loss, fm_log = model.codebook.loss(encode(model, fm_data, device))
            logger.update_metrics(fm_log, suffix="train/")
            if (fm_i + 1) % LOG_EVERY == 0:
                logger.log_metrics(fm_i)
            fm_loss.backward()
            optimizer.step()
            fm_i += 1

        model.codebook.update_codebook(pushforward=True, optim_steps=0)
        vq_loss = model.codebook.get_vq_loss(zes)
        if last_vq_loss is None:
            last_vq_loss = vq_loss
            patience = 0
        elif vq_loss >= (last_vq_loss - 1e-3):
            patience += 1
            if patience >= FM_PATIENCE:
                logger.update_metrics({"vq_loss": vq_loss.item()}, suffix="train/")
                logger.log_metrics(step)
                logger.log_plot({"codebook": vis_3d_codebook(zes, model.codebook.codebook)})
                break
        else:
            last_vq_loss = min(last_vq_loss, vq_loss)
            patience = 0
        print(f"==> Last vq_loss: {last_vq_loss}, patience = {patience}")

    print("==> Post optimizing the codebook")
    last_vq_loss = None
    train_iter = iter(train_dl)
    optim_epoch = 0
    while optim_epoch <= 2:
        for _ in range(FM_TRAIN_STEPS):
            train_iter, fm_data = next_batch(train_iter, train_dl)
            optimizer.zero_grad()
            z_e = encode(model, fm_data, device)
            zq_inds = model.codebook.quantize(z_e)[0]
            codebook_loss = F.mse_loss(model.codebook.embedding(zq_inds), z_e)
            codebook_loss.backward()
            optimizer.step()
        vq_loss = model.codebook.get_vq_loss(zes)
        if last_vq_loss is None:
            last_vq_loss = vq_loss
            patience = 0
        elif vq_loss >= (last_vq_loss - 1e-3):
            patience += 1
            if patience >= FM_PATIENCE:
                logger.update_metrics({"optim_vq_loss": vq_loss.item()}, suffix="train/")
                logger.log_metrics(step)
                logger.log_plot({"codebook": vis_3d_codebook(zes, model.codebook.embedding.weight.data)})
                break
        else:
            last_vq_loss = min(last_vq_loss, vq_loss)
            patience = 0
        optim_epoch += 1
        print(f"==> Last vq_loss: {last_vq_loss}, patience = {patience}")

    print("==> Update the prior codebook")
    model.codebook.prior_codebook = model.codebook.flow.encode(model.codebook.embedding.weight.data)
    return train_iter


def init_codebook_with_kmeans(model, train_dl, zes, logger, device, out_dir, max_samples=20000):
    print("==> Initializing the codebook with KMeans")
    train_zes = None
    for fm_data in train_dl:
        z_e = encode(model, fm_data, device)
        train_zes = z_e if train_zes is None else torch.cat([train_zes, z_e], 0)
        if train_zes.shape[0] > max_samples:
            break
    kmeans = KMeans(n_clusters=K_DIM).fit(train_zes.reshape(-1, train_zes.shape[-1]).cpu().numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_).to(train_zes.device)
    torch.save(centroids, f"{out_dir}/kmeans_centroids.pt")
    model.codebook.set_codebook(centroids)
    model.codebook.initialized = True
    vq_loss = model.codebook.get_vq_loss(zes)
    logger.update_metrics({"optim_vq_loss": vq_loss.item()}, suffix="train/")
    logger.log_plot({"codebook": vis_3d_codebook(zes, model.codebook.embedding.weight.data)})


@torch.no_grad()
def evaluate(model, test_dl, logger, step, device):
    model.eval()
    for batch_tau in tqdm(test_dl):
        tau_input = batch_tau.float().to(device)
        (z_e, z_q, _), pred_tau = model(tau_input)
        recon_loss = F.mse_loss(pred_tau, tau_input)
        codebook_loss = F.mse_loss(z_q, z_e)
        commitment_loss = F.mse_loss(z_e, z_q)
        loss = recon_loss + codebook_loss + 0.25 * commitment_loss
        logger.update_metrics(
            {
                "recon_loss": recon_loss.item(),
                "codebook_loss": codebook_loss.item(),
                "commitment_loss": commitment_loss.item(),
                "loss": loss.item(),
            },
            suffix="test/",
        )
    logger.log_metrics(step)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quantizer",
        choices=["flow", "kmeans"],
        default="flow",
        help="'flow': FlowVQ codebook; 'kmeans': VQ-VAE whose codebook is initialized with k-means.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Auto-encoder checkpoint 'model_ae_biattn_<step>.pt' to restart from. The codebook is "
        "re-initialized and training resumes at <step>.",
    )
    parser.add_argument("--data_dir", default="./data/droid")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="FlowVQ")
    parser.add_argument("--wandb_group", default="action_tokenization")
    return parser.parse_args()


def main():
    args = parse_args()
    use_flow = args.quantizer == "flow"
    # The k-means baseline shares the FlowQuantizer module (nearest neighbour lookup); its flow is unused.
    fm_update_every = FM_UPDATE_EVERY if use_flow else 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    fm_start_step = FM_START_STEP
    start_step = 0
    ckpt = None
    if args.ckpt is not None:
        match = re.search(r"_(\d+)\.pt$", os.path.basename(args.ckpt))
        if match is None:
            raise ValueError(f"Cannot parse the training step from {args.ckpt}, expected '..._<step>.pt'")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        assert ckpt["reg_emb"].shape[1] == N_REG, "Checkpoint was trained with a different number of registers"
        start_step = int(match.group(1))
        fm_start_step = max(fm_start_step, start_step)

    model = TransformerVQVAE(
        k_dim=K_DIM,
        z_dim=Z_DIM,
        in_dim=IN_DIM,
        n_reg=N_REG,
        max_seq_len=CHUNK_SIZE + 1,
        quantizer_type="flow",
        **MODEL_KWARGS,
    ).to(device)
    codebook_clone = None
    if ckpt is not None:
        # Restore the auto-encoder only; the codebook is re-initialized from scratch.
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in ckpt.items() if "codebook" not in k})
        model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    fm_optimizer = optim.Adam(model.codebook.parameters(), lr=1e-4)

    run_name = (
        f"action_{'flow' if use_flow else 'vqvae'}K{K_DIM}Z{Z_DIM}E{fm_start_step}F{fm_update_every}"
        f"_chunk{CHUNK_SIZE}{'_ar' if N_REG == 0 else '_mae'}_biattn"
    )
    out_dir = f"{args.output_dir}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    logger = init_logger(
        wandb_log=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=run_name,
    )

    train_ds = Trajectory(train_mode="train", data_dir=args.data_dir, chunk_size=CHUNK_SIZE)
    test_ds = Trajectory(train_mode="test", data_dir=args.data_dir, chunk_size=CHUNK_SIZE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    fm_train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    rand_th = float(FM_EVAL_STEPS) / len(test_dl)

    e_idx = 0
    train_iter = iter(train_dl)
    fm_train_iter = iter(fm_train_dl)
    for step in range(start_step, TOTAL_STEPS):
        model.train()

        try:
            batch_tau = next(train_iter)
        except StopIteration:
            e_idx += 1
            torch.save(model.state_dict(), f"{out_dir}/model_e{e_idx}.pt")
            train_iter = iter(train_dl)
            batch_tau = next(train_iter)

        # ---------------- auto-encoder / VQ step ----------------
        optimizer.zero_grad()
        tau_input = batch_tau.float().to(device)
        (z_e, z_q, _), pred_tau = model(tau_input)
        recon_loss = F.mse_loss(pred_tau, tau_input)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        loss = recon_loss + codebook_loss + 0.25 * commitment_loss
        fm_loss = torch.zeros_like(recon_loss)
        chamfer_dist = torch.zeros_like(recon_loss)
        if model.codebook.initialized:
            # How far the codebook drifted since the last online flow update.
            chamfer_dist = chamfer_distance(model.codebook.codebook, codebook_clone)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---------------- online flow-matching steps ----------------
        if use_flow and model.codebook.initialized:
            for _ in range(FM_UPDATE_TIMES):
                # NOTE: restarts from `train_dl` rather than `fm_train_dl` (same data), as in the paper runs.
                fm_train_iter, fm_batch_tau = next_batch(fm_train_iter, train_dl)
                fm_optimizer.zero_grad()
                fm_loss = model.codebook.loss(encode(model, fm_batch_tau, device))[0]
                fm_loss.backward()
                fm_optimizer.step()

        logger.update_metrics(
            {
                "recon_loss": recon_loss.item(),
                "codebook_loss": codebook_loss.item(),
                "commitment_loss": commitment_loss.item(),
                "fm_loss": fm_loss.item(),
                "chamfer_dist": chamfer_dist.item(),
                "loss": loss.item(),
            },
            suffix="train/",
        )
        if (step + 1) % LOG_EVERY == 0:
            print(f"step {step + 1}: recon_loss = {recon_loss.item()}")
            logger.log_metrics(step)

        # ---------------- codebook initialization / online updates ----------------
        if step >= fm_start_step:
            if not model.codebook.initialized:
                log_snapshot = logger.export_snapshot()
                torch.save(model.state_dict(), f"{out_dir}/model_ae_biattn_{step}.pt")
                zes = sample_test_latents(model, test_dl, rand_th, device)

                if use_flow:
                    train_iter = init_codebook_with_flow(model, optimizer, train_dl, zes, logger, step, device)
                    fm_start_step = step
                    logger.load_snapshot(log_snapshot)
                    torch.save(model.state_dict(), f"{out_dir}/model_ae_biattn_init.pt")
                else:
                    init_codebook_with_kmeans(model, train_dl, zes, logger, device, out_dir)
                codebook_clone = model.codebook.codebook.detach().clone()
            else:
                if use_flow and (step - fm_start_step) % fm_update_every == 0:
                    codebook_clone = model.codebook.codebook.detach().clone()
                    if recon_loss.item() < ONLINE_UPDATE_MAX_RECON:
                        model.codebook.update_codebook(pushforward=True, optim_steps=2)

                if (step - fm_start_step) % VIS_EVERY == 0:
                    log_snapshot = logger.export_snapshot()
                    zes = sample_test_latents(model, test_dl, rand_th, device)
                    if zes is not None:  # the random subset can be empty on tiny test sets
                        logger.log_plot({"codebook": vis_3d_codebook(zes, model.codebook.embedding.weight.data)})
                    logger.load_snapshot(log_snapshot)

        if (step + 1) % TEST_EVERY == 0:
            evaluate(model, test_dl, logger, step, device)


if __name__ == "__main__":
    main()
