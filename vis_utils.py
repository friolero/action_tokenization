import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import manifold


def get_vis_embedding(raw_embedding, codebook_embedding, method="tsne"):
    assert method in ["tsne", "isomap"], "Incompatible dimension reducing method."
    if raw_embedding.shape[1] > 3:
        combined = np.vstack([raw_embedding, codebook_embedding])
        if method == "isomap":
            dim_reductor = manifold.Isomap(n_neighbors=12, n_components=3, p=1)
        else:
            dim_reductor = manifold.TSNE(n_components=3)
        reduced = dim_reductor.fit_transform(combined)
        return reduced[: raw_embedding.shape[0]], reduced[raw_embedding.shape[0] :]
    else:
        return raw_embedding, codebook_embedding


def vis_3d_codebook(raw_zes, raw_vocab_t, n_vis=1000, logger=None):

    zes_vis = raw_zes.reshape(-1, raw_zes.shape[-1])
    if raw_vocab_t is None:
        raw_vocab_t = raw_zes
    ze_vis = zes_vis[torch.randperm(zes_vis.size(0))][:n_vis].cpu().numpy()
    # vocab_vis = raw_vocab_t.permute(0, 2, 3, 1).flatten(0, -2)
    vocab_vis = raw_vocab_t[torch.randperm(raw_vocab_t.size(0))][:n_vis].cpu().numpy()
    ze_vis, vocab_vis = get_vis_embedding(ze_vis, vocab_vis)

    x_min, y_min, z_min = ze_vis.min(axis=0)
    x_max, y_max, z_max = ze_vis.max(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    return fig
