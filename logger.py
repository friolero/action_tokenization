import uuid
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Logger:
    """Accumulates scalar metrics and logs their running mean to Weights & Biases (if enabled)."""

    def __init__(self, wandb_log, wandb_project, wandb_group, wandb_run_name=None, **config):
        self._log = wandb_log
        if self._log:
            assert HAS_WANDB, "Dependency missing. Install WandB by 'pip install wandb' or disable logging."
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or str(uuid.uuid4()),
                group=wandb_group,
                config=config,
            )
        self.reset_metrics()

    def log(self, metrics, step):
        if not self._log:
            return
        log = {"global_step": step}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log[k] = float(v)
            elif isinstance(v, (np.ndarray, torch.Tensor)) and v.size == 1:
                log[k] = float(v.item())
            else:
                warnings.warn(f"Unsupported log variable type for {k}: {type(v)}")
        wandb.log(log)

    def reset_metrics(self):
        self._running_metrics = {}

    def export_snapshot(self):
        """Stash the running metrics (e.g. before codebook initialization) and start afresh."""
        snapshot = self._running_metrics
        self.reset_metrics()
        return snapshot

    def load_snapshot(self, snapshot):
        self._running_metrics = snapshot

    def update_metrics(self, batch_log, suffix=""):
        for k, v in batch_log.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            assert isinstance(v, (int, float)), "Only scalar is supported for running metrics."
            self._running_metrics.setdefault(f"{suffix}{k}", []).append(float(v))

    def log_metrics(self, n_step):
        avg_metric = {k: np.mean(v).item() for k, v in self._running_metrics.items()}
        self.log(avg_metric, n_step)
        self.reset_metrics()

    def log_plot(self, plt_log):
        for k, fig in plt_log.items():
            if self._log:
                self.wandb_run.log({k: wandb.Image(fig)})
            plt.close(fig)


def init_logger(wandb_log, wandb_project, wandb_group, wandb_run_name, **kwargs):
    return Logger(wandb_log, wandb_project, wandb_group, wandb_run_name, **kwargs)
