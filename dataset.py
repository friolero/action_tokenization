import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Trajectory(Dataset):
    """Fixed-length chunks of DROID action trajectories (7-D: 6-DoF end-effector + gripper).

    On first use the RLDS dataset at ``data_dir`` is parsed with ``tensorflow_datasets`` and episodes are
    randomly split 80/20 into train/test. The result is cached in ``<data_dir>/parsed_data.pt``; delete
    that file to re-parse. NOTE: the split is drawn from the global numpy RNG, so reproducing the exact
    paper split requires the original ``parsed_data.pt``.
    """

    def __init__(self, train_mode, data_dir, chunk_size=None, success_only=True):
        assert train_mode in ["train", "test", "all"], f"Unknown train_mode {train_mode}"
        self.chunk_size = chunk_size
        self.success_only = success_only

        cache_fn = f"{data_dir}/parsed_data.pt"
        if not os.path.isfile(cache_fn):
            import tensorflow_datasets as tfds

            data_dir = os.path.normpath(data_dir)
            ds = tfds.load(os.path.basename(data_dir), data_dir=os.path.dirname(data_dir), split="train")
            train_eef_vels, test_eef_vels = self.parse_dataset(ds, self.success_only)
            torch.save({"train_eef_vels": train_eef_vels, "test_eef_vels": test_eef_vels}, cache_fn)
        else:
            print(f"==> Loading from {cache_fn}")
            data = torch.load(cache_fn, weights_only=False)
            train_eef_vels = data["train_eef_vels"]
            test_eef_vels = data["test_eef_vels"]

        if self.chunk_size is None:
            train_trajs = self.pad(train_eef_vels)
            test_trajs = self.pad(test_eef_vels)
        else:
            train_trajs = self.chunk(train_eef_vels, chunk_size)
            test_trajs = self.chunk(test_eef_vels, chunk_size)

        if train_mode == "all":
            self.trajs = train_trajs + test_trajs
        elif train_mode == "train":
            self.trajs = train_trajs
        else:
            self.trajs = test_trajs

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    @staticmethod
    def parse_dataset(ds, success_only):
        eef_vels = {"train": {1: [], 0: []}, "test": {1: [], 0: []}}
        for episode in ds:
            mode = "train" if np.random.uniform() < 0.8 else "test"
            ep_eef_vels = []
            for data in episode["steps"]:
                ep_eef_vels.append(data["action"].numpy())
                if data["is_last"] or data["reward"]:
                    success = int(data["reward"].numpy().item())
                    eef_vels[mode][success].append(np.stack(ep_eef_vels))
        if success_only:
            return eef_vels["train"][1], eef_vels["test"][1]
        return eef_vels["train"][1] + eef_vels["train"][0], eef_vels["test"][1] + eef_vels["test"][0]

    @staticmethod
    def pad(raw_trajs, pad_value=0.0):
        max_steps = max(traj.shape[0] for traj in raw_trajs)
        padded = []
        for traj in raw_trajs:
            pad_len = max_steps - traj.shape[0]
            if pad_len > 0:
                padding = np.full((pad_len, traj.shape[1]), pad_value, dtype=traj.dtype)
                traj = np.concatenate([traj, padding], axis=0)
            padded.append(traj)
        return padded

    @staticmethod
    def chunk(raw_trajs, chunk_size):
        """All overlapping windows of length ``chunk_size`` (stride 1); shorter episodes are dropped."""
        chunks = []
        for traj in raw_trajs:
            for start in range(0, traj.shape[0] - chunk_size + 1):
                chunks.append(traj[start : start + chunk_size])
        return chunks
