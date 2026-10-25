import glob
import os
from time import time

import numpy as np
import tensorflow_datasets as tfds
import torch
from PIL import Image


def display_time(t_in_sec):
    h = int(t_in_sec // 3600)
    t_in_sec = t_in_sec % 3600
    m = int(t_in_sec // 60)
    t_in_sec = int(t_in_sec % 60)
    t_str = f"{h:02d}h{m:02d}m{t_in_sec:02d}s"
    return t_str


class Trajectory:

    def __init__(self, data_name="droid_100", data_dir="./data", delete_raw=False):
        self.data_dir = data_dir
        self.data_name = data_name
        for mode in ["train", "test"]:
            for success in [0, 1]:
                os.makedirs(f"{data_dir}/{data_name}/wrist_images/{mode}/{success}", exist_ok=True)

        # not os.path.isfile(f"{data_dir}/{data_name}/proprioceptives.pt"):
        if True:
            ds = tfds.load(data_name, data_dir=data_dir, split="train")
            train_pps, test_pps = self.parse_dataset(ds)
            # torch.save(pps["train"], f"{self.data_dir}/{self.data_name}/wrist_images/train/proprioceptives.pt")
            # torch.save(pps["test"], f"{self.data_dir}/{self.data_name}/wrist_images/test/proprioceptives.pt")
        else:
            train_pps = torch.load(f"{data_dir}/{data_name}/wrist_images/train/proprioceptives.pt")
            test_pps = torch.load(f"{data_dir}/{data_name}/wrist_images/test/proprioceptives.pt")

        if delete_raw:
            raw_fs = glob.glob(f"{data_dir}/{data_name}/*")
            for raw_f in raw_fs:
                if "wrist_images" in raw_f or "proprioceptives.pt" in raw_f or "parsed_data.pt" in raw_f:
                    continue
                else:
                    print(f"rm -rf {raw_f}")
                    os.system(f"rm -rf {raw_f}")

    def parse_dataset(self, ds):
        """
        "observation": {
            # gripper position state
            "gripper_position": tf.Tensor(1, dtype=float64),
            # robot Cartesian state
            "cartesian_position": tf.Tensor(6, dtype=float64),
            # joint position state
            "joint_position": tf.Tensor(7, dtype=float64),
            # wrist camera RGB left viewpoint
            "wrist_image_left": tf.Image(180, 320, 3, dtype=uint8),
            # exterior camera 1 left viewpoint
            "exterior_image_1_left": tf.Image(180, 320, 3, dtype=uint8),
            # exterior camera 2 left viewpoint
            "exterior_image_2_left": tf.Image(180, 320, 3, dtype=uint8),
        }
        """
        max_ep_len = -1
        pps = {"train": {0: [], 1: []}, "test": {0: [], 1: []}}
        ep_count = 0
        start_time = time()
        for episode in ds:

            if np.random.uniform() < 0.8:
                mode = "train"
            else:
                mode = "test"
            wrist_images = []
            proprios = []
            for i, data in enumerate(episode["steps"]):
                if i == 0:
                    li_0 = [
                        data["language_instruction"],
                        data["language_instruction_2"],
                        data["language_instruction_3"],
                    ]
                    print(li_0[0].numpy().decode("utf-8"))
                else:
                    li_1 = [
                        data["language_instruction"],
                        data["language_instruction_2"],
                        data["language_instruction_3"],
                    ]
                    if li_1 != li_0:
                        import ipdb

                        ipdb.set_trace()
                    continue
                wrist_images.append(data["observation"]["wrist_image_left"].numpy())
                proprios.append(
                    np.hstack(
                        [data["observation"]["joint_position"].numpy(), data["observation"]["gripper_position"].numpy()]
                    )
                )
                if data["is_last"] or data["reward"]:
                    success = int(data["reward"].numpy().item())
                    pps[mode][success].append(proprios)
                    for step_idx, egocentric_img in enumerate(wrist_images):
                        if os.path.isfile(
                            f"{self.data_dir}/{self.data_name}/wrist_images/{mode}/{success}/ep{len(pps[mode][success])-1:06d}_step{len(wrist_images) - 1:05d}.jpeg"
                        ):
                            break
                        Image.fromarray(egocentric_img).save(
                            f"{self.data_dir}/{self.data_name}/wrist_images/{mode}/{success}/ep{len(pps[mode][success])-1:06d}_step{step_idx:05d}.jpeg"
                        )
                    if max_ep_len < i + 1:
                        print(f"==> New max trajectory length at {i+1}")
                        max_ep_len = max(max_ep_len, i + 1)
            ep_count += 1
            if ep_count % 1000 == 0:
                torch.save(pps["train"], f"{self.data_dir}/{self.data_name}/wrist_images/train/proprioceptives.pt")
                torch.save(pps["test"], f"{self.data_dir}/{self.data_name}/wrist_images/test/proprioceptives.pt")
            if ep_count % 100 == 0:
                print(f"==> EP {ep_count} - {display_time(time() - start_time)}")
                start_time = time()
        return pps


if __name__ == "__main__":
    ds = Trajectory()
    import ipdb

    ipdb.set_trace()
