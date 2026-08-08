import os
import json
import numpy as np
import imageio.v2 as imageio


class DataLoader:
    def __init__(self, base_dir, split="train", half_res=False):
        self.base_dir = os.path.expanduser(base_dir)
        self.split = split
        self.half_res = half_res
        self.load_data()

    def load_data(self):
        json_path = os.path.join(self.base_dir, f"transforms_{self.split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find transforms file at: {json_path}")

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = float(meta["camera_angle_x"])
        imgs, poses = [], []

        print(f"Loading {self.split} data (half_res={self.half_res})...")
        for frame in meta["frames"]:
            fname = os.path.join(self.base_dir, frame["file_path"] + ".png")
            if not os.path.exists(fname):
                fname = os.path.join(self.base_dir, "..", frame["file_path"] + ".png")
            img = imageio.imread(fname)
            pose = np.array(frame["transform_matrix"])
            imgs.append(img)
            poses.append(pose)

        self.imgs = (np.array(imgs) / 255.0).astype(np.float32)
        self.poses = np.array(poses).astype(np.float32)

        H, W = self.imgs[0].shape[:2]
        if self.half_res:
            H, W = H // 2, W // 2
            self.imgs = self.imgs[:, ::2, ::2, :]
        self.H, self.W = H, W
        self.focal = 0.5 * W / np.tan(0.5 * self.camera_angle_x)
        self.N = len(self.imgs)
        print(
            f"Loaded {self.N} images: shape {self.imgs.shape}, focal={self.focal:.2f}"
        )

    def get_full_image_rays(self, idx):
        pose = self.poses[idx]
        i, j = np.meshgrid(
            np.arange(self.W, dtype=np.float32),
            np.arange(self.H, dtype=np.float32),
            indexing="xy",
        )
        dirs = np.stack(
            [
                (i - self.W * 0.5) / self.focal,
                -(j - self.H * 0.5) / self.focal,
                -np.ones_like(i),
            ],
            -1,
        )
        rays_d = np.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        return rays_o, rays_d
