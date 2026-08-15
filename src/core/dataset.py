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

        # Pre-blend alpha channel with white background
        if self.imgs.shape[-1] == 4:
            print("Pre-blending alpha channel with white background...")
            alpha = self.imgs[..., 3:4]
            self.imgs = self.imgs[..., :3] * alpha + (1.0 - alpha)

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

        # Pre-compute all rays
        print(f"Pre-computing all rays for {self.split} split...")
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

        dirs = np.broadcast_to(dirs, (self.N, self.H, self.W, 3))
        self.rays_d = np.einsum("nhwi,nji->nhwj", dirs, self.poses[:, :3, :3])
        self.rays_o = np.broadcast_to(
            self.poses[:, None, None, :3, 3], self.rays_d.shape
        )
        print("Ray pre-computation complete.")

    def get_full_image_rays(self, idx):
        return self.rays_o[idx], self.rays_d[idx]
