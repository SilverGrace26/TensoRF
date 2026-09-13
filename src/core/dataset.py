import os
import json
import numpy as np
import imageio.v2 as imageio


def encode_view_directions_np(directions, num_freqs=4):
    freqs = 2.0 ** np.linspace(0, num_freqs - 1, num_freqs, dtype=np.float32)
    dirs_enc = np.concatenate(
        [
            np.sin(directions[..., None] * freqs),
            np.cos(directions[..., None] * freqs),
        ],
        -1,
    )
    dirs_enc = dirs_enc.reshape(
        directions.shape[0], directions.shape[1], directions.shape[2], -1
    )
    return np.concatenate([dirs_enc, directions], axis=-1).astype(np.float32)


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

        print(
            f"Pre-computing rays, norms, and view encodings for {self.split} split..."
        )
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

        self.rays_d_norm = np.linalg.norm(self.rays_d, axis=-1).astype(np.float32)
        view_dirs = self.rays_d / self.rays_d_norm[..., None]
        self.dirs_enc = encode_view_directions_np(view_dirs)

        print("Ray pre-computation complete.")

    def get_full_image_rays(self, idx):
        return self.rays_o[idx], self.rays_d[idx]

    def get_training_chunk(
        self, chunk_size, batch_size_per_device, n_devices, is_precrop
    ):
        total_rays = n_devices * batch_size_per_device
        N_steps = chunk_size

        rays_o_chunk = np.empty(
            (N_steps, n_devices, batch_size_per_device, 3), dtype=np.float32
        )
        rays_d_chunk = np.empty(
            (N_steps, n_devices, batch_size_per_device, 3), dtype=np.float32
        )
        dirs_enc_chunk = np.empty(
            (N_steps, n_devices, batch_size_per_device, 27), dtype=np.float32
        )
        norms_chunk = np.empty(
            (N_steps, n_devices, batch_size_per_device), dtype=np.float32
        )
        rgb_chunk = np.empty(
            (N_steps, n_devices, batch_size_per_device, 3), dtype=np.float32
        )

        center_crop_size = min(self.H, self.W) // 2
        h_start = (self.H - center_crop_size) // 2
        w_start = (self.W - center_crop_size) // 2

        for step in range(N_steps):
            img_ids = np.random.randint(0, self.N, size=(total_rays,))

            if is_precrop:
                js = np.random.randint(
                    h_start, h_start + center_crop_size, size=(total_rays,)
                )
                is_ = np.random.randint(
                    w_start, w_start + center_crop_size, size=(total_rays,)
                )
            else:
                js = np.random.randint(0, self.H, size=(total_rays,))
                is_ = np.random.randint(0, self.W, size=(total_rays,))

            rays_o_chunk[step] = self.rays_o[img_ids, js, is_].reshape(
                (n_devices, batch_size_per_device, 3)
            )
            rays_d_chunk[step] = self.rays_d[img_ids, js, is_].reshape(
                (n_devices, batch_size_per_device, 3)
            )
            dirs_enc_chunk[step] = self.dirs_enc[img_ids, js, is_].reshape(
                (n_devices, batch_size_per_device, 27)
            )
            norms_chunk[step] = self.rays_d_norm[img_ids, js, is_].reshape(
                (n_devices, batch_size_per_device)
            )
            rgb_chunk[step] = self.imgs[img_ids, js, is_].reshape(
                (n_devices, batch_size_per_device, 3)
            )

        return rays_o_chunk, rays_d_chunk, dirs_enc_chunk, norms_chunk, rgb_chunk
