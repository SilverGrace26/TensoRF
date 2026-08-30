import jax
import jax.numpy as jnp
import jax.scipy
import equinox as eqx
import numpy as np
from typing import Tuple

from geometry.rays import (
    sample_along_rays,
    encode_view_directions,
    compute_ray_aabb_intersections,
)
from geometry.rendering import compute_volumetric_rendering


class TensoRF(eqx.Module):
    grid_dim: int = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    bbox_min: jax.Array
    bbox_max: jax.Array
    alpha_mask: jax.Array
    den_planes: Tuple[jax.Array, ...]
    den_lines: Tuple[jax.Array, ...]
    app_planes: Tuple[jax.Array, ...]
    app_lines: Tuple[jax.Array, ...]
    basis_mat: jax.Array
    mlp_render: eqx.nn.MLP

    def __init__(
        self,
        key,
        grid_dim=128,
        n_comp_den=[8, 8, 8],
        n_comp_app=[24, 24, 24],
        bbox_min=-1.5,
        bbox_max=1.5,
    ):
        keys = jax.random.split(key, 10)
        self.bbox_min = jnp.array([bbox_min] * 3)
        self.bbox_max = jnp.array([bbox_max] * 3)
        self.grid_dim = grid_dim

        self.alpha_mask = jnp.ones((128, 128, 128), dtype=jnp.bool_)

        backend = jax.default_backend()
        self.compute_dtype = jnp.bfloat16 if backend == "tpu" else jnp.float16

        self.den_planes, self.den_lines = self.init_tensor_components(
            keys, n_comp_den, 0
        )
        self.app_planes, self.app_lines = self.init_tensor_components(
            keys, n_comp_app, 6
        )
        self.basis_mat, self.mlp_render = self.init_decoders(keys, sum(n_comp_app))

    def init_tensor_components(self, keys, n_components, key_offset):
        planes, lines = [], []
        plane_dims = [[0, 1], [0, 2], [1, 2]]
        for i, (c, p_dim) in enumerate(zip(n_components, plane_dims)):
            p_shape = (c, self.grid_dim, self.grid_dim)
            l_shape = (c, self.grid_dim, 1)
            p = jax.random.normal(keys[key_offset + i], p_shape) * 0.1
            l = jax.random.normal(keys[key_offset + i + 3], l_shape) * 0.1
            planes.append(p)
            lines.append(l)
        return tuple(planes), tuple(lines)

    def init_decoders(self, keys, app_dim):
        basis_matrix = jax.random.normal(keys[9], (app_dim, 27)) * 0.1
        mlp_render = eqx.nn.MLP(
            in_size=54, out_size=3, width_size=128, depth=2, key=keys[0]
        )
        return basis_matrix, mlp_render

    def normalize_coordinates(self, xyz):
        min_b = jax.lax.stop_gradient(self.bbox_min)
        max_b = jax.lax.stop_gradient(self.bbox_max)
        return jax.lax.stop_gradient((xyz - min_b) / (max_b - min_b))

    def interpolate_tensor_components(self, xyz_normed, planes, lines):
        grid_dim = self.grid_dim
        scaled_coords = jax.lax.stop_gradient(xyz_normed * (grid_dim - 1))
        x, y, z = scaled_coords[..., 0], scaled_coords[..., 1], scaled_coords[..., 2]

        def _corner_weights(coord):
            c0 = jnp.clip(jnp.floor(coord), 0, grid_dim - 1).astype(jnp.int32)
            c1 = jnp.clip(c0 + 1, 0, grid_dim - 1)
            w = jnp.clip(coord - c0, 0.0, 1.0)
            return c0, c1, w

        def bilinear_interp(plane, coord_u, coord_v):
            # plane: (C, grid_dim, grid_dim) -> single flat 1-D gather, no vmap
            C = plane.shape[0]
            u0, u1, wu = _corner_weights(coord_u)
            v0, v1, wv = _corner_weights(coord_v)
            plane_flat = plane.reshape(C, grid_dim * grid_dim)

            def corner(vi, ui):
                return jnp.take(plane_flat, vi * grid_dim + ui, axis=1)

            c0 = corner(v0, u0) * (1 - wu) + corner(v0, u1) * wu
            c1 = corner(v1, u0) * (1 - wu) + corner(v1, u1) * wu
            return c0 * (1 - wv) + c1 * wv

        def linear_interp(line, coord):
            C = line.shape[0]
            c0, c1, w = _corner_weights(coord)
            line_flat = line.reshape(C, grid_dim)
            return (
                jnp.take(line_flat, c0, axis=1) * (1 - w)
                + jnp.take(line_flat, c1, axis=1) * w
            )

        plane_xy = bilinear_interp(planes[0], x, y)
        line_z = linear_interp(lines[0], z)
        plane_xz = bilinear_interp(planes[1], x, z)
        line_y = linear_interp(lines[1], y)
        plane_yz = bilinear_interp(planes[2], y, z)
        line_x = linear_interp(lines[2], x)

        return [plane_xy * line_z, plane_xz * line_y, plane_yz * line_x]

    def get_sigma_feat(self, xyz_normed):
        # Let the grids remain in their native FP32 precision to safely accumulate tiny gradients
        den_planes = self.den_planes
        den_lines = self.den_lines
        app_planes = self.app_planes
        app_lines = self.app_lines

        den_components = self.interpolate_tensor_components(
            xyz_normed, den_planes, den_lines
        )
        sigma = sum(jnp.sum(comp, axis=0) for comp in den_components)

        # [FIX]: Use standard softplus without the extreme -10.0 bottleneck
        sigma = jax.nn.softplus(sigma)

        app_components = self.interpolate_tensor_components(
            xyz_normed, app_planes, app_lines
        )
        app_feats = jnp.concatenate(app_components, axis=0).T
        return sigma, app_feats

    def __call__(self, rays_o, rays_d, key, bg_color):
        n_samples = 192

        den_planes = self.den_planes
        den_lines = self.den_lines
        app_planes = self.app_planes
        app_lines = self.app_lines

        near, far, hit_mask = compute_ray_aabb_intersections(
            rays_o, rays_d, self.bbox_min, self.bbox_max
        )
        far = jnp.maximum(far, near + 1e-5)

        pts, z_vals = sample_along_rays(rays_o, rays_d, near, far, n_samples, key)

        pts_flat = pts.reshape(-1, 3)
        pts_norm = self.normalize_coordinates(pts_flat)
        valid_mask = jax.lax.stop_gradient(
            ((pts_norm >= 0.0) & (pts_norm <= 1.0)).all(axis=-1)
        )

        alpha_res = self.alpha_mask.shape[0]
        grid_idx = jnp.clip(
            jnp.floor(pts_norm * alpha_res).astype(jnp.int32), 0, alpha_res - 1
        )

        # --- FLATTENED TPU INDEXING ---
        flat_idx = (
            grid_idx[:, 0] * (alpha_res * alpha_res)
            + grid_idx[:, 1] * alpha_res
            + grid_idx[:, 2]
        )
        flat_mask = self.alpha_mask.flatten()
        in_alpha = jnp.take(flat_mask, flat_idx)
        # ------------------------------

        mask = valid_mask & in_alpha
        pts_norm = jnp.clip(pts_norm, 0.0, 1.0)

        den_components = self.interpolate_tensor_components(
            pts_norm, den_planes, den_lines
        )
        sigma = sum(jnp.sum(comp, axis=0) for comp in den_components)
        sigma = jax.nn.softplus(sigma)

        sigma = jnp.where(mask, sigma, 0.0)
        sigma = sigma.reshape(rays_o.shape[0], n_samples)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jnp.concatenate(
            [dists, jnp.broadcast_to(1e10, dists[..., :1].shape)], -1
        )
        dists = dists * jnp.linalg.norm(rays_d[..., None, :], axis=-1)

        app_components = self.interpolate_tensor_components(
            pts_norm, app_planes, app_lines
        )
        app_feats = jnp.concatenate(app_components, axis=0).T

        basis = self.basis_mat.astype(self.compute_dtype)
        app_feats_proj = app_feats @ basis

        view_dirs = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
        dirs_flat = view_dirs[:, None, :].repeat(n_samples, axis=1).reshape(-1, 3)
        dirs_enc = encode_view_directions(dirs_flat)
        dirs_enc = dirs_enc.astype(self.compute_dtype)

        mlp_input = jnp.concatenate([app_feats_proj, dirs_enc], axis=-1)
        rgb_flat_cast = jax.vmap(self.mlp_render)(mlp_input)

        rgb_flat = rgb_flat_cast.astype(jnp.float32)
        rgb_full = jax.nn.sigmoid(rgb_flat)
        rgb_full = rgb_full.reshape(rays_o.shape[0], n_samples, 3)

        return compute_volumetric_rendering(rgb_full, sigma, dists, z_vals, bg_color)


def update_alpha_mask(model):
    res = model.alpha_mask.shape[0]
    x = jnp.linspace(0, 1, res)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    pts_norm = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    @jax.jit
    def get_sigma(pts):
        sigma, _ = model.get_sigma_feat(pts)
        return sigma

    sigmas = []
    chunk = res * res
    for i in range(0, pts_norm.shape[0], chunk):
        pts_chunk = pts_norm[i : i + chunk]
        sigmas.append(np.array(get_sigma(pts_chunk)))

    sigma_grid = np.concatenate(sigmas, axis=0)

    extent = np.linalg.norm(np.array(model.bbox_max) - np.array(model.bbox_min))
    step_size = extent / 192.0

    alpha = 1.0 - np.exp(-sigma_grid * step_size)
    mask_flat = alpha > 0.0001
    mask = mask_flat.reshape(res, res, res)

    new_mask = jax.device_put(jnp.array(mask))
    return eqx.tree_at(lambda m: m.alpha_mask, model, new_mask)


def shrink_bbox(model):
    mask = np.array(model.alpha_mask)
    if not np.any(mask):
        return model

    indices = np.where(mask)
    min_idx = np.array([np.min(indices[0]), np.min(indices[1]), np.min(indices[2])])
    max_idx = np.array([np.max(indices[0]), np.max(indices[1]), np.max(indices[2])])

    res = mask.shape[0]
    max_idx = np.maximum(max_idx, min_idx + 1)

    min_norm = min_idx / (res - 1.0)
    max_norm = max_idx / (res - 1.0)

    old_min = np.array(model.bbox_min)
    old_max = np.array(model.bbox_max)
    extent = old_max - old_min

    new_bbox_min = old_min + min_norm * extent
    new_bbox_max = old_min + max_norm * extent

    voxel_size = (new_bbox_max - new_bbox_min) / res
    new_bbox_min -= voxel_size
    new_bbox_max += voxel_size

    grid_dim = model.grid_dim
    ix0, iy0, iz0 = np.floor(min_norm * (grid_dim - 1)).astype(int)
    ix1, iy1, iz1 = np.ceil(max_norm * (grid_dim - 1)).astype(int) + 1

    ix0, iy0, iz0 = max(0, ix0), max(0, iy0), max(0, iz0)
    ix1, iy1, iz1 = min(grid_dim, ix1), min(grid_dim, iy1), min(grid_dim, iz1)

    ix1 = max(ix1, ix0 + 1)
    iy1 = max(iy1, iy0 + 1)
    iz1 = max(iz1, iz0 + 1)

    def crop_planes(planes):
        p_xy = planes[0][:, iy0:iy1, ix0:ix1]
        p_xz = planes[1][:, iz0:iz1, ix0:ix1]
        p_yz = planes[2][:, iz0:iz1, iy0:iy1]
        return (jnp.array(p_xy), jnp.array(p_xz), jnp.array(p_yz))

    def crop_lines(lines):
        l_z = lines[0][:, iz0:iz1, :]
        l_y = lines[1][:, iy0:iy1, :]
        l_x = lines[2][:, ix0:ix1, :]
        return (jnp.array(l_z), jnp.array(l_y), jnp.array(l_x))

    new_den_planes = crop_planes([np.array(p) for p in model.den_planes])
    new_den_lines = crop_lines([np.array(l) for l in model.den_lines])
    new_app_planes = crop_planes([np.array(p) for p in model.app_planes])
    new_app_lines = crop_lines([np.array(l) for l in model.app_lines])

    model = eqx.tree_at(
        lambda m: m.bbox_min, model, jax.device_put(jnp.array(new_bbox_min))
    )
    model = eqx.tree_at(
        lambda m: m.bbox_max, model, jax.device_put(jnp.array(new_bbox_max))
    )

    model = eqx.tree_at(
        lambda m: [m.den_planes, m.den_lines, m.app_planes, m.app_lines],
        model,
        [new_den_planes, new_den_lines, new_app_planes, new_app_lines],
    )
    return model


def upsample_tensoRF(old_model, new_grid_dim, key):
    new_model = TensoRF(key, grid_dim=new_grid_dim)

    def resize_components(old_comps, new_res, is_line=False):
        new_comps = []
        for old_c in old_comps:
            if is_line:
                target_shape = (old_c.shape[0], new_res, 1)
            else:
                target_shape = (old_c.shape[0], new_res, new_res)
            new_c = jax.image.resize(old_c, target_shape, method="linear")
            new_comps.append(new_c)
        return tuple(new_comps)

    new_den_planes = resize_components(
        old_model.den_planes, new_grid_dim, is_line=False
    )
    new_den_lines = resize_components(old_model.den_lines, new_grid_dim, is_line=True)
    new_app_planes = resize_components(
        old_model.app_planes, new_grid_dim, is_line=False
    )
    new_app_lines = resize_components(old_model.app_lines, new_grid_dim, is_line=True)

    new_model = eqx.tree_at(
        lambda m: [
            m.den_planes,
            m.den_lines,
            m.app_planes,
            m.app_lines,
            m.mlp_render,
            m.basis_mat,
            m.bbox_min,
            m.bbox_max,
            m.alpha_mask,
        ],
        new_model,
        [
            new_den_planes,
            new_den_lines,
            new_app_planes,
            new_app_lines,
            old_model.mlp_render,
            old_model.basis_mat,
            old_model.bbox_min,
            old_model.bbox_max,
            old_model.alpha_mask,
        ],
    )
    return new_model
