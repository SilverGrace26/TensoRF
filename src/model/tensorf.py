import jax
import jax.numpy as jnp
import jax.scipy
import equinox as eqx
from typing import Tuple

from geometry.rays import sample_along_rays, encode_view_directions
from geometry.rendering import compute_volumetric_rendering


class TensoRF(eqx.Module):
    grid_dim: int = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    bbox_min: jax.Array
    bbox_max: jax.Array
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

        x = scaled_coords[..., 0]
        y = scaled_coords[..., 1]
        z = scaled_coords[..., 2]

        def bilinear_interp(plane, coord_u, coord_v):
            u0 = jnp.floor(coord_u).astype(jnp.int32)
            v0 = jnp.floor(coord_v).astype(jnp.int32)
            u1 = u0 + 1
            v1 = v0 + 1

            u0 = jnp.clip(u0, 0, grid_dim - 1)
            v0 = jnp.clip(v0, 0, grid_dim - 1)
            u1 = jnp.clip(u1, 0, grid_dim - 1)
            v1 = jnp.clip(v1, 0, grid_dim - 1)

            plane = jnp.moveaxis(plane, 0, -1)
            c00 = plane[v0, u0]
            c01 = plane[v0, u1]
            c10 = plane[v1, u0]
            c11 = plane[v1, u1]

            wu = jnp.expand_dims(coord_u - u0, axis=-1)
            wv = jnp.expand_dims(coord_v - v0, axis=-1)

            c0 = c00 * (1 - wu) + c01 * wu
            c1 = c10 * (1 - wu) + c11 * wu
            c = c0 * (1 - wv) + c1 * wv

            return jnp.moveaxis(c, -1, 0)

        def linear_interp(line, coord):
            u0 = jnp.floor(coord).astype(jnp.int32)
            u1 = u0 + 1

            u0 = jnp.clip(u0, 0, grid_dim - 1)
            u1 = jnp.clip(u1, 0, grid_dim - 1)

            line = jnp.moveaxis(line, 0, -1)
            c0 = line[u0, 0]
            c1 = line[u1, 0]

            wu = jnp.expand_dims(coord - u0, axis=-1)
            c = c0 * (1 - wu) + c1 * wu

            return jnp.moveaxis(c, -1, 0)

        plane_xy = bilinear_interp(planes[0], x, y)
        line_z = linear_interp(lines[0], z)

        plane_xz = bilinear_interp(planes[1], x, z)
        line_y = linear_interp(lines[1], y)

        plane_yz = bilinear_interp(planes[2], y, z)
        line_x = linear_interp(lines[2], x)

        results = [plane_xy * line_z, plane_xz * line_y, plane_yz * line_x]

        return results

    def get_sigma_feat(self, xyz_normed):
        den_planes = tuple(x.astype(self.compute_dtype) for x in self.den_planes)
        den_lines = tuple(x.astype(self.compute_dtype) for x in self.den_lines)
        app_planes = tuple(x.astype(self.compute_dtype) for x in self.app_planes)
        app_lines = tuple(x.astype(self.compute_dtype) for x in self.app_lines)

        den_components = self.interpolate_tensor_components(
            xyz_normed, den_planes, den_lines
        )
        sigma = sum(jnp.sum(comp, axis=0) for comp in den_components)
        sigma = jax.nn.softplus(sigma) * 5.0

        app_components = self.interpolate_tensor_components(
            xyz_normed, app_planes, app_lines
        )
        app_feats = jnp.concatenate(app_components, axis=0).T
        return sigma, app_feats

    def __call__(self, rays_o, rays_d, key, bg_color):
        n_samples = 192
        n_important = 48

        den_planes = tuple(x.astype(self.compute_dtype) for x in self.den_planes)
        den_lines = tuple(x.astype(self.compute_dtype) for x in self.den_lines)
        app_planes = tuple(x.astype(self.compute_dtype) for x in self.app_planes)
        app_lines = tuple(x.astype(self.compute_dtype) for x in self.app_lines)

        pts, z_vals = sample_along_rays(rays_o, rays_d, n_samples, key)

        pts_flat = pts.reshape(-1, 3)
        pts_norm = self.normalize_coordinates(pts_flat)
        mask = jax.lax.stop_gradient(((pts_norm > 0.0) & (pts_norm < 1.0)).all(axis=-1))
        pts_norm = jnp.clip(pts_norm, 0.0, 1.0)

        den_components = self.interpolate_tensor_components(
            pts_norm, den_planes, den_lines
        )
        sigma = sum(jnp.sum(comp, axis=0) for comp in den_components)
        sigma = jax.nn.softplus(sigma) * 5.0

        sigma = sigma * mask
        sigma = sigma.reshape(rays_o.shape[0], n_samples)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jnp.concatenate(
            [dists, jnp.broadcast_to(1e10, dists[..., :1].shape)], -1
        )
        dists = dists * jnp.linalg.norm(rays_d[..., None, :], axis=-1)

        alpha = 1.0 - jnp.exp(-sigma * dists)
        transmittance = jnp.cumprod(1.0 - alpha + 1e-10, axis=-1)
        weights = alpha * jnp.concatenate(
            [jnp.ones((alpha.shape[0], 1)), transmittance[..., :-1]], -1
        )

        _, top_indices = jax.lax.top_k(weights, n_important)
        top_indices = jnp.sort(top_indices, axis=-1)

        batch_indices = jnp.arange(rays_o.shape[0])[:, None]
        z_vals_imp = z_vals[batch_indices, top_indices]
        pts_imp = pts[batch_indices, top_indices, :]
        sigma_imp = sigma[batch_indices, top_indices]
        dists_imp = dists[batch_indices, top_indices]

        pts_imp_flat = pts_imp.reshape(-1, 3)
        pts_imp_norm = self.normalize_coordinates(pts_imp_flat)
        pts_imp_norm = jnp.clip(pts_imp_norm, 0.0, 1.0)

        app_components = self.interpolate_tensor_components(
            pts_imp_norm, app_planes, app_lines
        )
        app_feats = jnp.concatenate(app_components, axis=0).T

        basis = self.basis_mat.astype(self.compute_dtype)
        app_feats_proj = app_feats @ basis

        view_dirs = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
        dirs_flat = view_dirs[:, None, :].repeat(n_important, axis=1).reshape(-1, 3)
        dirs_enc = encode_view_directions(dirs_flat)
        dirs_enc = dirs_enc.astype(self.compute_dtype)

        mlp_input = jnp.concatenate([app_feats_proj, dirs_enc], axis=-1)
        rgb_flat_cast = jax.vmap(self.mlp_render)(mlp_input)

        rgb_flat = rgb_flat_cast.astype(jnp.float32)
        rgb_imp = jax.nn.sigmoid(rgb_flat)
        rgb_imp = rgb_imp.reshape(rays_o.shape[0], n_important, 3)

        return compute_volumetric_rendering(
            rgb_imp, sigma_imp, dists_imp, z_vals_imp, bg_color
        )


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
        ],
        new_model,
        [
            new_den_planes,
            new_den_lines,
            new_app_planes,
            new_app_lines,
            old_model.mlp_render,
            old_model.basis_mat,
        ],
    )
    return new_model
