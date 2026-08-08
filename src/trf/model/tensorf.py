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
        self.compute_dtype = jnp.bfloat16 if backend == "tpu" else jnp.float32

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
        return (xyz - min_b) / (max_b - min_b)

    def interpolate_tensor_components(self, xyz_normed, planes, lines):
        grid_dim = self.grid_dim
        # Scale coordinates to grid indices [0, grid_dim - 1]
        scaled_coords = xyz_normed * (grid_dim - 1)

        # Coordinate extraction for planes (X, Y, Z)
        # Note: XLA handles these explicit extractions much better than stacked arrays
        x = scaled_coords[..., 0]
        y = scaled_coords[..., 1]
        z = scaled_coords[..., 2]

        # Helper function for fast Bilinear Interpolation
        def bilinear_interp(plane, coord_u, coord_v):
            # plane shape: (C, H, W)
            u0 = jnp.floor(coord_u).astype(jnp.int32)
            v0 = jnp.floor(coord_v).astype(jnp.int32)
            u1 = u0 + 1
            v1 = v0 + 1

            # Clip to grid boundaries
            u0 = jnp.clip(u0, 0, grid_dim - 1)
            v0 = jnp.clip(v0, 0, grid_dim - 1)
            u1 = jnp.clip(u1, 0, grid_dim - 1)
            v1 = jnp.clip(v1, 0, grid_dim - 1)

            # Gather corner values (vectorized across all C channels automatically)
            # We use jnp.moveaxis to bring C to the end for easy broadcasting, then move it back
            plane = jnp.moveaxis(plane, 0, -1)
            c00 = plane[v0, u0]
            c01 = plane[v0, u1]
            c10 = plane[v1, u0]
            c11 = plane[v1, u1]

            # Interpolation weights
            wu = jnp.expand_dims(coord_u - u0, axis=-1)
            wv = jnp.expand_dims(coord_v - v0, axis=-1)

            # Compute bilinear combination
            c0 = c00 * (1 - wu) + c01 * wu
            c1 = c10 * (1 - wu) + c11 * wu
            c = c0 * (1 - wv) + c1 * wv

            return jnp.moveaxis(c, -1, 0)  # Return to (C, N_rays)

        # Helper function for fast Linear Interpolation
        def linear_interp(line, coord):
            # line shape: (C, H, 1)
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

        # Apply interpolations based on the TensoRF projection logic
        # XY plane & Z line
        plane_xy = bilinear_interp(planes[0], x, y)
        line_z = linear_interp(lines[0], z)

        # XZ plane & Y line
        plane_xz = bilinear_interp(planes[1], x, z)
        line_y = linear_interp(lines[1], y)

        # YZ plane & X line
        plane_yz = bilinear_interp(planes[2], y, z)
        line_x = linear_interp(lines[2], x)

        results = [plane_xy * line_z, plane_xz * line_y, plane_yz * line_x]

        return results

    def get_sigma_feat(self, xyz_normed):
        den_components = self.interpolate_tensor_components(
            xyz_normed, self.den_planes, self.den_lines
        )
        sigma = sum(jnp.sum(comp, axis=0) for comp in den_components)
        sigma = jax.nn.softplus(sigma) * 5.0

        app_components = self.interpolate_tensor_components(
            xyz_normed, self.app_planes, self.app_lines
        )
        app_feats = jnp.concatenate(app_components, axis=0).T
        return sigma, app_feats

    def __call__(self, rays_o, rays_d, key, bg_color):
        n_samples = 192
        pts, z_vals = sample_along_rays(rays_o, rays_d, n_samples, key)

        pts_flat = pts.reshape(-1, 3)
        pts_norm = self.normalize_coordinates(pts_flat)
        mask = ((pts_norm > 0.0) & (pts_norm < 1.0)).all(axis=-1)
        pts_norm = jnp.clip(pts_norm, 0.0, 1.0)

        sigma, app_feats = self.get_sigma_feat(pts_norm)
        sigma = sigma * mask

        app_feats_proj = app_feats @ self.basis_mat

        view_dirs = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
        dirs_flat = view_dirs[:, None, :].repeat(n_samples, axis=1).reshape(-1, 3)

        dirs_enc = encode_view_directions(dirs_flat)
        mlp_input = jnp.concatenate([app_feats_proj, dirs_enc], axis=-1)

        mlp_input_cast = mlp_input.astype(self.compute_dtype)
        rgb_flat_cast = jax.vmap(self.mlp_render)(mlp_input_cast)

        rgb_flat = rgb_flat_cast.astype(jnp.float32)
        rgb = jax.nn.sigmoid(rgb_flat)

        sigma = sigma.reshape(rays_o.shape[0], n_samples)
        rgb = rgb.reshape(rays_o.shape[0], n_samples, 3)

        return compute_volumetric_rendering(rgb, sigma, z_vals, rays_d, bg_color)


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
