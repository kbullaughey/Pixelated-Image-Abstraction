"""Pixelated image abstraction.

Implementation of Gerstner et al., "Pixelated Image Abstraction" (NPAR 2012).
The output pixels are superpixels (a SLIC variant constrained to a regular
grid) whose colours are chosen from a palette that is built with mass-
constrained deterministic annealing (MCDA), so the palette and the pixel
assignment are optimised together.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .color import lab_to_rgb, rgb_to_lab


@dataclass
class Params:
    """Tunable parameters. Defaults follow the paper where it gives values."""

    colors: int = 8
    """Maximum number of colours in the output palette."""
    compactness: float = 45.0
    """SLIC weight of spatial distance vs. colour distance (the paper's m)."""
    alpha: float = 0.7
    """Temperature decay factor applied each time the palette converges."""
    t_final: float = 1.0
    """Annealing stops once the temperature drops below this."""
    eps_palette: float = 1.0
    """Palette is converged when the total colour change drops below this."""
    eps_cluster: float = 0.25
    """Sub-cluster separation (Lab units) above which a colour is split."""
    perturbation: float = 0.1
    """Size (Lab units) of the nudge used to seed sub-cluster pairs."""
    smoothing: float = 0.4
    """Laplacian smoothing weight pulling superpixel centres to their grid neighbours."""
    sigma_space: float = 0.87
    """Bilateral filter spatial sigma, in output pixels (0 disables the filter)."""
    sigma_color: float = 8.0
    """Bilateral filter range sigma, in Lab units (0 disables the filter)."""
    saturation: float = 1.1
    """Final multiplier on Lab a/b channels, as in the paper's post-processing."""
    work_scale: int = 6
    """Maximum working-image pixels per output pixel (per side). Lower is faster."""
    max_iter: int = 1000
    """Hard cap on the number of iterations."""


@dataclass
class Progress:
    iteration: int
    temperature: float
    palette_size: int
    """Number of distinct colours currently in the palette."""


@dataclass
class Result:
    image: Image.Image
    """Output image at the requested size (RGB, or RGBA if the input had alpha)."""
    palette: np.ndarray
    """``(K, 3)`` uint8 RGB colours used in the output."""
    labels: np.ndarray
    """Superpixel index of each working-image pixel, shape ``(H, W)``."""
    iterations: int
    extra: dict = field(default_factory=dict)


def pixelate(
    image: Image.Image,
    width: int,
    height: int,
    params: Params | None = None,
    progress: Callable[[Progress], None] | None = None,
) -> Result:
    """Abstract ``image`` into a ``width`` x ``height`` pixel-art image."""
    p = params or Params()
    if width < 1 or height < 1:
        raise ValueError("output size must be positive")
    if p.colors < 1:
        raise ValueError("colors must be at least 1")

    # Work on a copy resized so each output pixel covers exactly S x S input pixels.
    S = max(1, min(p.work_scale, round(min(image.width / width, image.height / height))))
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    work = image.convert("RGBA").resize((width * S, height * S), Image.Resampling.LANCZOS)
    rgba = np.asarray(work, dtype=np.float64)
    alpha = rgba[..., 3] / 255.0
    lab = rgb_to_lab(rgba[..., :3] * alpha[..., None])  # composite onto black
    H, W = lab.shape[:2]
    N = width * height

    pix = lab.reshape(-1, 3)
    ys, xs = np.divmod(np.arange(H * W, dtype=np.float64), W)
    pos = np.stack([xs, ys], axis=1)

    # Views of the working image as (grid row, row in cell, grid col, col in cell),
    # so values per superpixel broadcast against every pixel of its grid cell.
    blocks = (height, S, width, S)
    pix_b = [np.ascontiguousarray(lab[..., k], dtype=np.float32).reshape(blocks) for k in range(3)]
    xs_b = xs.astype(np.float32).reshape(blocks)
    ys_b = ys.astype(np.float32).reshape(blocks)
    grid_idx = np.pad(np.arange(N).reshape(height, width), 1, constant_values=-1)

    gy, gx = np.divmod(np.arange(N), width)
    centers = np.stack([gx * S + (S - 1) / 2, gy * S + (S - 1) / 2], axis=1).astype(np.float64)
    labels = np.broadcast_to(np.arange(N).reshape(height, 1, width, 1), blocks).ravel().copy()
    sp_colors = _mean_by_label(pix, labels, N, fallback=np.zeros((N, 3)))
    spatial_weight = p.compactness / S

    # Palette starts as one colour: the mean, held as a pair of sub-clusters
    # nudged apart along the principal axis so it can split when T drops.
    filtered = _bilateral(sp_colors.reshape(height, width, 3), p).reshape(N, 3)
    axis, variance = _principal_axis(filtered, np.full(N, 1.0 / N))
    mean = filtered.mean(axis=0)
    palette = np.array([mean, mean + p.perturbation * axis])
    prob_c = np.array([0.5, 0.5])
    clusters: list[list[int]] = [[0, 1]]
    paired = p.colors > 1
    if not paired:
        palette, prob_c, clusters = palette[:1], prob_c[:1] * 2, [[0]]

    T = 1.1 * 2.0 * variance  # the paper's T0 = 1.1 * Tc, where Tc = 2 * lambda_max
    iteration = 0
    assoc = np.full((N, len(palette)), 1.0 / len(palette))

    while iteration < p.max_iter:
        iteration += 1

        # 1. Refine superpixels: SLIC where each pixel may join the superpixel of
        # its own grid cell or of any of the 8 cells around it.
        labels = _assign_pixels(pix_b, xs_b, ys_b, sp_colors, centers, grid_idx, spatial_weight)
        centers = _mean_by_label(pos, labels, N, fallback=centers)
        centers = _laplacian_smooth(centers.reshape(height, width, 2), p.smoothing).reshape(N, 2)
        sp_colors = _mean_by_label(pix, labels, N, fallback=sp_colors)
        filtered = _bilateral(sp_colors.reshape(height, width, 3), p).reshape(N, 3)

        # 2. Associate superpixels with palette colours (soft, at temperature T).
        dist2 = ((filtered[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        logits = np.log(np.maximum(prob_c, 1e-300))[None, :] - dist2 / T
        logits -= logits.max(axis=1, keepdims=True)
        assoc = np.exp(logits)
        assoc /= assoc.sum(axis=1, keepdims=True)

        # 3. Refine palette colours.
        prob_c = assoc.mean(axis=0)
        weighted = assoc.T @ filtered / N
        new_palette = palette.copy()
        live = prob_c > 1e-12
        new_palette[live] = weighted[live] / prob_c[live, None]
        change = np.linalg.norm(new_palette - palette, axis=1).sum()
        palette = new_palette

        if progress:
            progress(Progress(iteration, T, len(clusters)))

        # 4. On convergence: cool down, then split any sub-cluster pairs that separated.
        if change < p.eps_palette:
            if T <= p.t_final:
                break
            T = max(T * p.alpha, p.t_final)
            if paired:
                palette, prob_c, clusters, paired = _expand(
                    palette, prob_c, clusters, assoc, filtered, p
                )

    # Final hard assignment: each superpixel takes its most probable colour.
    cluster_colors = np.array([_cluster_color(palette, prob_c, c) for c in clusters])
    cluster_assoc = np.stack([assoc[:, c].sum(axis=1) for c in clusters], axis=1)
    choice = cluster_assoc.argmax(axis=1)
    out_lab = cluster_colors[choice]
    out_lab[:, 1:] *= p.saturation
    out_rgb = lab_to_rgb(out_lab)

    used = np.unique(choice)
    boosted = cluster_colors[used].copy()
    boosted[:, 1:] *= p.saturation
    result_palette = lab_to_rgb(boosted)

    out = out_rgb.reshape(height, width, 3)
    if has_alpha:
        sp_alpha = _mean_by_label(alpha.reshape(-1, 1), labels, N, fallback=np.zeros((N, 1)))
        mask = np.where(sp_alpha[:, 0] >= 0.5, 255, 0).astype(np.uint8)
        out = np.dstack([out, mask.reshape(height, width)])
        img = Image.fromarray(out, "RGBA")
    else:
        img = Image.fromarray(out, "RGB")

    return Result(
        image=img,
        palette=result_palette,
        labels=labels.reshape(H, W),
        iterations=iteration,
        extra={"work_scale": S, "final_temperature": T},
    )


def _assign_pixels(pix_b, xs_b, ys_b, sp_colors, centers, grid_idx, spatial_weight) -> np.ndarray:
    """Label each working pixel with its nearest superpixel among the 3x3 neighbouring cells."""
    height, S, width, _ = xs_b.shape
    far = np.float32(1e9)  # off-grid neighbours get a centre far away, so they never win

    def padded(values: np.ndarray, fill: float) -> np.ndarray:
        return np.pad(values.reshape(height, width).astype(np.float32), 1, constant_values=fill)

    cols = [padded(sp_colors[:, k], 0) for k in range(3)]
    cx, cy = padded(centers[:, 0], far), padded(centers[:, 1], far)

    best = np.full(xs_b.shape, np.inf, dtype=np.float32)
    labels = np.zeros(xs_b.shape, dtype=np.int64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            win = (slice(dy, dy + height), slice(dx, dx + width))

            def at(grid):
                return grid[win].reshape(height, 1, width, 1)

            dc = (pix_b[0] - at(cols[0])) ** 2
            dc += (pix_b[1] - at(cols[1])) ** 2
            dc += (pix_b[2] - at(cols[2])) ** 2
            dp = (xs_b - at(cx)) ** 2
            dp += (ys_b - at(cy)) ** 2
            d = np.sqrt(dc)
            d += spatial_weight * np.sqrt(dp)
            better = d < best
            best = np.where(better, d, best)
            labels = np.where(better, at(grid_idx), labels)
    return labels.ravel()


def _mean_by_label(values: np.ndarray, labels: np.ndarray, n: int, fallback: np.ndarray) -> np.ndarray:
    """Per-label mean of ``values`` rows; labels with no members keep ``fallback``."""
    counts = np.bincount(labels, minlength=n)
    sums = np.stack([np.bincount(labels, values[:, k], minlength=n) for k in range(values.shape[1])], axis=1)
    out = fallback.copy()
    nonempty = counts > 0
    out[nonempty] = sums[nonempty] / counts[nonempty, None]
    return out


def _laplacian_smooth(grid: np.ndarray, weight: float) -> np.ndarray:
    """Move each superpixel centre towards the mean of its 4-connected grid neighbours."""
    if weight <= 0:
        return grid
    total = np.zeros_like(grid)
    count = np.zeros(grid.shape[:2] + (1,))
    total[1:] += grid[:-1]
    count[1:] += 1
    total[:-1] += grid[1:]
    count[:-1] += 1
    total[:, 1:] += grid[:, :-1]
    count[:, 1:] += 1
    total[:, :-1] += grid[:, 1:]
    count[:, :-1] += 1
    has = count > 0
    neighbours = np.where(has, total / np.maximum(count, 1), grid)
    return (1 - weight) * grid + weight * neighbours


def _bilateral(grid: np.ndarray, p: Params) -> np.ndarray:
    """Edge-preserving smoothing of superpixel colours over the output grid."""
    if p.sigma_space <= 0 or p.sigma_color <= 0:
        return grid
    r = max(1, math.ceil(2 * p.sigma_space))
    h, w = grid.shape[:2]
    padded = np.pad(grid, ((r, r), (r, r), (0, 0)), mode="edge")
    inside = np.pad(np.ones((h, w)), r)
    acc = np.zeros_like(grid)
    norm = np.zeros((h, w))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = padded[r + dy : r + dy + h, r + dx : r + dx + w]
            valid = inside[r + dy : r + dy + h, r + dx : r + dx + w]
            d2 = ((shifted - grid) ** 2).sum(axis=2)
            wgt = valid * np.exp(-(dx * dx + dy * dy) / (2 * p.sigma_space**2) - d2 / (2 * p.sigma_color**2))
            acc += wgt[..., None] * shifted
            norm += wgt
    return acc / norm[..., None]


def _principal_axis(colors: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Largest-variance direction (and its variance) of weighted colours."""
    total = weights.sum()
    if total <= 0:
        return np.array([1.0, 0.0, 0.0]), 0.0
    mean = weights @ colors / total
    centred = colors - mean
    cov = (centred * weights[:, None]).T @ centred / total
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, -1], float(max(vals[-1], 0.0))


def _cluster_color(palette: np.ndarray, prob_c: np.ndarray, members: list[int]) -> np.ndarray:
    w = prob_c[members]
    if w.sum() <= 0:
        return palette[members].mean(axis=0)
    return w @ palette[members] / w.sum()


def _expand(palette, prob_c, clusters, assoc, colors, p: Params):
    """Split converged sub-cluster pairs that have separated into new colours.

    Pairs that did not separate are re-perturbed so they can split at a lower
    temperature. Once the palette reaches ``p.colors`` the pairs are merged
    into single colours and annealing continues without further splitting.
    """
    palette = list(palette)
    prob_c = list(prob_c)
    out: list[list[int]] = []

    def seed_pair(i: int) -> int:
        """Add a new sub-cluster next to ``i``, sharing its probability mass."""
        axis, _ = _principal_axis(colors, assoc[:, i])
        palette.append(palette[i] + p.perturbation * axis)
        prob_c[i] /= 2
        prob_c.append(prob_c[i])
        return len(palette) - 1

    n_clusters = len(clusters)
    for a, b in clusters:
        if n_clusters < p.colors and np.linalg.norm(palette[a] - palette[b]) > p.eps_cluster:
            out.append([a, seed_pair(a)])
            out.append([b, seed_pair(b)])
            n_clusters += 1
        else:
            axis, _ = _principal_axis(colors, assoc[:, a] + assoc[:, b])
            palette[b] = palette[a] + p.perturbation * axis
            out.append([a, b])

    palette = np.array(palette)
    prob_c = np.array(prob_c)
    if len(out) < p.colors:
        return palette, prob_c, out, True

    # Palette is full: condense each pair into a single colour.
    merged = np.array([_cluster_color(palette, prob_c, c) for c in out])
    merged_p = np.array([prob_c[c].sum() for c in out])
    return merged, merged_p, [[k] for k in range(len(out))], False


def resize_baseline(
    image: Image.Image, width: int, height: int, colors: int | None, method: str
) -> Image.Image:
    """Naive downscale (+ optional median-cut quantisation) for comparison."""
    resample = {"nearest": Image.Resampling.NEAREST, "box": Image.Resampling.BOX}[method]
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    small = image.convert("RGBA" if has_alpha else "RGB").resize((width, height), resample)
    if colors:
        rgb = small.convert("RGB").quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        rgb = rgb.convert("RGB")
        if has_alpha:
            rgb.putalpha(small.getchannel("A"))
        small = rgb
    return small
