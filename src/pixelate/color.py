"""sRGB <-> CIELAB conversion (D65 white point) in plain NumPy."""

from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 255] (any shape ``(..., 3)``) to Lab."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA**2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert Lab values (any shape ``(..., 3)``) to sRGB uint8."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4.0 / 29.0)) * _WHITE
    lin = np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)
