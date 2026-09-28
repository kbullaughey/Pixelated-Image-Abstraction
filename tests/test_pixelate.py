import numpy as np
import pytest
from PIL import Image

from pixelate import Params, pixelate
from pixelate.cli import main
from pixelate.color import lab_to_rgb, rgb_to_lab


def _quadrants(w=96, h=64, alpha=False) -> Image.Image:
    """Four flat colour blocks; an easy image with a known ideal answer."""
    a = np.zeros((h, w, 4 if alpha else 3), dtype=np.uint8)
    a[: h // 2, : w // 2, :3] = (220, 30, 30)
    a[: h // 2, w // 2 :, :3] = (30, 200, 40)
    a[h // 2 :, : w // 2, :3] = (20, 40, 210)
    a[h // 2 :, w // 2 :, :3] = (240, 240, 240)
    if alpha:
        a[..., 3] = 255
        a[h // 2 :, w // 2 :, 3] = 0
    return Image.fromarray(a)


def test_lab_round_trip():
    rgb = np.random.default_rng(0).integers(0, 256, size=(500, 3))
    assert np.abs(lab_to_rgb(rgb_to_lab(rgb)).astype(int) - rgb).max() <= 1


def test_lab_reference_values():
    np.testing.assert_allclose(rgb_to_lab([255, 255, 255]), [100, 0, 0], atol=0.01)
    np.testing.assert_allclose(rgb_to_lab([255, 0, 0]), [53.24, 80.09, 67.20], atol=0.05)


def test_non_square_output_recovers_flat_colours():
    result = pixelate(_quadrants(), 12, 8, Params(colors=4, saturation=1.0))
    assert result.image.size == (12, 8)
    assert result.image.mode == "RGB"
    assert len(result.palette) == 4
    out = np.asarray(result.image).astype(int)
    for (y, x), expected in {(0, 0): (220, 30, 30), (0, 11): (30, 200, 40),
                             (7, 0): (20, 40, 210), (7, 11): (240, 240, 240)}.items():
        assert np.abs(out[y, x] - expected).max() <= 3


def test_palette_size_is_a_maximum():
    result = pixelate(_quadrants(), 12, 8, Params(colors=2))
    assert len(np.unique(np.asarray(result.image).reshape(-1, 3), axis=0)) <= 2


def test_alpha_is_preserved():
    result = pixelate(_quadrants(alpha=True), 12, 8, Params(colors=4))
    assert result.image.mode == "RGBA"
    a = np.asarray(result.image)[..., 3]
    assert a[0, 0] == 255 and a[7, 11] == 0


def test_rejects_bad_size():
    with pytest.raises(ValueError):
        pixelate(_quadrants(), 0, 8)


def test_cli_writes_output(tmp_path):
    src = tmp_path / "in.png"
    _quadrants().save(src)
    out = tmp_path / "sub" / "out.png"
    pal = tmp_path / "pal.png"
    assert main([str(src), "-o", str(out), "-W", "12", "-c", "4", "--scale", "3", "--palette", str(pal), "-q"]) == 0
    assert Image.open(out).size == (36, 24)  # height follows the 3:2 aspect ratio, then x3
    assert Image.open(pal).size == (4 * 32, 32)


def test_cli_default_output_name(tmp_path):
    src = tmp_path / "pic.png"
    _quadrants().save(src)
    assert main([str(src), "-s", "6x4", "-m", "nearest", "-q"]) == 0
    assert Image.open(tmp_path / "pic_pixelated_6x4.png").size == (6, 4)


def test_cli_rejects_size_and_width(tmp_path, capsys):
    src = tmp_path / "pic.png"
    _quadrants().save(src)
    with pytest.raises(SystemExit):
        main([str(src), "-s", "6x4", "-W", "6"])
    assert "not both" in capsys.readouterr().err
