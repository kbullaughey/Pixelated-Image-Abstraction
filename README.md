# Pixelated Image Abstraction

Turns an image into pixel art using the algorithm from Gerstner et al.,
[*Pixelated Image Abstraction*](https://gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA/) (NPAR 2012).

It works out where the output pixels go and which palette they use together, not one after the other:

- **Superpixels.** Each output pixel is a superpixel, found with a SLIC variant
  that keeps the superpixels on a regular grid. Pixel boundaries therefore follow
  edges in the image.
- **Palette.** The palette is built by mass-constrained deterministic annealing.
  It starts as a single colour and splits colours as the "temperature" drops,
  until it reaches the requested palette size.

A downscale followed by colour quantisation usually gives muddier results.

This is plain NumPy + Pillow. No compilation step is needed.

## Install

With [uv](https://docs.astral.sh/uv/):

```sh
uv sync                  # creates .venv with the pixelate command
uv run pixelate --help
```

To put `pixelate` (and `pixelate-gui`) on your PATH globally:

```sh
uv tool install .              # snapshot of the current code
uv tool install --editable .   # or: always run the code in this checkout
```

Or with pip: `pip install .`, which installs the `pixelate` and `pixelate-gui` commands.

## Command line

```sh
# 120x160 output, at most 16 colours, also save a 4x enlarged copy for viewing
uv run pixelate imgs/test-sprite.png -s 120x160 -c 16 -o sprite.png
uv run pixelate imgs/test-sprite.png -s 120x160 -c 16 --scale 4 -o sprite@4x.png

# give only a width (or height) to keep the aspect ratio
uv run pixelate imgs/in.jpg -W 64 -c 8

# save the palette as a swatch strip too
uv run pixelate imgs/in.jpg -W 64 --palette palette.png

# naive baselines for comparison: downscale + median-cut quantisation
uv run pixelate imgs/in.jpg -W 64 -m box
```

| Option | Meaning |
| --- | --- |
| `-o, --output FILE` | Output path. Default: `<input>_pixelated_<W>x<H>.png` next to the input. |
| `-s, --size WxH` | Output size in pixels. |
| `-W, --width` / `-H, --height` | Give one to keep the aspect ratio, or both to set the size exactly. Default: width 64. |
| `-c, --colors N` | Maximum palette size (default 8). |
| `--scale N` | Enlarge the saved image N× with nearest neighbour, which makes it easy to view. |
| `-m, --method` | `paper` (default), or `nearest` / `box` baselines. |
| `--palette FILE` | Also write the palette as an image. |
| `--work-scale N` | Working pixels per output pixel, per side (default 6). Lower is faster. Higher follows edges more closely. |
| `-q, --quiet` | No progress output. |

Other tuning flags are `--compactness`, `--smoothing`, `--sigma-space`,
`--sigma-color`, `--saturation`, `--alpha` and `--max-iter`. See `pixelate --help`.

Transparent input (such as RGBA sprites) is supported. Each output pixel is
opaque or fully transparent, depending on the coverage of its superpixel.

## GUI

```sh
uv run pixelate-gui [IMAGE]
```

If Tk reports `Can't find a usable init.tcl`, your uv-managed Python predates a
Tcl packaging fix. Run `uv python upgrade` and then `uv sync --reinstall`.

## Python API

```python
from PIL import Image
from pixelate import Params, pixelate

result = pixelate(Image.open("in.jpg"), 64, 48, Params(colors=12))
result.image.save("out.png")
print(result.palette)  # (K, 3) uint8 RGB
```

## Tests

```sh
uv run pytest
```
