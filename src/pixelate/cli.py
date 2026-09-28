"""Command-line interface: ``pixelate INPUT -o OUTPUT --size 64x64``."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from PIL import Image

from .abstraction import Params, Progress, pixelate, resize_baseline


def _parse_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX*,]\s*(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT (e.g. 64x48), got {text!r}")
    w, h = int(m[1]), int(m[2])
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return w, h


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    defaults = Params()
    parser = argparse.ArgumentParser(
        prog="pixelate",
        description="Turn an image into pixel art (Gerstner et al., 'Pixelated Image Abstraction').",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input image")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output file (default: INPUT_pixelated_WxH.png next to the input)",
    )

    size = parser.add_argument_group("output size (give --size, or one of --width/--height to keep aspect ratio)")
    size.add_argument("-s", "--size", type=_parse_size, metavar="WxH", help="output size in pixels, e.g. 64x48")
    size.add_argument("-W", "--width", type=_positive_int, help="output width in pixels")
    size.add_argument("-H", "--height", type=_positive_int, help="output height in pixels")
    size.add_argument(
        "--scale", type=_positive_int, default=1,
        help="upscale the saved image by this factor (nearest neighbour) so it is easy to view",
    )

    parser.add_argument("-c", "--colors", type=_positive_int, default=defaults.colors, help="maximum palette size")
    parser.add_argument(
        "-m", "--method", choices=["paper", "nearest", "box"], default="paper",
        help="'paper' is the full algorithm; 'nearest'/'box' are naive downscale + median-cut baselines",
    )
    parser.add_argument("--palette", type=Path, metavar="FILE", help="also save the palette as an image swatch")
    parser.add_argument("-q", "--quiet", action="store_true", help="don't print progress")

    tuning = parser.add_argument_group("algorithm tuning (paper method only)")
    tuning.add_argument("--work-scale", type=_positive_int, default=defaults.work_scale,
                        help="max working pixels per output pixel per side; lower is faster, higher follows edges better")
    tuning.add_argument("--compactness", type=float, default=defaults.compactness,
                        help="superpixel compactness (higher = more grid-like)")
    tuning.add_argument("--smoothing", type=float, default=defaults.smoothing,
                        help="Laplacian smoothing of superpixel centres (0..1)")
    tuning.add_argument("--sigma-space", type=float, default=defaults.sigma_space,
                        help="bilateral filter spatial sigma in output pixels (0 = off)")
    tuning.add_argument("--sigma-color", type=float, default=defaults.sigma_color,
                        help="bilateral filter colour sigma in Lab units (0 = off)")
    tuning.add_argument("--saturation", type=float, default=defaults.saturation,
                        help="saturation multiplier applied to the result")
    tuning.add_argument("--alpha", type=float, default=defaults.alpha, help="annealing cooling factor (0..1)")
    tuning.add_argument("--max-iter", type=_positive_int, default=defaults.max_iter, help="iteration cap")
    return parser


def _output_size(args: argparse.Namespace, image: Image.Image, parser: argparse.ArgumentParser) -> tuple[int, int]:
    if args.size and (args.width or args.height):
        parser.error("use either --size or --width/--height, not both")
    if args.size:
        return args.size
    if args.width and args.height:
        return args.width, args.height
    if args.width:
        return args.width, max(1, round(image.height * args.width / image.width))
    if args.height:
        return max(1, round(image.width * args.height / image.height)), args.height
    w = 64
    return w, max(1, round(image.height * w / image.width))


def _save_palette(colors, path: Path, swatch: int = 32) -> None:
    img = Image.new("RGB", (swatch * max(1, len(colors)), swatch))
    for i, c in enumerate(colors):
        img.paste(tuple(int(v) for v in c), (i * swatch, 0, (i + 1) * swatch, swatch))
    img.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        image = Image.open(args.input)
        image.load()
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read {args.input}: {exc}")

    width, height = _output_size(args, image, parser)
    output = args.output or args.input.with_name(f"{args.input.stem}_pixelated_{width}x{height}.png")

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    log(f"{args.input} ({image.width}x{image.height}) -> {width}x{height}, up to {args.colors} colours [{args.method}]")
    start = time.perf_counter()

    palette = None
    if args.method == "paper":
        params = Params(
            colors=args.colors,
            compactness=args.compactness,
            smoothing=args.smoothing,
            sigma_space=args.sigma_space,
            sigma_color=args.sigma_color,
            saturation=args.saturation,
            alpha=args.alpha,
            work_scale=args.work_scale,
            max_iter=args.max_iter,
        )
        last = [-1]

        def on_progress(pr: Progress) -> None:
            if pr.palette_size != last[0]:
                last[0] = pr.palette_size
                log(f"  iter {pr.iteration:4d}  T={pr.temperature:9.2f}  colours={pr.palette_size}")

        result = pixelate(image, width, height, params, progress=None if args.quiet else on_progress)
        out = result.image
        palette = result.palette
        log(f"  done: {result.iterations} iterations, {len(palette)} colours")
    else:
        out = resize_baseline(image, width, height, args.colors, args.method)
        if args.palette:
            palette = sorted(set(out.convert("RGB").getdata()))

    if args.scale > 1:
        out = out.resize((width * args.scale, height * args.scale), Image.Resampling.NEAREST)

    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    if args.palette and palette is not None:
        _save_palette(palette, args.palette)
        log(f"palette -> {args.palette}")
    log(f"wrote {output} ({out.width}x{out.height}) in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
