#!/usr/bin/env python3
"""Render logo.svg to PNG and image-diff it against a reference image.

Usage: python3 diff.py REFERENCE.png [logo.svg]
Outputs: render.png, diff_heatmap.png, side_by_side.png and prints scores.
"""
import sys
import cairosvg
from PIL import Image, ImageChops
import math

ref_path = sys.argv[1]
svg_path = sys.argv[2] if len(sys.argv) > 2 else "logo.svg"

ref = Image.open(ref_path).convert("RGBA")
W, H = ref.size

# Composite reference onto white so transparent areas match the SVG render bg.
white = Image.new("RGBA", (W, H), (255, 255, 255, 255))
ref = Image.alpha_composite(white, ref).convert("RGB")

cairosvg.svg2png(url=svg_path, write_to="render.png",
                 output_width=W, output_height=H, background_color="white")
ren = Image.open("render.png").convert("RGB")

diff = ImageChops.difference(ref, ren)
diff.save("diff_heatmap.png")

# Mean abs error + simple similarity score.
hist = diff.histogram()
total = 0
count = 0
for channel in range(3):
    base = channel * 256
    for value in range(256):
        n = hist[base + value]
        total += value * n
        count += n
mae = total / count if count else 0
similarity = 100 * (1 - mae / 255)

side = Image.new("RGB", (W * 3, H), (255, 255, 255))
side.paste(ref, (0, 0))
side.paste(ren, (W, 0))
side.paste(diff, (W * 2, 0))
side.save("side_by_side.png")

print(f"size={W}x{H}  mean_abs_error={mae:.2f}/255  similarity={similarity:.2f}%")
print("wrote render.png, diff_heatmap.png, side_by_side.png")
