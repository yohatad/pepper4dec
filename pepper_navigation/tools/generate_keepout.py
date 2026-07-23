#!/usr/bin/env python3
"""
generate_keepout.py.

Generates a Nav2-compatible keepout zone mask (PGM image + YAML metadata) from
a list of rectangle and circle exclusion zones defined in pixel coordinates.
Run with `python3 generate_keepout.py` after editing the MAP_CONFIG,
RECTANGLES, and CIRCLES sections below to match the target map and desired
keepout areas.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: March 31, 2026
Version: v1.0
"""

import numpy as np
import os

# ─────────────────────────────────────────────
#  CONFIG — Edit this section
# ─────────────────────────────────────────────

MAP_CONFIG = {
    "width":      226,
    "height":     167,
    "resolution": 0.05,
    "origin":     [-2.88804, -5.12021, 0.0],  # copied from map.yaml
    "output_name": "keepout_zone",
    "output_dir":  ".",
}

# Rectangles: defined in PIXEL coordinates (col, row from top-left)
# Format: {"x": left_col, "y": top_row, "w": width_px, "h": height_px}
RECTANGLES = [
    {"x": 67, "y": 85,  "w": 5, "h": 5},   # example rect 1
    {"x": 103, "y": 85, "w": 5, "h": 5},  # example rect 2
]

# Circles: defined in PIXEL coordinates
# Format: {"cx": center_col, "cy": center_row, "r": radius_px}
CIRCLES = [
    # {"cx": 600, "cy": 150, "r": 60},   # example circle 1
    # {"cx": 200, "cy": 400, "r": 40},   # example circle 2
]

# ─────────────────────────────────────────────


def create_keepout_image(width, height, rectangles, circles):
    """
    Create keepout mask as numpy array.

    White (255) = free space
    Black (0)   = keepout zone
    """
    # Start with all white (free space)
    img = np.full((height, width), 255, dtype=np.uint8)

    # Draw rectangles
    for rect in rectangles:
        x1 = max(0, rect["x"])
        y1 = max(0, rect["y"])
        x2 = min(width,  rect["x"] + rect["w"])
        y2 = min(height, rect["y"] + rect["h"])
        img[y1:y2, x1:x2] = 0
        print(f"  Rectangle: col={x1}-{x2}, row={y1}-{y2}")

    # Draw circles
    for circle in circles:
        cx, cy, r = circle["cx"], circle["cy"], circle["r"]
        for row in range(max(0, cy - r), min(height, cy + r + 1)):
            for col in range(max(0, cx - r), min(width, cx + r + 1)):
                if (col - cx) ** 2 + (row - cy) ** 2 <= r ** 2:
                    img[row, col] = 0
        print(f"  Circle: center=({cx},{cy}), radius={r}px")

    return img


def save_pgm(img, filepath):
    """Save numpy array as binary PGM (P5 format) — required by Nav2."""
    height, width = img.shape
    with open(filepath, "wb") as f:
        # PGM header
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode("ascii"))
        # Pixel data (row by row, top to bottom)
        f.write(img.tobytes())
    print(f"  Saved PGM: {filepath}")


def save_yaml(config, pgm_filename, yaml_filepath):
    """Save Nav2-compatible map YAML."""
    origin = config["origin"]
    content = f"""image: {pgm_filename}
resolution: {config['resolution']}
origin: [{origin[0]}, {origin[1]}, {origin[2]}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(yaml_filepath, "w") as f:
        f.write(content)
    print(f"  Saved YAML: {yaml_filepath}")


def meters_to_pixels(x_m, y_m, origin, resolution, height):
    """Convert world coordinates (meters) to pixel coordinates."""
    col = int((x_m - origin[0]) / resolution)
    row = height - int((y_m - origin[1]) / resolution)  # flip Y axis
    return col, row


def pixels_to_meters(col, row, origin, resolution, height):
    """Convert pixel coordinates to world coordinates (meters)."""
    x_m = col * resolution + origin[0]
    y_m = (height - row) * resolution + origin[1]
    return x_m, y_m


def main():
    cfg = MAP_CONFIG
    width = cfg["width"]
    height = cfg["height"]
    out_dir = cfg["output_dir"]
    out_name = cfg["output_name"]

    os.makedirs(out_dir, exist_ok=True)

    pgm_path = os.path.join(out_dir, f"{out_name}.pgm")
    yaml_path = os.path.join(out_dir, f"{out_name}.yaml")

    print(f"\n{'='*50}")
    print("  Nav2 Keepout Zone Generator")
    print(f"{'='*50}")
    print(f"  Map size : {width} x {height} px")
    print(f"  Resolution: {cfg['resolution']} m/px")
    print(f"  Origin   : {cfg['origin']}")
    print()

    print("Drawing keepout zones...")
    img = create_keepout_image(width, height, RECTANGLES, CIRCLES)

    # Stats
    total_px = width * height
    keepout_px = int(np.sum(img == 0))
    free_px = total_px - keepout_px
    keepout_m2 = keepout_px * (cfg["resolution"] ** 2)

    print("\nStats:")
    print(f"  Total pixels   : {total_px}")
    print(f"  Keepout pixels : {keepout_px} ({100*keepout_px/total_px:.1f}%)")
    print(f"  Free pixels    : {free_px} ({100*free_px/total_px:.1f}%)")
    print(f"  Keepout area   : {keepout_m2:.2f} m²")

    print("\nSaving files...")
    save_pgm(img, pgm_path)
    save_yaml(cfg, f"{out_name}.pgm", yaml_path)

    print(f"\n{'='*50}")
    print(f"  Done! Files saved to: {os.path.abspath(out_dir)}/")
    print(f"{'='*50}")

    print(f"""
Nav2 keepout_filter params.yaml:
─────────────────────────────────
keepout_filter:
  ros__parameters:
    use_sim_time: false
    filter_info_topic: "/costmap_filter_info"
    mask_topic: "/keepout_filter_mask"

map_server:
  ros__parameters:
    yaml_filename: "{os.path.abspath(yaml_path)}"
""")


if __name__ == "__main__":
    main()
