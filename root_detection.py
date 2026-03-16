"""
Root Detection for OT-2 Plate Images
"""

import numpy as np
import cv2
import pybullet as p


def detect_roots(plate_image_path, debug=False):
    """Detect root positions in the plate image.
    Returns list of (norm_x, norm_y) in [0,1] relative to the plate region.
    """
    img = cv2.imread(plate_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [root_detect] Could not load {plate_image_path}")
        return []

    h, w = img.shape
    _, plate_mask = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    plate_contour = max(contours, key=cv2.contourArea)
    x_p, y_p, w_p, h_p = cv2.boundingRect(plate_contour)
    plate_crop = img[y_p:y_p+h_p, x_p:x_p+w_p]

    plate_blur = cv2.GaussianBlur(plate_crop, (5, 5), 0)
    _, root_mask = cv2.threshold(plate_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(root_mask, connectivity=8)

    min_area = 0.002 * w_p * h_p
    max_area = 0.25 * w_p * h_p

    root_positions = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            cx, cy = centroids[label_id]
            root_positions.append((cx / w_p, cy / h_p))

    root_positions.sort(key=lambda r: r[0])

    if debug:
        print(f"  [root_detect] Plate: x={x_p}, y={y_p}, w={w_p}, h={h_p}")
        print(f"  [root_detect] Found {len(root_positions)} roots")
        for i, (nx, ny) in enumerate(root_positions):
            print(f"    Root {i+1}: ({nx:.3f}, {ny:.3f})")

    return root_positions


def roots_to_world_coords(root_positions, sim, drop_height=0.10):
    """Convert normalized root positions to world coordinates."""
    plate_size = 0.15
    spec_pos = p.getBasePositionAndOrientation(sim.specimenIds[0])[0]
    plate_cx, plate_cy, plate_cz = spec_pos

    print(f"  [coord_map] Specimen: [{plate_cx:.4f}, {plate_cy:.4f}, {plate_cz:.4f}]")

    world_coords = []
    for i, (nx, ny) in enumerate(root_positions):
        wx = plate_cx - (ny - 0.5) * plate_size
        wy = plate_cy + (nx - 0.5) * plate_size
        wz = plate_cz + drop_height
        world_coords.append([wx, wy, wz])
        print(f"  [coord_map] Root {i+1}: ({nx:.3f},{ny:.3f}) -> [{wx:.4f}, {wy:.4f}, {wz:.4f}]")

    return world_coords
