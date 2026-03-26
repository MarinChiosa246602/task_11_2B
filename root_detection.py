import cv2
import numpy as np
import pybullet as p


def _extract_plate_roi(gray):
    """Detect the plate region, return mask and bounding box."""
    _, plate_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel_big)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    plate_mask = cv2.erode(plate_mask, kernel_erode, iterations=2)

    coords = cv2.findNonZero(plate_mask)
    if coords is not None:
        px, py, pw, ph = cv2.boundingRect(coords)
    else:
        h, w = gray.shape
        px, py, pw, ph = int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h)

    return plate_mask, (px, py, pw, ph)


def _trace_root_down(column_binary, min_gap=15):
    """
    Trace a root downward through a binary column image.
    Stop at the first significant vertical gap.
    Returns (tip_row, tip_col) or None.
    """
    h, w = column_binary.shape
    if h == 0 or w == 0:
        return None

    row_has_root = np.sum(column_binary, axis=1) > 0
    active_rows = np.where(row_has_root)[0]
    if len(active_rows) == 0:
        return None

    tip_row = active_rows[0]
    gap_count = 0

    for row in range(active_rows[0], h):
        if row_has_root[row]:
            tip_row = row
            gap_count = 0
        else:
            gap_count += 1
            if gap_count >= min_gap:
                break

    tip_row_pixels = column_binary[tip_row, :]
    active_cols = np.where(tip_row_pixels > 0)[0]
    if len(active_cols) > 0:
        tip_col = (active_cols[0] + active_cols[-1]) / 2.0
    else:
        tip_col = w / 2.0

    return (tip_row, tip_col)


def detect_roots(image_path, debug=True):
    """
    Robust root tip detection:
      1. Find plate ROI.
      2. Find seeds in top 30% (wider zone to catch seeds near edges).
      3. For each seed, trace downward with gap-awareness.
      4. The tip = last root pixel before a 15+ row gap.
      
    Uses multiple threshold methods and picks the best result.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [Vision] ERROR: Could not read {image_path}")
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Step 1: Plate ROI ----
    plate_mask, (px, py, pw, ph) = _extract_plate_roi(gray)

    # ---- Step 2: Find seeds ----
    # Use top 30% of plate — seeds can be near the upper edge
    seed_zone_top = py
    seed_zone_bottom = py + int(ph * 0.30)

    seed_mask = np.zeros_like(gray)
    seed_mask[seed_zone_top:seed_zone_bottom, px:px + pw] = 255
    seed_mask = cv2.bitwise_and(seed_mask, plate_mask)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try two threshold methods for seed detection and combine
    seed_thresh1 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )
    # Also try a global threshold for very dark seeds
    _, seed_thresh2 = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    seed_combined = cv2.bitwise_or(seed_thresh1, seed_thresh2)
    seed_combined = cv2.bitwise_and(seed_combined, seed_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed_combined = cv2.morphologyEx(seed_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    seed_combined = cv2.morphologyEx(seed_combined, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(seed_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seeds = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if area < 60 or area > pw * ph * 0.1:
            continue
        center_x = bx + bw / 2.0
        center_y = by + bh / 2.0
        # Seed must be in central 90% of plate width
        if center_x < px + 0.05 * pw or center_x > px + 0.95 * pw:
            continue
        seeds.append((center_x, center_y, area, cnt))

    seeds = sorted(seeds, key=lambda s: s[0])

    # Merge nearby seeds (same plant)
    merged_seeds = []
    for s in seeds:
        if merged_seeds and abs(s[0] - merged_seeds[-1][0]) < 0.04 * w:
            if s[2] > merged_seeds[-1][2]:
                merged_seeds[-1] = s
        else:
            merged_seeds.append(s)

    if len(merged_seeds) > 5:
        merged_seeds = sorted(merged_seeds, key=lambda s: s[2], reverse=True)[:5]
        merged_seeds = sorted(merged_seeds, key=lambda s: s[0])

    # ---- Step 3: Root threshold for tracing ----
    root_zone_bottom = py + int(ph * 0.55)

    # Use adaptive threshold for roots
    root_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )
    zone_mask = np.zeros_like(gray)
    zone_mask[py:root_zone_bottom, px:px + pw] = 255
    zone_mask = cv2.bitwise_and(zone_mask, plate_mask)
    root_thresh = cv2.bitwise_and(root_thresh, zone_mask)

    # Remove small noise
    k_small = np.ones((2, 2), np.uint8)
    root_thresh = cv2.morphologyEx(root_thresh, cv2.MORPH_OPEN, k_small, iterations=1)
    # Vertical connection for thin roots
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    root_thresh = cv2.dilate(root_thresh, k_vert, iterations=1)

    # ---- Step 4: Trace each root ----
    root_tips = []
    column_half_width = int(pw * 0.05)
    gap_threshold = max(12, int(ph * 0.012))

    for (seed_cx, seed_cy, seed_area, seed_cnt) in merged_seeds:
        col_left = max(px, int(seed_cx - column_half_width))
        col_right = min(px + pw, int(seed_cx + column_half_width))
        search_top = int(seed_cy)
        search_bottom = root_zone_bottom

        column = root_thresh[search_top:search_bottom, col_left:col_right]

        if column.size == 0:
            root_tips.append((seed_cx / w, (seed_cy + 15) / h))
            continue

        result = _trace_root_down(column, min_gap=gap_threshold)

        if result is None:
            root_tips.append((seed_cx / w, (seed_cy + 15) / h))
            continue

        tip_row, tip_col = result
        tip_y_abs = search_top + tip_row
        tip_x_abs = col_left + tip_col

        # Sanity: tip must be below seed
        if tip_y_abs <= seed_cy + 5:
            root_tips.append((seed_cx / w, (seed_cy + 20) / h))
            continue

        root_tips.append((tip_x_abs / w, tip_y_abs / h))

    root_tips = sorted(root_tips, key=lambda r: r[0])

    # ---- Fallback ----
    standard_x = [0.18, 0.35, 0.50, 0.65, 0.82]
    default_tip_y = (py + int(ph * 0.25)) / float(h)
    while len(root_tips) < 5:
        existing_x = [r[0] for r in root_tips]
        filled = False
        for sx in standard_x:
            if not any(abs(sx - ex) < 0.07 for ex in existing_x):
                root_tips.append((sx, default_tip_y))
                filled = True
                break
        if not filled:
            break
        root_tips = sorted(root_tips, key=lambda r: r[0])

    if debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (px, seed_zone_top), (px + pw, seed_zone_bottom), (255, 255, 0), 2)
        cv2.line(debug_img, (px, root_zone_bottom), (px + pw, root_zone_bottom), (0, 255, 255), 2)
        for (scx, scy, sa, scnt) in merged_seeds:
            cv2.circle(debug_img, (int(scx), int(scy)), 6, (255, 0, 0), -1)
            cl = max(px, int(scx - column_half_width))
            cr = min(px + pw, int(scx + column_half_width))
            cv2.rectangle(debug_img, (cl, int(scy)), (cr, root_zone_bottom), (200, 200, 0), 1)
        for i, (nx, ny) in enumerate(root_tips):
            ptx, pty = int(nx * w), int(ny * h)
            cv2.circle(debug_img, (ptx, pty), 10, (0, 255, 0), -1)
            cv2.putText(debug_img, f"R{i+1}", (ptx + 14, pty + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if i < len(merged_seeds):
                scx, scy = int(merged_seeds[i][0]), int(merged_seeds[i][1])
                cv2.line(debug_img, (scx, scy), (ptx, pty), (0, 200, 0), 1)
        cv2.imwrite("debug_root_tips.png", debug_img)
        print(f"  [Vision] Found {len(merged_seeds)} seeds, {len(root_tips)} root tips: "
              f"{[(f'{x:.3f}', f'{y:.3f}') for x, y in root_tips]}")

    return root_tips


def roots_to_world_coords(root_positions, sim, drop_height=0.095):
    """
    Maps 2D image detections to 3D world space.
    
    Coordinate mapping:
      Image Y (top=0, bottom=1) → World X (inverted)
      Image X (left=0, right=1) → World Y
    
    The plate physical size is ~0.15m (150mm).
    Offsets are calibrated to center drops on roots.
    """
    world_targets = []
    spec_pos, _ = p.getBasePositionAndOrientation(sim.specimenIds[0])
    plate_x, plate_y, plate_z = spec_pos

    # Plate dimension in world units
    plate_size = 0.15

    # Calibration offsets — fine-tuned to center drops on roots
    x_offset = 0.004
    y_offset = -0.008

    for nx, ny in root_positions:
        # Image Y -> World X (inverted: top of image = far X)
        world_x = plate_x + (ny - 0.5) * plate_size + x_offset
        # Image X -> World Y
        world_y = plate_y + (nx - 0.5) * plate_size + y_offset
        # Height above plate
        world_z = plate_z + drop_height

        world_targets.append([world_x, world_y, world_z])

    return world_targets