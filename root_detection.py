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


def _trace_by_smoothed_intensity(column_gray, min_gap=20, dark_threshold=12, smooth_window=15):
    """
    Trace a root using SMOOTHED intensity contrast.
    
    1. For each row, compute contrast = median - min pixel.
    2. Smooth the contrast with a rolling window.
    3. A row has root if smoothed contrast > threshold.
    4. Stop at first gap of min_gap rows where smoothed contrast is low.
    
    Smoothing is critical: it averages out isolated noisy pixels that
    would bridge gaps in raw contrast, while preserving the continuous
    signal from actual roots.
    """
    h, w = column_gray.shape
    if h == 0 or w == 0:
        return None

    # Per-row contrast: how much darker is the darkest pixel vs background
    row_min = np.min(column_gray, axis=1).astype(float)
    row_median = np.median(column_gray, axis=1).astype(float)
    contrast = row_median - row_min

    # Smooth the contrast profile — kills isolated noise spikes
    kernel = np.ones(smooth_window) / smooth_window
    smooth_contrast = np.convolve(contrast, kernel, mode='same')

    # Root present if smoothed contrast exceeds threshold
    row_has_root = smooth_contrast > dark_threshold

    active_rows = np.where(row_has_root)[0]
    if len(active_rows) == 0:
        return None

    # Trace with gap detection on smoothed signal
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

    # Find darkest pixel x at tip row
    tip_row_data = column_gray[tip_row, :]
    tip_col = float(np.argmin(tip_row_data))

    return (tip_row, tip_col)


def detect_roots(image_path, debug=True):
    """
    Seed-first detection with smoothed intensity profiling.
    
    The smoothed contrast naturally separates:
    - Root signal (sustained dark line → high smoothed contrast)
    - Noise (isolated dark pixels → averaged away by smoothing)
    - Water droplets below roots (appear after a gap)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [Vision] ERROR: Could not read {image_path}")
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Step 1: Plate ROI ----
    plate_mask, (px, py, pw, ph) = _extract_plate_roi(gray)

    # ---- Step 2: Find seeds in top 30% ----
    seed_zone_top = py
    seed_zone_bottom = py + int(ph * 0.30)

    seed_mask = np.zeros_like(gray)
    seed_mask[seed_zone_top:seed_zone_bottom, px:px + pw] = 255
    seed_mask = cv2.bitwise_and(seed_mask, plate_mask)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    seed_thresh1 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )
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
        if center_x < px + 0.05 * pw or center_x > px + 0.95 * pw:
            continue
        seeds.append((center_x, center_y, area, cnt))

    seeds = sorted(seeds, key=lambda s: s[0])

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

    # ---- Step 3: Trace each root ----
    root_zone_bottom = py + int(ph * 0.55)
    root_tips = []
    column_half_width = int(pw * 0.04)
    gap_threshold = max(20, int(ph * 0.02))

    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)

    for (seed_cx, seed_cy, seed_area, seed_cnt) in merged_seeds:
        col_left = max(px, int(seed_cx - column_half_width))
        col_right = min(px + pw, int(seed_cx + column_half_width))
        search_top = int(seed_cy)
        search_bottom = root_zone_bottom

        column = gray_smooth[search_top:search_bottom, col_left:col_right]

        if column.size == 0:
            root_tips.append((seed_cx / w, (seed_cy + 15) / h))
            continue

        # Smoothed intensity tracing
        result = _trace_by_smoothed_intensity(
            column,
            min_gap=gap_threshold,
            dark_threshold=12,
            smooth_window=15
        )

        if result is None:
            # Fallback: try more sensitive
            result = _trace_by_smoothed_intensity(
                column, min_gap=gap_threshold,
                dark_threshold=8, smooth_window=20
            )

        if result is None:
            root_tips.append((seed_cx / w, (seed_cy + 15) / h))
            continue

        tip_row, tip_col = result
        tip_y_abs = search_top + tip_row
        tip_x_abs = col_left + tip_col

        root_tips.append((tip_x_abs / w, tip_y_abs / h))

    root_tips = sorted(root_tips, key=lambda r: r[0])

    # ---- Fallback ----
    standard_x = [0.18, 0.35, 0.50, 0.65, 0.82]
    default_tip_y = (py + int(ph * 0.35)) / float(h)
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
    """Maps 2D image detections to 3D world space."""
    world_targets = []
    spec_pos, _ = p.getBasePositionAndOrientation(sim.specimenIds[0])
    plate_x, plate_y, plate_z = spec_pos

    plate_size = 0.15
    x_offset = 0.004
    y_offset = -0.008

    for nx, ny in root_positions:
        world_x = plate_x + (ny - 0.5) * plate_size + x_offset
        world_y = plate_y + (nx - 0.5) * plate_size + y_offset
        world_z = plate_z + drop_height
        world_targets.append([world_x, world_y, world_z])

    return world_targets