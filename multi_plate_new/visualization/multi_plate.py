import os

import cv2
import numpy as np

from single_plate_pose.detection import find_predictions_recursive


def draw_multi_plate_detections(image, detections, save_path=None):
    vis = image.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = f"#{i} conf={det.confidence:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        if det.points is not None:
            for p in det.points:
                cv2.circle(vis, tuple(p.astype(int)), 4, (255, 0, 0), -1)

    if save_path is not None:
        cv2.imwrite(save_path, vis)

    return vis

def visualize_raw_predictions(image, raw_result):
    predictions = find_predictions_recursive(raw_result)

    vis = image.copy()

    for idx, pred in enumerate(predictions):

        if not all(k in pred for k in ("x", "y", "width", "height")):
            continue

        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        conf = pred.get("confidence", 0)

        cv2.putText(
            vis,
            f"{idx}: {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis

def make_crops_grid_from_folder(
        crops_dir,
        save_path=None,
        thumb_width=300,
        cols=3,
        show=True
):
    image_files = [
        f for f in os.listdir(crops_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()

    thumbs = []

    for idx, filename in enumerate(image_files):
        path = os.path.join(crops_dir, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_width / w
        new_h = int(h * scale)

        thumb = cv2.resize(
            img,
            (thumb_width, new_h),
            interpolation=cv2.INTER_AREA
        )

        cv2.putText(
            thumb,
            filename,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        thumbs.append(thumb)

    if len(thumbs) == 0:
        raise RuntimeError(f"No crop images found in: {crops_dir}")

    max_h = max(t.shape[0] for t in thumbs)

    padded = []
    for t in thumbs:
        pad_h = max_h - t.shape[0]
        t_padded = cv2.copyMakeBorder(
            t,
            0,
            pad_h,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        padded.append(t_padded)

    rows = []

    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]

        while len(row_imgs) < cols:
            blank = np.zeros_like(padded[0])
            row_imgs.append(blank)

        row = np.hstack(row_imgs)
        rows.append(row)

    grid = np.vstack(rows)

    if save_path is not None:
        cv2.imwrite(save_path, grid)

    if show:
        cv2.imshow("Plate crops grid", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid

def draw_refined_corners(image, refinement_results, save_path=None):
    vis = image.copy()

    for r in refinement_results:
        idx = r["index"]

        if not r["success"]:
            continue

        pts = r["image_points"].astype(int)

        cv2.polylines(vis, [pts], True, (255, 0, 0), 3)

        for j, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"{idx}.{j}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    if save_path is not None:
        cv2.imwrite(save_path, vis)

    return vis

def draw_refined_plates_grid(
        image,
        refinement_results,
        save_path=None,
        thumb_width=500,
        cols=2,
        show=True,
        padding_ratio=0.35
):
    import cv2
    import numpy as np

    cells = []

    for r in refinement_results:
        if not r["success"]:
            continue

        idx = r["index"]
        pts = r["image_points"].astype(np.float32)

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        w = x2 - x1
        h = y2 - y1

        pad_x = w * padding_ratio
        pad_y = h * padding_ratio

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(image.shape[1], int(x2 + pad_x))
        y2 = min(image.shape[0], int(y2 + pad_y))

        crop = image[y1:y2, x1:x2].copy()

        pts_crop = pts.copy()
        pts_crop[:, 0] -= x1
        pts_crop[:, 1] -= y1
        pts_crop_int = pts_crop.astype(int)

        cv2.polylines(crop, [pts_crop_int], True, (0, 255, 255), 2)

        for j, (x, y) in enumerate(pts_crop_int):
            cv2.circle(crop, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(
                crop,
                str(j),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        conf = r["detection"].confidence

        cv2.putText(
            crop,
            f"#{idx} ({conf:.2f})",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        ch, cw = crop.shape[:2]
        scale = thumb_width / cw
        thumb_h = int(ch * scale)

        thumb = cv2.resize(
            crop,
            (thumb_width, thumb_h),
            interpolation=cv2.INTER_AREA
        )

        cells.append(thumb)

    if len(cells) == 0:
        raise RuntimeError("No successful refined plates to visualize.")

    max_h = max(c.shape[0] for c in cells)

    padded = []
    for c in cells:
        pad_h = max_h - c.shape[0]
        c = cv2.copyMakeBorder(
            c,
            0,
            pad_h,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        padded.append(c)

    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]

        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))

        rows.append(np.hstack(row))

    grid = np.vstack(rows)

    if save_path is not None:
        cv2.imwrite(save_path, grid)

    if show:
        cv2.imshow("Refined plates grid", grid)
        cv2.waitKey(0)
        cv2.destroyWindow("Refined plates grid")

def draw_homography_plates_grid(
        image,
        refinement_results,
        save_path=None,
        warped_width=520,
        warped_height=120,
        cols=2,
        show=True
):
    cells = []

    dst_pts = np.array([
        [0, 0],
        [warped_width - 1, 0],
        [warped_width - 1, warped_height - 1],
        [0, warped_height - 1]
    ], dtype=np.float32)

    for r in refinement_results:
        if not r["success"]:
            continue

        idx = r["index"]
        pts = r["image_points"].astype(np.float32)

        H_img_to_rect = cv2.getPerspectiveTransform(pts, dst_pts)

        warped = cv2.warpPerspective(
            image,
            H_img_to_rect,
            (warped_width, warped_height)
        )

        canvas = np.zeros(
            (warped_height + 35, warped_width, 3),
            dtype=np.uint8
        )

        canvas[35:, :] = warped

        conf = r["detection"].confidence
        cv2.putText(
            canvas,
            f"Plate #{idx}  conf={conf:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            canvas,
            (0, 35),
            (warped_width - 1, warped_height + 34),
            (0, 255, 255),
            2
        )

        cells.append(canvas)

    if len(cells) == 0:
        raise RuntimeError("No successful refined plates to visualize.")

    rows = []

    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]

        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))

        rows.append(np.hstack(row))

    grid = np.vstack(rows)

    if save_path is not None:
        cv2.imwrite(save_path, grid)

    if show:
        cv2.imshow("Homography input plates grid", grid)
        cv2.waitKey(0)
        cv2.destroyWindow("Homography input plates grid")

    return grid

def draw_single_plate_refinement_debug(
        image,
        refinement_result,
        save_path=None,
        warp_width=520,
        warp_height=120,
        show=False
):
    if not refinement_result["success"]:
        return None

    r = refinement_result
    idx = r["index"]
    pts = r["image_points"].astype(np.float32)
    mask = r["mask"]

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    pad_x = int((x2 - x1) * 0.6)
    pad_y = int((y2 - y1) * 0.8)

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(image.shape[1], int(x2 + pad_x))
    y2 = min(image.shape[0], int(y2 + pad_y))

    crop = image[y1:y2, x1:x2].copy()

    pts_crop = pts.copy()
    pts_crop[:, 0] -= x1
    pts_crop[:, 1] -= y1
    pts_crop_int = pts_crop.astype(int)

    crop_with_quad = crop.copy()
    cv2.polylines(crop_with_quad, [pts_crop_int], True, (0, 255, 255), 3)

    for j, (x, y) in enumerate(pts_crop_int):
        cv2.circle(crop_with_quad, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            crop_with_quad,
            str(j),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    dst = np.array([
        [0, 0],
        [warp_width - 1, 0],
        [warp_width - 1, warp_height - 1],
        [0, warp_height - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(pts, dst)

    warped = cv2.warpPerspective(
        image,
        H,
        (warp_width, warp_height)
    )

    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    crop_h, crop_w = crop.shape[:2]

    mask_vis = cv2.resize(
        mask_vis,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )

    warped_vis = cv2.resize(
        warped,
        (crop_w, crop_h),
        interpolation=cv2.INTER_AREA
    )

    cv2.putText(crop, "original crop", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(mask_vis, "yellow mask", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(crop_with_quad, "selected quad", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(warped_vis, "warped by selected quad", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    top = np.hstack([crop, mask_vis])
    bottom = np.hstack([crop_with_quad, warped_vis])
    debug_grid = np.vstack([top, bottom])

    cv2.putText(
        debug_grid,
        f"Plate #{idx}",
        (10, debug_grid.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    if save_path is not None:
        cv2.imwrite(save_path, debug_grid)

    if show:
        cv2.imshow(f"plate {idx} refinement debug", debug_grid)
        cv2.waitKey(0)
        cv2.destroyWindow(f"plate {idx} refinement debug")

    return debug_grid


def draw_all_plate_refinement_debugs(
        image,
        refinement_results,
        output_dir,
        show=False
):
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for r in refinement_results:
        if not r["success"]:
            continue

        idx = r["index"]

        save_path = os.path.join(
            output_dir,
            f"plate_{idx}_refinement_debug.jpg"
        )

        draw_single_plate_refinement_debug(
            image,
            r,
            save_path=save_path,
            show=show
        )

        saved_paths.append(save_path)

    return saved_paths

