from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


from video_pose.frame_processor import process_sampled_frame
from video_pose.frame_sampler import read_video_metadata, sample_video_frames
from video_pose.filtering import get_accepted_detections
from video_pose.tracking import PlateTrackerConfig, associate_frame_detections

from video_pose.temporal_corner_refinement.alignment import warp_quad
from video_pose.temporal_corner_refinement.window import SlidingTrackWindowBuilder
from video_pose.temporal_corner_refinement.methods.geometry_alignment import (
    run_geometry_baseline,
)
from video_pose.temporal_corner_refinement.methods.corner_color import (
    CornerColorConfig,
    run_corner_color,
)


API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set.")


VIDEO_PATH = PROJECT_ROOT / "cars_photos" / "test_video.mov"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "video_pose"
    / "outputs"
    / "temporal_corner_refinement"
    / "corner_color"
)
PANELS_DIR = OUTPUT_DIR / "panels"

CSV_PATH = OUTPUT_DIR / "corner_color_per_track_diverse.csv"
CANDIDATE_CSV_PATH = OUTPUT_DIR / "corner_color_candidates.csv"


PROCESS_EVERY_N_FRAMES = 1
USE_CACHED_ROBOFLOW_JSON = True

MIN_CONFIDENCE = 0.35
MIN_DETECTION_AREA = 300.0
MIN_WIDTH = 80.0
MIN_HEIGHT = 20.0
NMS_IOU_THRESHOLD = 0.50

REFINEMENT_METHOD = "robust_mask_lines"
REFINEMENT_DEBUG = False
REFINEMENT_PANEL_DEBUG = False

WINDOW_SIZE = 7
MAX_FRAMES = None

WINDOW_BORDER_MARGIN_PX = 4.0

CORNER_CONFIG = CornerColorConfig()

PANELS_PER_TRACK = 2
MIN_PANEL_CENTER_GAP_FRAMES = 12

CELL_WIDTH = 300
CELL_HEIGHT = 220
LABEL_HEIGHT = 46

PANEL_COLUMNS = WINDOW_SIZE
PANEL_ROWS = 2
PANEL_WIDTH = PANEL_COLUMNS * CELL_WIDTH
PANEL_HEIGHT = PANEL_ROWS * (CELL_HEIGHT + LABEL_HEIGHT)

CORNER_NAMES = ("TL", "TR", "BR", "BL")


def observation_has_full_visibility(obs) -> bool:
    q = np.asarray(obs.detection.corners, dtype=np.float64)

    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return False

    h, w = obs.image.shape[:2]

    return bool(
        np.min(q[:, 0]) >= WINDOW_BORDER_MARGIN_PX
        and np.max(q[:, 0]) <= (w - 1 - WINDOW_BORDER_MARGIN_PX)
        and np.min(q[:, 1]) >= WINDOW_BORDER_MARGIN_PX
        and np.max(q[:, 1]) <= (h - 1 - WINDOW_BORDER_MARGIN_PX)
    )


def window_has_full_visibility(window) -> bool:
    return all(
        observation_has_full_visibility(obs)
        for obs in window.observations
    )


def aligned_raw_quads(window, H_to_reference) -> list[np.ndarray] | None:
    quads = []

    for i, obs in enumerate(window.observations):
        H = H_to_reference[i]

        if H is None:
            return None

        raw = np.asarray(
            obs.detection.corners,
            dtype=np.float64,
        )

        if i == window.center_index:
            q_ref = raw.copy()
        else:
            q_ref = warp_quad(raw, H)

        if not np.all(np.isfinite(q_ref)):
            return None

        quads.append(q_ref)

    return quads


def quad_width(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)

    return float(
        0.5
        * (
            np.linalg.norm(q[1] - q[0])
            + np.linalg.norm(q[2] - q[3])
        )
    )


def raw_variability_metrics(
    aligned_quads: list[np.ndarray],
) -> tuple[float, np.ndarray]:
    """
    Independent experiment-selection metric.

    For each corner:
      median distance of the seven aligned RAW estimates from their coordinate-
      wise median position, normalized by the median plate width.

    Overall variability = mean of the four corner variabilities.
    """
    stack = np.stack(aligned_quads, axis=0)  # [N, 4, 2]

    center_by_corner = np.median(
        stack,
        axis=0,
    )

    distances = np.linalg.norm(
        stack - center_by_corner[None, :, :],
        axis=2,
    )

    corner_px = np.median(
        distances,
        axis=0,
    )

    plate_width = float(
        np.median(
            [quad_width(q) for q in aligned_quads]
        )
    )

    corner_norm = corner_px / max(plate_width, 1.0)
    overall = float(np.mean(corner_norm))

    return overall, corner_norm


def propagate_reference_quad(
    window,
    H_to_reference,
    q_ref: np.ndarray,
):
    output = [
        np.asarray(
            obs.detection.corners,
            dtype=np.float64,
        ).copy()
        for obs in window.observations
    ]

    for i, H_i_to_ref in enumerate(H_to_reference):
        if H_i_to_ref is None:
            continue

        if i == window.center_index:
            output[i] = np.asarray(
                q_ref,
                dtype=np.float64,
            ).copy()
            continue

        try:
            H_ref_to_i = np.linalg.inv(H_i_to_ref)
        except np.linalg.LinAlgError:
            continue

        output[i] = warp_quad(
            q_ref,
            H_ref_to_i,
        )

    return tuple(output)


def quad_bbox(corners: np.ndarray):
    q = np.asarray(corners, dtype=np.float64)
    return (
        float(np.min(q[:, 0])),
        float(np.min(q[:, 1])),
        float(np.max(q[:, 0])),
        float(np.max(q[:, 1])),
    )


def crop_for_quads(
    image: np.ndarray,
    quads: list[np.ndarray],
):
    boxes = [quad_bbox(q) for q in quads]

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    x1 -= 0.22 * w
    x2 += 0.22 * w
    y1 -= 1.10 * h
    y2 += 1.10 * h

    image_h, image_w = image.shape[:2]
    target_aspect = CELL_WIDTH / CELL_HEIGHT

    crop_w = max(x2 - x1, 1.0)
    crop_h = max(y2 - y1, 1.0)
    crop_aspect = crop_w / crop_h

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    if crop_aspect < target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect

    crop_w = min(crop_w, float(image_w))
    crop_h = min(crop_h, float(image_h))

    x1 = cx - 0.5 * crop_w
    x2 = cx + 0.5 * crop_w
    y1 = cy - 0.5 * crop_h
    y2 = cy + 0.5 * crop_h

    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if x2 > image_w:
        shift = x2 - image_w
        x1 -= shift
        x2 = float(image_w)

    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if y2 > image_h:
        shift = y2 - image_h
        y1 -= shift
        y2 = float(image_h)

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_w), x2)
    y2 = min(float(image_h), y2)

    xi1 = max(0, int(np.floor(x1)))
    yi1 = max(0, int(np.floor(y1)))
    xi2 = min(image_w, int(np.ceil(x2)))
    yi2 = min(image_h, int(np.ceil(y2)))

    return (
        image[yi1:yi2, xi1:xi2].copy(),
        (xi1, yi1),
    )


def draw_quad(
    image: np.ndarray,
    corners: np.ndarray,
    offset,
    color,
    *,
    thickness: int,
):
    ox, oy = offset

    pts = np.asarray(corners, dtype=np.float64).copy()
    pts[:, 0] -= ox
    pts[:, 1] -= oy
    pts = np.round(pts).astype(np.int32)

    cv2.polylines(
        image,
        [pts.reshape(-1, 1, 2)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_to_cell(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros(
            (CELL_HEIGHT, CELL_WIDTH, 3),
            dtype=np.uint8,
        )

    h, w = image.shape[:2]

    scale = min(
        CELL_WIDTH / max(w, 1),
        CELL_HEIGHT / max(h, 1),
    )

    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (rw, rh))

    canvas = np.zeros(
        (CELL_HEIGHT, CELL_WIDTH, 3),
        dtype=np.uint8,
    )

    x0 = max(0, (CELL_WIDTH - rw) // 2)
    y0 = max(0, (CELL_HEIGHT - rh) // 2)

    canvas[
        y0:y0 + rh,
        x0:x0 + rw,
    ] = resized

    return canvas


def make_cell(
    image: np.ndarray,
    raw_quad: np.ndarray,
    corner_quad: np.ndarray,
    *,
    row_name: str,
    frame_index: int,
    corner_source_frames,
) -> np.ndarray:
    crop, offset = crop_for_quads(
        image,
        [raw_quad, corner_quad],
    )

    if row_name == "RAW":
        draw_quad(
            crop,
            raw_quad,
            offset,
            (0, 255, 0),
            thickness=3,
        )
        label_text = f"RAW | F{frame_index}"
    else:
        draw_quad(
            crop,
            corner_quad,
            offset,
            (0, 165, 255),
            thickness=3,
        )
        sources = " ".join(
            f"{name}{frame}"
            for name, frame in zip(
                CORNER_NAMES,
                corner_source_frames,
            )
        )
        label_text = f"CORNER | F{frame_index} | {sources}"

    cell_img = fit_to_cell(crop)
    label = np.zeros((LABEL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        label,
        label_text,
        (8, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([label, cell_img])


def render_panel(
    window,
    corner_output,
    corner_source_frames,
) -> np.ndarray:
    panel = np.zeros(
        (PANEL_HEIGHT, PANEL_WIDTH, 3),
        dtype=np.uint8,
    )
    row_height = CELL_HEIGHT + LABEL_HEIGHT

    for i, obs in enumerate(window.observations):
        raw = np.asarray(obs.detection.corners, dtype=np.float64)

        raw_cell = make_cell(
            obs.image, raw, corner_output[i],
            row_name="RAW",
            frame_index=obs.frame_index,
            corner_source_frames=corner_source_frames,
        )
        corner_cell = make_cell(
            obs.image, raw, corner_output[i],
            row_name="CORNER",
            frame_index=obs.frame_index,
            corner_source_frames=corner_source_frames,
        )

        x0 = i * CELL_WIDTH
        panel[0:row_height, x0:x0 + CELL_WIDTH] = raw_cell
        panel[row_height:2 * row_height, x0:x0 + CELL_WIDTH] = corner_cell

    return panel

def main():
    metadata = read_video_metadata(VIDEO_PATH)

    print("\n========== TEMPORAL CORNER-WISE COLOR ==========")
    print(f"Video      : {VIDEO_PATH}")
    print(f"Resolution : {metadata.width}x{metadata.height}")
    print(f"FPS        : {metadata.fps:.3f}")
    print(f"Window     : {WINDOW_SIZE} observations")
    print("Rows       : RAW / CORNER")
    print(
        f"Selection  : up to {PANELS_PER_TRACK} panels per track by RAW variability"
    )
    print(
        f"Min gap    : {MIN_PANEL_CENTER_GAP_FRAMES} center frames between panels"
    )
    print("==========================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)

    for old_panel in PANELS_DIR.glob("*.png"):
        old_panel.unlink()

    window_builder = SlidingTrackWindowBuilder(WINDOW_SIZE)

    tracks = []
    next_track_id = 0

    tracker_config = PlateTrackerConfig(
        max_frame_gap=3 * PROCESS_EVERY_N_FRAMES,
    )

    # Store lightweight candidate descriptors first.
    # We will select up to PANELS_PER_TRACK non-adjacent windows per track
    # AFTER scanning the video, based only on RAW variability.
    raw_candidates_by_track = {}

    candidate_rows = []
    valid_window_count = 0
    evaluated_corner_count = 0

    for sampled_frame in sample_video_frames(
        VIDEO_PATH,
        every_n_frames=PROCESS_EVERY_N_FRAMES,
        max_sampled_frames=MAX_FRAMES,
    ):
        print(f"Processing frame {sampled_frame.frame_index}")

        detections = process_sampled_frame(
            sampled_frame,
            API_KEY,
            video_name=VIDEO_PATH.name,
            use_cached_json=USE_CACHED_ROBOFLOW_JSON,
            min_confidence=MIN_CONFIDENCE,
            min_detection_area=MIN_DETECTION_AREA,
            min_width=MIN_WIDTH,
            min_height=MIN_HEIGHT,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            refinement_method=REFINEMENT_METHOD,
            refinement_debug=REFINEMENT_DEBUG,
            refinement_panel_debug=REFINEMENT_PANEL_DEBUG,
        )

        accepted = get_accepted_detections(detections)

        tracks, next_track_id = associate_frame_detections(
            accepted,
            tracks,
            next_track_id=next_track_id,
            config=tracker_config,
        )

        for detection in accepted:
            window = window_builder.add(
                image=sampled_frame.image,
                detection=detection,
            )

            if window is None:
                continue

            if not window_has_full_visibility(window):
                continue

            geom = run_geometry_baseline(window)

            if not geom.consensus.success:
                continue

            aligned = aligned_raw_quads(
                window,
                geom.H_to_reference,
            )

            if aligned is None:
                continue

            raw_var, corner_var = raw_variability_metrics(
                aligned
            )

            valid_window_count += 1

            raw_candidates_by_track.setdefault(
                window.track_id,
                []
            ).append(
                {
                    "window": window,
                    "geom": geom,
                    "raw_var": raw_var,
                    "corner_var": corner_var.copy(),
                }
            )

    # ------------------------------------------------------------------
    # Select up to two diverse hard windows PER TRACK.
    # Selection is based ONLY on RAW variability.
    # Windows from the same track must be separated by at least
    # MIN_PANEL_CENTER_GAP_FRAMES at their center frames.
    # ------------------------------------------------------------------
    selected_descriptors = []

    for track_id, track_candidates in sorted(
        raw_candidates_by_track.items()
    ):
        ranked = sorted(
            track_candidates,
            key=lambda item: item["raw_var"],
            reverse=True,
        )

        chosen = []

        for item in ranked:
            center_frame = item["window"].center.frame_index

            if any(
                abs(
                    center_frame
                    - other["window"].center.frame_index
                )
                < MIN_PANEL_CENTER_GAP_FRAMES
                for other in chosen
            ):
                continue

            chosen.append(item)

            if len(chosen) >= PANELS_PER_TRACK:
                break

        selected_descriptors.extend(chosen)

    # Run the final corner-wise method only on the selected windows.
    selected = []

    for item in selected_descriptors:
        window = item["window"]
        geom = item["geom"]
        raw_var = item["raw_var"]
        corner_var = item["corner_var"]

        corner_result = run_corner_color(
            window,
            geom.H_to_reference,
            config=CORNER_CONFIG,
        )

        if not corner_result.success:
            continue

        evaluated_corner_count += 1

        corner_output = propagate_reference_quad(
            window,
            geom.H_to_reference,
            corner_result.corner_quad_in_reference,
        )

        panel = render_panel(
            window,
            corner_output,
            corner_result.corner_source_frames,
        )

        summary = {
            "track_id": window.track_id,
            "start_frame": window.start_frame,
            "center_frame": window.center.frame_index,
            "end_frame": window.end_frame,
            "raw_variability": raw_var,
            "raw_tl_variability": float(corner_var[0]),
            "raw_tr_variability": float(corner_var[1]),
            "raw_br_variability": float(corner_var[2]),
            "raw_bl_variability": float(corner_var[3]),
            "corner_tl_source_frame": corner_result.corner_source_frames[0],
            "corner_tr_source_frame": corner_result.corner_source_frames[1],
            "corner_br_source_frame": corner_result.corner_source_frames[2],
            "corner_bl_source_frame": corner_result.corner_source_frames[3],
            "corner_tl_score": corner_result.corner_scores[0],
            "corner_tr_score": corner_result.corner_scores[1],
            "corner_br_score": corner_result.corner_scores[2],
            "corner_bl_score": corner_result.corner_scores[3],
            "corner_joint_score": corner_result.joint_score,
            "corner_mixed_source_count": corner_result.mixed_source_count,
        }

        for candidate in corner_result.candidates:
            candidate_rows.append(
                {
                    "track_id": window.track_id,
                    "start_frame": window.start_frame,
                    "center_frame": window.center.frame_index,
                    "end_frame": window.end_frame,
                    "raw_variability": raw_var,
                    "corner_index": candidate.corner_index,
                    "corner_name": CORNER_NAMES[candidate.corner_index],
                    "source_index": candidate.source_index,
                    "source_frame": candidate.source_frame,
                    "aggregate_score": candidate.aggregate_score,
                    "is_corner_winner": int(
                        corner_result.corner_source_indices[candidate.corner_index]
                        == candidate.source_index
                    ),
                }
            )

        selected.append({"summary": summary, "panel": panel})

        print(
            f"  SELECTED | Track {window.track_id} | "
            f"{window.start_frame}->{window.end_frame} | "
            f"RAW var={raw_var:.5f} | "
            f"corners={corner_result.corner_source_frames}"
        )

    # Stable presentation order: track, then center frame.
    selected.sort(
        key=lambda item: (
            item["summary"]["track_id"],
            item["summary"]["center_frame"],
        )
    )

    rows = []

    for rank, item in enumerate(selected, start=1):
        summary = item["summary"]
        panel = item["panel"]

        panel_path = (
            PANELS_DIR
            / (
                f"rank_{rank:02d}_"
                f"track_{int(summary['track_id']):03d}_"
                f"f{int(summary['start_frame']):06d}_"
                f"{int(summary['end_frame']):06d}_"
                f"rawvar_{float(summary['raw_variability']):.5f}.png"
            )
        )

        if not cv2.imwrite(str(panel_path), panel):
            raise RuntimeError(
                f"Could not save panel: {panel_path}"
            )

        rows.append(
            {
                "rank": rank,
                **summary,
                "panel_path": str(panel_path),
            }
        )

    fieldnames = [
        "rank",
        "track_id",
        "start_frame",
        "center_frame",
        "end_frame",

        "raw_variability",
        "raw_tl_variability",
        "raw_tr_variability",
        "raw_br_variability",
        "raw_bl_variability",



        "corner_tl_source_frame",
        "corner_tr_source_frame",
        "corner_br_source_frame",
        "corner_bl_source_frame",

        "corner_tl_score",
        "corner_tr_score",
        "corner_br_score",
        "corner_bl_score",

        "corner_joint_score",
        "corner_mixed_source_count",

        "panel_path",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    candidate_fieldnames = [
        "track_id",
        "start_frame",
        "center_frame",
        "end_frame",
        "raw_variability",
        "corner_index",
        "corner_name",
        "source_index",
        "source_frame",
        "aggregate_score",
        "is_corner_winner",
    ]

    with CANDIDATE_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=candidate_fieldnames,
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    print("\n========== COMPLETE ==========")
    print(f"Valid windows          : {valid_window_count}")
    print(f"Corner windows evaluated: {evaluated_corner_count}")
    print(f"Saved diverse panels   : {len(rows)}")
    print(f"Panels dir             : {PANELS_DIR}")
    print(f"Summary CSV            : {CSV_PATH}")
    print(f"Candidate CSV          : {CANDIDATE_CSV_PATH}")
    print("==============================\n")


if __name__ == "__main__":
    main()
