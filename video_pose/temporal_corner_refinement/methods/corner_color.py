from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from ..alignment import warp_quad
from ..window import TrackWindow


@dataclass(frozen=True)
class CornerColorConfig:
    color_segment_samples: int = 18
    local_segment_fraction: float = 0.30

    color_inner_offset_px: float = 4.0
    color_outer_offset_px: float = 4.0
    color_patch_radius_px: int = 1
    color_reference_shrink: float = 0.45
    color_sigma_lab: float = 24.0
    min_valid_color_frames: int = 4

    # Hard geometry guards.
    min_quad_area_px2: float = 50.0
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 7.5

    # Soft joint-selection regularization.
    side_length_penalty_weight: float = 0.10
    area_penalty_weight: float = 0.05
    aspect_penalty_weight: float = 0.08


@dataclass(frozen=True)
class CornerCandidateScore:
    corner_index: int
    source_index: int
    source_frame: int
    aggregate_score: float


@dataclass(frozen=True)
class CornerColorResult:
    success: bool
    reason: str

    corner_source_indices: tuple[int | None, int | None, int | None, int | None]
    corner_source_frames: tuple[int | None, int | None, int | None, int | None]
    corner_scores: tuple[float | None, float | None, float | None, float | None]

    corner_quad_in_reference: np.ndarray | None
    joint_score: float | None
    mixed_source_count: int

    candidates: tuple[CornerCandidateScore, ...]


@dataclass(frozen=True)
class _PreparedObservationColor:
    lab_image: np.ndarray
    ref_lab: np.ndarray | None


def _shrink_quad(corners: np.ndarray, fraction: float) -> np.ndarray:
    q = np.asarray(corners, dtype=np.float64)
    center = np.mean(q, axis=0)
    return center + fraction * (q - center)


def _prepare_observation_color(
    image: np.ndarray,
    raw_quad: np.ndarray,
    *,
    config: CornerColorConfig,
) -> _PreparedObservationColor:
    # Expensive conversion is done ONCE per observation.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    inner = _shrink_quad(raw_quad, config.color_reference_shrink)
    poly = np.round(inner).astype(np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)

    pixels = lab[mask > 0]
    ref_lab = None

    if len(pixels) >= 20:
        ref_lab = np.median(pixels, axis=0).astype(np.float64)

    return _PreparedObservationColor(
        lab_image=lab,
        ref_lab=ref_lab,
    )


def _lab_similarity(
    prepared: _PreparedObservationColor,
    point: np.ndarray,
    *,
    config: CornerColorConfig,
) -> float | None:
    if prepared.ref_lab is None:
        return None

    lab = prepared.lab_image
    h, w = lab.shape[:2]

    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    r = config.color_patch_radius_px

    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)

    if x2 <= x1 or y2 <= y1:
        return None

    patch = lab[y1:y2, x1:x2].reshape(-1, 3).astype(np.float64)

    if len(patch) == 0:
        return None

    color = np.median(patch, axis=0)
    dist = float(np.linalg.norm(color - prepared.ref_lab))

    return float(
        np.exp(
            -0.5 * (dist / config.color_sigma_lab) ** 2
        )
    )


def _local_edge_score_from_corner(
    prepared: _PreparedObservationColor,
    quad: np.ndarray,
    corner_index: int,
    neighbor_index: int,
    *,
    config: CornerColorConfig,
) -> float | None:
    """
    Score only a SHORT boundary segment starting at corner_index and moving
    toward neighbor_index. This is what allows TR and BR on the same physical
    edge to be judged independently.
    """
    if prepared.ref_lab is None:
        return None

    q = np.asarray(quad, dtype=np.float64)
    center = np.mean(q, axis=0)

    a = q[corner_index]
    b = q[neighbor_index]

    edge = b - a
    edge_len = float(np.linalg.norm(edge))

    if edge_len < 2.0:
        return None

    n = np.array(
        [-edge[1], edge[0]],
        dtype=np.float64,
    )
    n /= max(np.linalg.norm(n), 1e-9)

    midpoint = 0.5 * (a + b)

    # Orient n toward the plate interior.
    if np.dot(center - midpoint, n) < 0:
        n = -n

    ts = np.linspace(
        0.03,
        config.local_segment_fraction,
        config.color_segment_samples,
    )

    values = []

    for t in ts:
        p = (1.0 - t) * a + t * b

        p_in = p + config.color_inner_offset_px * n
        p_out = p - config.color_outer_offset_px * n

        s_in = _lab_similarity(
            prepared,
            p_in,
            config=config,
        )
        s_out = _lab_similarity(
            prepared,
            p_out,
            config=config,
        )

        if s_in is None or s_out is None:
            continue

        values.append(s_in - s_out)

    if not values:
        return None

    return float(np.median(values))


def _corner_score_on_target(
    prepared: _PreparedObservationColor,
    quad: np.ndarray,
    corner_index: int,
    *,
    config: CornerColorConfig,
) -> float | None:
    """
    TL uses TOP-near-TL + LEFT-near-TL
    TR uses TOP-near-TR + RIGHT-near-TR
    BR uses RIGHT-near-BR + BOTTOM-near-BR
    BL uses BOTTOM-near-BL + LEFT-near-BL
    """
    prev_index = (corner_index - 1) % 4
    next_index = (corner_index + 1) % 4

    score_prev = _local_edge_score_from_corner(
        prepared,
        quad,
        corner_index,
        prev_index,
        config=config,
    )
    score_next = _local_edge_score_from_corner(
        prepared,
        quad,
        corner_index,
        next_index,
        config=config,
    )

    if score_prev is None or score_next is None:
        return None

    return float(0.5 * (score_prev + score_next))


def _build_hypotheses_in_reference(
    window: TrackWindow,
    H_to_reference,
) -> list[tuple[int, np.ndarray]]:
    hypotheses = []

    for source_index, obs in enumerate(window.observations):
        H = H_to_reference[source_index]
        if H is None:
            continue

        raw = np.asarray(obs.detection.corners, dtype=np.float64)

        if source_index == window.center_index:
            q_ref = raw.copy()
        else:
            q_ref = warp_quad(raw, H)

        if np.all(np.isfinite(q_ref)):
            hypotheses.append((source_index, q_ref))

    return hypotheses


def _project_reference_quad_to_target(
    q_ref: np.ndarray,
    target_index: int,
    window: TrackWindow,
    H_to_reference,
) -> np.ndarray | None:
    H_target_to_ref = H_to_reference[target_index]

    if H_target_to_ref is None:
        return None

    if target_index == window.center_index:
        return np.asarray(q_ref, dtype=np.float64).copy()

    try:
        H_ref_to_target = np.linalg.inv(H_target_to_ref)
    except np.linalg.LinAlgError:
        return None

    return warp_quad(q_ref, H_ref_to_target)


def _compute_corner_scores(
    window: TrackWindow,
    H_to_reference,
    hypotheses: list[tuple[int, np.ndarray]],
    *,
    config: CornerColorConfig,
) -> tuple[
    dict[int, dict[int, float]],
    list[CornerCandidateScore],
]:
    prepared = []

    for obs in window.observations:
        prepared.append(
            _prepare_observation_color(
                obs.image,
                np.asarray(obs.detection.corners, dtype=np.float64),
                config=config,
            )
        )

    scores: dict[int, dict[int, float]] = {
        0: {},
        1: {},
        2: {},
        3: {},
    }

    candidate_rows: list[CornerCandidateScore] = []

    for source_index, q_ref in hypotheses:
        for corner_index in range(4):
            target_scores = []

            for target_index in range(len(window.observations)):
                q_target = _project_reference_quad_to_target(
                    q_ref,
                    target_index,
                    window,
                    H_to_reference,
                )

                if q_target is None:
                    continue

                score = _corner_score_on_target(
                    prepared[target_index],
                    q_target,
                    corner_index,
                    config=config,
                )

                if score is not None and np.isfinite(score):
                    target_scores.append(float(score))

            if len(target_scores) < config.min_valid_color_frames:
                continue

            aggregate = float(np.median(target_scores))
            scores[corner_index][source_index] = aggregate

            candidate_rows.append(
                CornerCandidateScore(
                    corner_index=int(corner_index),
                    source_index=int(source_index),
                    source_frame=int(
                        window.observations[source_index].frame_index
                    ),
                    aggregate_score=aggregate,
                )
            )

    return scores, candidate_rows


def _quad_area(q: np.ndarray) -> float:
    return abs(
        float(
            cv2.contourArea(
                np.asarray(q, dtype=np.float32)
            )
        )
    )


def _quad_aspect_ratio(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)

    width = 0.5 * (
        np.linalg.norm(q[1] - q[0])
        + np.linalg.norm(q[2] - q[3])
    )
    height = 0.5 * (
        np.linalg.norm(q[3] - q[0])
        + np.linalg.norm(q[2] - q[1])
    )

    return float(width / max(height, 1e-9))


def _side_lengths(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)

    return np.array(
        [
            np.linalg.norm(q[1] - q[0]),
            np.linalg.norm(q[2] - q[1]),
            np.linalg.norm(q[3] - q[2]),
            np.linalg.norm(q[0] - q[3]),
        ],
        dtype=np.float64,
    )


def _geometry_reference(
    hypotheses: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, float, float]:
    side_stack = np.stack(
        [_side_lengths(q) for _, q in hypotheses],
        axis=0,
    )
    ref_sides = np.median(side_stack, axis=0)

    ref_area = float(
        np.median(
            [_quad_area(q) for _, q in hypotheses]
        )
    )
    ref_aspect = float(
        np.median(
            [_quad_aspect_ratio(q) for _, q in hypotheses]
        )
    )

    return ref_sides, ref_area, ref_aspect


def _joint_geometry_penalty(
    q: np.ndarray,
    *,
    ref_sides: np.ndarray,
    ref_area: float,
    ref_aspect: float,
    config: CornerColorConfig,
) -> float | None:
    if not np.all(np.isfinite(q)):
        return None

    area = _quad_area(q)

    if area < config.min_quad_area_px2:
        return None

    if not cv2.isContourConvex(
        np.round(q).astype(np.int32)
    ):
        return None

    aspect = _quad_aspect_ratio(q)

    if (
        aspect < config.min_aspect_ratio
        or aspect > config.max_aspect_ratio
    ):
        return None

    sides = _side_lengths(q)

    if np.any(sides < 2.0):
        return None

    side_penalty = float(
        np.mean(
            np.abs(
                np.log(
                    np.maximum(sides, 1e-6)
                    / np.maximum(ref_sides, 1e-6)
                )
            )
        )
    )

    area_penalty = abs(
        float(
            np.log(
                max(area, 1e-6)
                / max(ref_area, 1e-6)
            )
        )
    )

    aspect_penalty = abs(
        float(
            np.log(
                max(aspect, 1e-6)
                / max(ref_aspect, 1e-6)
            )
        )
    )

    return (
        config.side_length_penalty_weight * side_penalty
        + config.area_penalty_weight * area_penalty
        + config.aspect_penalty_weight * aspect_penalty
    )


def run_corner_color(
    window: TrackWindow,
    H_to_reference,
    *,
    config: CornerColorConfig | None = None,
) -> CornerColorResult:
    if config is None:
        config = CornerColorConfig()

    hypotheses = _build_hypotheses_in_reference(
        window,
        H_to_reference,
    )

    if not hypotheses:
        return CornerColorResult(
            success=False,
            reason="no_valid_hypotheses",
            corner_source_indices=(None, None, None, None),
            corner_source_frames=(None, None, None, None),
            corner_scores=(None, None, None, None),
            corner_quad_in_reference=None,
            joint_score=None,
            mixed_source_count=0,
            candidates=tuple(),
        )

    hypothesis_by_source = {
        source_index: np.asarray(q_ref, dtype=np.float64).copy()
        for source_index, q_ref in hypotheses
    }

    score_map, candidate_rows = _compute_corner_scores(
        window,
        H_to_reference,
        hypotheses,
        config=config,
    )

    if any(len(score_map[c]) == 0 for c in range(4)):
        return CornerColorResult(
            success=False,
            reason="missing_corner_scores",
            corner_source_indices=(None, None, None, None),
            corner_source_frames=(None, None, None, None),
            corner_scores=(None, None, None, None),
            corner_quad_in_reference=None,
            joint_score=None,
            mixed_source_count=0,
            candidates=tuple(candidate_rows),
        )

    ref_sides, ref_area, ref_aspect = _geometry_reference(hypotheses)

    source_choices = [
        list(score_map[c].keys())
        for c in range(4)
    ]

    best_joint_score = -np.inf
    best_sources = None
    best_quad = None

    # Maximum is 7^4 = 2401 combinations for a 7-observation window.
    for sources in product(*source_choices):
        q = np.stack(
            [
                hypothesis_by_source[sources[0]][0],
                hypothesis_by_source[sources[1]][1],
                hypothesis_by_source[sources[2]][2],
                hypothesis_by_source[sources[3]][3],
            ],
            axis=0,
        )

        geometry_penalty = _joint_geometry_penalty(
            q,
            ref_sides=ref_sides,
            ref_area=ref_area,
            ref_aspect=ref_aspect,
            config=config,
        )

        if geometry_penalty is None:
            continue

        appearance_score = float(
            sum(
                score_map[c][sources[c]]
                for c in range(4)
            )
        )

        joint_score = appearance_score - geometry_penalty

        if joint_score > best_joint_score:
            best_joint_score = joint_score
            best_sources = tuple(int(s) for s in sources)
            best_quad = q.copy()

    if best_sources is None or best_quad is None:
        return CornerColorResult(
            success=False,
            reason="no_valid_joint_combination",
            corner_source_indices=(None, None, None, None),
            corner_source_frames=(None, None, None, None),
            corner_scores=(None, None, None, None),
            corner_quad_in_reference=None,
            joint_score=None,
            mixed_source_count=0,
            candidates=tuple(candidate_rows),
        )

    source_frames = tuple(
        int(window.observations[s].frame_index)
        for s in best_sources
    )
    chosen_scores = tuple(
        float(score_map[c][best_sources[c]])
        for c in range(4)
    )

    return CornerColorResult(
        success=True,
        reason="ok",
        corner_source_indices=best_sources,
        corner_source_frames=source_frames,
        corner_scores=chosen_scores,
        corner_quad_in_reference=best_quad,
        joint_score=float(best_joint_score),
        mixed_source_count=len(set(best_sources)),
        candidates=tuple(candidate_rows),
    )
