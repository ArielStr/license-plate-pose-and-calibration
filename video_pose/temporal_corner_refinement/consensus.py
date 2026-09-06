from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import warp_quad
from .scoring import quad_pair_distance_px
from .window import TrackWindow


@dataclass(frozen=True)
class GeometryCandidate:
    source_index: int
    source_frame_index: int
    transformed_quad_in_reference: np.ndarray
    median_distance_to_others_px: float
    mean_distance_to_others_px: float
    valid_comparisons: int


@dataclass(frozen=True)
class GeometryConsensusResult:
    success: bool
    reason: str
    reference_index: int
    best_source_index: int | None
    best_source_frame_index: int | None
    best_quad_in_reference: np.ndarray | None
    candidates: tuple[GeometryCandidate, ...]


def choose_geometry_consensus(
    window: TrackWindow,
    H_to_reference: list[np.ndarray | None],
    *,
    reference_index: int | None = None,
) -> GeometryConsensusResult:
    """
    Baseline only.

    Every frame's independently refined quad becomes a hypothesis.
    All valid hypotheses are warped into a common reference frame.
    The hypothesis with the smallest median distance to the other hypotheses
    wins.

    IMPORTANT:
    This is deliberately geometry-only and is NOT our final refinement method.
    Its purpose is to establish the common 7-frame experiment infrastructure.
    """
    if reference_index is None:
        reference_index = window.center_index

    transformed: list[tuple[int, np.ndarray]] = []

    for i, observation in enumerate(window.observations):
        H = H_to_reference[i]
        if H is None:
            continue

        if i == reference_index:
            q = np.asarray(
                observation.detection.corners,
                dtype=np.float64,
            )
        else:
            q = warp_quad(observation.detection.corners, H)

        if np.all(np.isfinite(q)):
            transformed.append((i, q))

    if len(transformed) < 3:
        return GeometryConsensusResult(
            False,
            "too_few_aligned_hypotheses",
            reference_index,
            None,
            None,
            None,
            tuple(),
        )

    candidates = []

    for i, quad in transformed:
        distances = [
            quad_pair_distance_px(quad, other)
            for j, other in transformed
            if j != i
        ]

        candidates.append(
            GeometryCandidate(
                source_index=i,
                source_frame_index=window.observations[i].frame_index,
                transformed_quad_in_reference=quad,
                median_distance_to_others_px=float(np.median(distances)),
                mean_distance_to_others_px=float(np.mean(distances)),
                valid_comparisons=len(distances),
            )
        )

    best = min(
        candidates,
        key=lambda c: c.median_distance_to_others_px,
    )

    return GeometryConsensusResult(
        True,
        "ok",
        reference_index,
        best.source_index,
        best.source_frame_index,
        best.transformed_quad_in_reference,
        tuple(candidates),
    )
