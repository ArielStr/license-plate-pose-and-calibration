from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..alignment import (
    AlignmentConfig,
    PairwiseAlignment,
    compose_adjacent_homographies_to_reference,
    estimate_pairwise_homography,
    warp_quad,
)
from ..consensus import GeometryConsensusResult, choose_geometry_consensus
from ..window import TrackWindow


@dataclass(frozen=True)
class GeometryBaselineOutput:
    alignments: tuple[PairwiseAlignment, ...]
    H_to_reference: tuple[np.ndarray | None, ...]
    consensus: GeometryConsensusResult
    output_quads: tuple[np.ndarray, ...]


def run_geometry_baseline(
    window: TrackWindow,
    *,
    alignment_config: AlignmentConfig | None = None,
) -> GeometryBaselineOutput:
    observations = window.observations

    alignments = []
    for i in range(len(observations) - 1):
        alignments.append(
            estimate_pairwise_homography(
                observations[i],
                observations[i + 1],
                config=alignment_config,
            )
        )

    H_to_reference = compose_adjacent_homographies_to_reference(
        alignments,
        reference_index=window.center_index,
    )

    consensus = choose_geometry_consensus(
        window,
        H_to_reference,
        reference_index=window.center_index,
    )

    output_quads = [
        np.asarray(obs.detection.corners, dtype=np.float64).copy()
        for obs in observations
    ]

    # V0 visualization baseline:
    # propagate the winning hypothesis from the common reference system back
    # into every frame. This lets us SEE what a pure geometry-consensus method
    # would do, including where it fails. It is not production refinement.
    if consensus.success and consensus.best_quad_in_reference is not None:
        q_ref = consensus.best_quad_in_reference

        for i, H_i_to_ref in enumerate(H_to_reference):
            if H_i_to_ref is None:
                continue

            if i == window.center_index:
                output_quads[i] = q_ref.copy()
                continue

            try:
                H_ref_to_i = np.linalg.inv(H_i_to_ref)
            except np.linalg.LinAlgError:
                continue

            output_quads[i] = warp_quad(q_ref, H_ref_to_i)

    return GeometryBaselineOutput(
        alignments=tuple(alignments),
        H_to_reference=tuple(H_to_reference),
        consensus=consensus,
        output_quads=tuple(output_quads),
    )
