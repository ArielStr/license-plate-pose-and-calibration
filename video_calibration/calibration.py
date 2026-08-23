from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from multi_plate_new.intrinsics import (
    estimate_B_from_homographies,
    compute_K_from_B,
)
from video_calibration.observation import PlateObservation


@dataclass(frozen=True)
class CalibrationPipelineConfig:
    """
    Configuration for iterative Zhang calibration.

    Each planar homography contributes two Zhang constraints.
    Therefore at least three homographies are required for the
    six unknown entries of the symmetric matrix B.
    """

    min_observations: int = 3

    # Stop when every observation has a normalized residual below
    # this value.
    max_observation_residual: float = 0.08

    # Avoid removing an excessive number of observations.
    max_rejection_fraction: float = 0.35

    # Protect against endless loops.
    max_iterations: int = 20


@dataclass
class CalibrationIteration:
    iteration_index: int
    num_observations: int
    K_est: Optional[np.ndarray]
    B: np.ndarray
    singular_values: np.ndarray
    condition_number: float
    residuals: list[float]
    worst_local_index: int
    worst_residual: float
    removed_frame_index: Optional[int] = None


@dataclass
class CalibrationResult:
    success: bool
    K_est: Optional[np.ndarray]

    used_observations: list[PlateObservation]
    rejected_observations: list[PlateObservation]

    iterations: list[CalibrationIteration]
    failure_reason: Optional[str] = None


def observation_to_homography_dict(
    observation: PlateObservation,
) -> dict:
    """
    Convert PlateObservation into the dictionary format expected by the
    existing Zhang implementation.
    """
    return {
        "H": observation.homography,
        "frame_index": observation.frame_index,
        "timestamp_sec": observation.timestamp_sec,
    }


def observations_to_homography_dicts(
    observations: list[PlateObservation],
) -> list[dict]:
    return [
        observation_to_homography_dict(observation)
        for observation in observations
    ]


def _v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Build one Zhang constraint vector.

    This is intentionally kept local to the video-calibration package so
    that residual computation does not depend on a private helper from the
    existing implementation.
    """
    return np.array(
        [
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j],
        ],
        dtype=np.float64,
    )


def B_to_vector(B: np.ndarray) -> np.ndarray:
    """
    Convert the symmetric matrix B into Zhang's six-element vector.
    """
    return np.array(
        [
            B[0, 0],
            B[0, 1],
            B[1, 1],
            B[0, 2],
            B[1, 2],
            B[2, 2],
        ],
        dtype=np.float64,
    )


def normalized_constraint_residual(
    constraint: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute a scale-invariant algebraic residual:

        |v^T b| / (||v|| ||b||)

    The result is dimensionless and lies approximately in [0, 1].
    """
    denominator = (
        np.linalg.norm(constraint)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-15:
        return float("inf")

    return float(
        abs(np.dot(constraint, b)) / denominator
    )


def compute_observation_zhang_residual(
    observation: PlateObservation,
    B: np.ndarray,
) -> float:
    """
    Return one residual score for a single homography.

    Each homography contributes two constraints:

        h1^T B h2 = 0
        h1^T B h1 - h2^T B h2 = 0

    We combine them using root-mean-square.
    """
    H = observation.homography.astype(np.float64)

    if abs(H[2, 2]) <= 1e-15:
        return float("inf")

    H = H / H[2, 2]

    b = B_to_vector(B)

    v12 = _v_ij(H, 0, 1)
    v11_minus_v22 = (
        _v_ij(H, 0, 0)
        - _v_ij(H, 1, 1)
    )

    residual_orthogonality = normalized_constraint_residual(
        v12,
        b,
    )

    residual_equal_norm = normalized_constraint_residual(
        v11_minus_v22,
        b,
    )

    return float(
        np.sqrt(
            0.5
            * (
                residual_orthogonality ** 2
                + residual_equal_norm ** 2
            )
        )
    )


def compute_all_residuals(
    observations: list[PlateObservation],
    B: np.ndarray,
) -> list[float]:
    return [
        compute_observation_zhang_residual(
            observation,
            B,
        )
        for observation in observations
    ]


def analyze_constraint_matrix(
    V: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Return singular values and a diagnostic condition number.

    Since the Zhang system is homogeneous, the smallest singular value
    corresponds to the null-space direction. For diagnostics, we compare
    the largest singular value with the second-smallest one.
    """
    singular_values = np.linalg.svd(
        V,
        compute_uv=False,
    )

    if len(singular_values) < 2:
        return singular_values, float("inf")

    denominator = singular_values[-2]

    if denominator <= 1e-15:
        condition_number = float("inf")
    else:
        condition_number = float(
            singular_values[0] / denominator
        )

    return singular_values, condition_number


def estimate_calibration_once(
    observations: list[PlateObservation],
) -> tuple[
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Estimate B and K once from the supplied observations.
    """
    homographies = observations_to_homography_dicts(
        observations
    )

    B, V, _ = estimate_B_from_homographies(
        homographies
    )

    K_est = compute_K_from_B(B)

    singular_values, condition_number = (
        analyze_constraint_matrix(V)
    )

    return (
        K_est,
        B,
        V,
        singular_values,
        condition_number,
    )


def run_iterative_calibration(
    observations: list[PlateObservation],
    config: CalibrationPipelineConfig | None = None,
) -> CalibrationResult:
    """
    Run Zhang calibration with iterative worst-outlier rejection.
    """
    if config is None:
        config = CalibrationPipelineConfig()

    working_observations = [
        observation
        for observation in observations
        if observation.accepted
    ]

    if len(working_observations) < config.min_observations:
        return CalibrationResult(
            success=False,
            K_est=None,
            used_observations=working_observations,
            rejected_observations=[],
            iterations=[],
            failure_reason=(
                "not_enough_accepted_observations"
            ),
        )

    initial_count = len(working_observations)

    max_rejections = int(
        np.floor(
            initial_count
            * config.max_rejection_fraction
        )
    )

    rejected_observations: list[PlateObservation] = []
    iterations: list[CalibrationIteration] = []

    final_K: Optional[np.ndarray] = None

    for iteration_index in range(config.max_iterations):
        if len(working_observations) < config.min_observations:
            return CalibrationResult(
                success=False,
                K_est=None,
                used_observations=working_observations,
                rejected_observations=rejected_observations,
                iterations=iterations,
                failure_reason=(
                    "too_few_observations_after_rejection"
                ),
            )

        (
            K_est,
            B,
            _,
            singular_values,
            condition_number,
        ) = estimate_calibration_once(
            working_observations
        )

        residuals = compute_all_residuals(
            working_observations,
            B,
        )

        residual_array = np.asarray(
            residuals,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(residual_array)):
            worst_local_index = int(
                np.argmax(
                    np.where(
                        np.isfinite(residual_array),
                        residual_array,
                        np.inf,
                    )
                )
            )
        else:
            worst_local_index = int(
                np.argmax(residual_array)
            )

        worst_residual = float(
            residual_array[worst_local_index]
        )

        iteration = CalibrationIteration(
            iteration_index=iteration_index,
            num_observations=len(working_observations),
            K_est=K_est,
            B=B,
            singular_values=singular_values,
            condition_number=condition_number,
            residuals=residuals,
            worst_local_index=worst_local_index,
            worst_residual=worst_residual,
        )

        iterations.append(iteration)

        # Successful stopping condition.
        if (
            K_est is not None
            and worst_residual
            <= config.max_observation_residual
        ):
            final_K = K_est
            break

        # Do not remove observations just because B is not positive definite.
        if worst_residual <= config.max_observation_residual:
            return CalibrationResult(
                success=False,
                K_est=None,
                used_observations=working_observations,
                rejected_observations=rejected_observations,
                iterations=iterations,
                failure_reason=(
                    "constraints_fit_but_K_extraction_failed"
                ),
            )
        can_remove_more = (
            len(rejected_observations)
            < max_rejections
        )

        enough_will_remain = (
            len(working_observations) - 1
            >= config.min_observations
        )

        if not can_remove_more:
            return CalibrationResult(
                success=False,
                K_est=K_est,
                used_observations=working_observations,
                rejected_observations=rejected_observations,
                iterations=iterations,
                failure_reason=(
                    "maximum_rejection_fraction_reached"
                ),
            )

        if not enough_will_remain:
            return CalibrationResult(
                success=False,
                K_est=K_est,
                used_observations=working_observations,
                rejected_observations=rejected_observations,
                iterations=iterations,
                failure_reason=(
                    "cannot_reject_without_falling_below_minimum"
                ),
            )

        removed_observation = working_observations.pop(
            worst_local_index
        )

        removed_observation.reject(
            "zhang_constraint_outlier"
        )

        rejected_observations.append(
            removed_observation
        )

        iteration.removed_frame_index = (
            removed_observation.frame_index
        )

    if final_K is None:
        return CalibrationResult(
            success=False,
            K_est=None,
            used_observations=working_observations,
            rejected_observations=rejected_observations,
            iterations=iterations,
            failure_reason="maximum_iterations_reached",
        )

    return CalibrationResult(
        success=True,
        K_est=final_K,
        used_observations=working_observations,
        rejected_observations=rejected_observations,
        iterations=iterations,
        failure_reason=None,
    )


def print_calibration_result(
    result: CalibrationResult,
) -> None:
    print("\n========== VIDEO CALIBRATION ==========")
    print(f"Success              : {result.success}")
    print(
        f"Used observations    : "
        f"{len(result.used_observations)}"
    )
    print(
        f"Rejected observations: "
        f"{len(result.rejected_observations)}"
    )

    if result.failure_reason is not None:
        print(
            f"Failure reason       : "
            f"{result.failure_reason}"
        )

    print("\nIterations:")

    for iteration in result.iterations:
        removed_text = ""

        if iteration.removed_frame_index is not None:
            removed_text = (
                f", removed frame="
                f"{iteration.removed_frame_index}"
            )

        print(
            f"  iteration={iteration.iteration_index:2d}, "
            f"observations={iteration.num_observations:2d}, "
            f"worst_residual={iteration.worst_residual:.6f}, "
            f"condition={iteration.condition_number:.3e}"
            f"{removed_text}"
        )

    if result.K_est is not None:
        print("\nEstimated K:")
        print(result.K_est)

    if result.rejected_observations:
        print("\nRejected observations:")

        for observation in result.rejected_observations:
            print(
                f"  frame={observation.frame_index}, "
                f"time={observation.timestamp_sec:.3f}, "
                f"quality={observation.quality_score:.3f}"
            )

    print("=======================================\n")