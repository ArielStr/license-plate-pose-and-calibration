from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from multi_plate_new.homographies import (
    compute_plate_homography,
)


@dataclass(frozen=True)
class CornerOptimizationConfig:
    """
    Configuration for moving detected plate corners so that their
    homographies become consistent with a known reference calibration.
    """

    # Maximum movement allowed for every x/y coordinate.
    max_shift_px: float = 10.0

    # Penalizes corner movement.
    #
    # Larger value:
    #   less corner movement, weaker improvement in constraints.
    #
    # Smaller value:
    #   more freedom to move corners.
    movement_weight: float = 1e-3

    max_function_evaluations: int = 2_000

    verbose: int = 1


@dataclass
class PlateCornerOptimization:
    plate_index: int
    original_corners: np.ndarray
    optimized_corners: np.ndarray
    shifts: np.ndarray
    shift_distances: np.ndarray

    initial_constraint_residuals: np.ndarray
    final_constraint_residuals: np.ndarray


@dataclass
class CornerOptimizationResult:
    success: bool
    message: str

    initial_cost: float
    final_cost: float

    initial_constraint_rms: float
    final_constraint_rms: float

    mean_corner_shift_px: float
    max_corner_shift_px: float

    optimized_refinement_results: list[dict]
    plate_results: list[PlateCornerOptimization]

    scipy_result: object


def compute_reference_B(K_ref: np.ndarray) -> np.ndarray:
    """
    Compute:

        B_ref = K_ref^{-T} K_ref^{-1}

    This is the matrix that every valid plate homography should satisfy.
    """
    K_ref = np.asarray(K_ref, dtype=np.float64)

    if K_ref.shape != (3, 3):
        raise ValueError(
            f"K_ref must have shape (3, 3), received {K_ref.shape}"
        )

    if not np.all(np.isfinite(K_ref)):
        raise ValueError("K_ref contains non-finite values")

    K_inverse = np.linalg.inv(K_ref)

    B_ref = K_inverse.T @ K_inverse

    # Numerical symmetry.
    return 0.5 * (B_ref + B_ref.T)


def normalize_homography(H: np.ndarray) -> np.ndarray:
    """
    Normalize a homography to remove its arbitrary scale.
    """
    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError(
            f"H must have shape (3, 3), received {H.shape}"
        )

    if not np.all(np.isfinite(H)):
        raise ValueError("Homography contains non-finite values")

    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]

    norm = np.linalg.norm(H)

    if norm <= 1e-12:
        raise ValueError("Degenerate homography")

    return H / norm


def compute_zhang_residuals_for_H(
    H: np.ndarray,
    B_ref: np.ndarray,
) -> np.ndarray:
    """
    Compute the two normalized Zhang constraints for one homography.

    Ideal values:

        h1^T B h2 = 0

        h1^T B h1 = h2^T B h2
    """
    H = normalize_homography(H)

    h1 = H[:, 0]
    h2 = H[:, 1]

    h1_B_h1 = float(h1.T @ B_ref @ h1)
    h2_B_h2 = float(h2.T @ B_ref @ h2)
    h1_B_h2 = float(h1.T @ B_ref @ h2)

    epsilon = 1e-12

    orthogonality_denominator = np.sqrt(
        max(
            abs(h1_B_h1 * h2_B_h2),
            epsilon,
        )
    )

    equal_norm_denominator = max(
        abs(h1_B_h1) + abs(h2_B_h2),
        epsilon,
    )

    orthogonality_residual = (
        h1_B_h2 / orthogonality_denominator
    )

    equal_norm_residual = (
        (h1_B_h1 - h2_B_h2)
        / equal_norm_denominator
    )

    return np.array(
        [
            orthogonality_residual,
            equal_norm_residual,
        ],
        dtype=np.float64,
    )


def extract_successful_refinements(
    refinement_results: list[dict],
) -> list[dict]:
    """
    Return only successful refinement results containing valid corners.
    """
    successful: list[dict] = []

    for result in refinement_results:
        if not result.get("success", False):
            continue

        if "image_points" not in result:
            continue

        corners = np.asarray(
            result["image_points"],
            dtype=np.float64,
        )

        if corners.shape != (4, 2):
            continue

        if not np.all(np.isfinite(corners)):
            continue

        successful.append(result)

    return successful


def pack_original_corners(
    successful_results: list[dict],
) -> np.ndarray:
    """
    Return all corners in shape:

        (number_of_plates, 4, 2)
    """
    return np.stack(
        [
            np.asarray(
                result["image_points"],
                dtype=np.float64,
            )
            for result in successful_results
        ],
        axis=0,
    )


def shifts_vector_to_array(
    shift_vector: np.ndarray,
    num_plates: int,
) -> np.ndarray:
    """
    Convert the flat optimizer vector into:

        (number_of_plates, 4, 2)
    """
    return np.asarray(
        shift_vector,
        dtype=np.float64,
    ).reshape(num_plates, 4, 2)


def compute_constraint_residuals(
    corners_per_plate: np.ndarray,
    B_ref: np.ndarray,
) -> np.ndarray:
    """
    Compute two Zhang residuals for every plate.
    """
    residuals: list[float] = []

    for corners in corners_per_plate:
        try:
            H = compute_plate_homography(corners)

            if H is None:
                return np.full(
                    2 * len(corners_per_plate),
                    1e3,
                    dtype=np.float64,
                )

            plate_residuals = compute_zhang_residuals_for_H(
                H,
                B_ref,
            )

        except (ValueError, np.linalg.LinAlgError):
            plate_residuals = np.array(
                [1e3, 1e3],
                dtype=np.float64,
            )

        residuals.extend(plate_residuals.tolist())

    return np.asarray(residuals, dtype=np.float64)


def constraint_rms(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)

    if residuals.size == 0:
        return float("nan")

    return float(
        np.sqrt(
            np.mean(residuals ** 2)
        )
    )


def optimize_corners_to_reference(
    refinement_results: list[dict],
    K_ref: np.ndarray,
    config: CornerOptimizationConfig | None = None,
) -> CornerOptimizationResult:
    """
    Move all plate corners simultaneously so that the resulting
    homographies satisfy the Zhang constraints of K_ref.

    The objective contains:

        1. Zhang constraint residuals relative to K_ref.
        2. A penalty on corner movement.
    """
    if config is None:
        config = CornerOptimizationConfig()

    successful_results = extract_successful_refinements(
        refinement_results
    )

    if len(successful_results) < 3:
        raise ValueError(
            "At least three successful plate refinements are required"
        )

    B_ref = compute_reference_B(K_ref)

    original_corners = pack_original_corners(
        successful_results
    )

    num_plates = len(successful_results)
    num_variables = num_plates * 4 * 2

    initial_shift_vector = np.zeros(
        num_variables,
        dtype=np.float64,
    )

    sqrt_movement_weight = np.sqrt(
        config.movement_weight
    )

    def residual_function(
        shift_vector: np.ndarray,
    ) -> np.ndarray:
        shifts = shifts_vector_to_array(
            shift_vector,
            num_plates,
        )

        candidate_corners = (
            original_corners + shifts
        )

        geometric_residuals = compute_constraint_residuals(
            candidate_corners,
            B_ref,
        )

        movement_residuals = (
            sqrt_movement_weight
            * shift_vector
        )

        return np.concatenate(
            [
                geometric_residuals,
                movement_residuals,
            ]
        )

    def numerical_jacobian(
            shift_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Central finite-difference Jacobian with an absolute pixel step.

        We intentionally use a visible subpixel step because the homography
        implementation may internally convert points to float32.
        """
        step_px = 0.1

        base_residuals = residual_function(
            shift_vector
        )

        jacobian = np.zeros(
            (
                base_residuals.size,
                shift_vector.size,
            ),
            dtype=np.float64,
        )

        for variable_index in range(
                shift_vector.size
        ):
            positive = shift_vector.copy()
            negative = shift_vector.copy()

            positive[variable_index] += step_px
            negative[variable_index] -= step_px

            residual_positive = residual_function(
                positive
            )

            residual_negative = residual_function(
                negative
            )

            jacobian[:, variable_index] = (
                                                  residual_positive
                                                  - residual_negative
                                          ) / (2.0 * step_px)

        return jacobian

    lower_bounds = np.full(
        num_variables,
        -config.max_shift_px,
        dtype=np.float64,
    )

    upper_bounds = np.full(
        num_variables,
        config.max_shift_px,
        dtype=np.float64,
    )

    initial_geometric_residuals = (
        compute_constraint_residuals(
            original_corners,
            B_ref,
        )
    )
    debug_shift = initial_shift_vector.copy()

    # Move x of corner 0 in plate 0 by one pixel.
    debug_shift[0] = 1.0

    residual_at_zero = residual_function(
        initial_shift_vector
    )

    residual_after_one_pixel = residual_function(
        debug_shift
    )

    print("\n========== OPTIMIZATION SANITY CHECK ==========")
    print("Residual at zero:")
    print(residual_at_zero)

    print("\nResidual after moving one coordinate by 1 px:")
    print(residual_after_one_pixel)

    print(
        "\nResidual difference norm:",
        np.linalg.norm(
            residual_after_one_pixel
            - residual_at_zero
        ),
    )
    print("================================================\n")
    optimization = least_squares(
        residual_function,
        x0=initial_shift_vector,
        jac=numerical_jacobian,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        max_nfev=config.max_function_evaluations,
        verbose=config.verbose,
    )

    optimized_shifts = shifts_vector_to_array(
        optimization.x,
        num_plates,
    )

    optimized_corners = (
        original_corners + optimized_shifts
    )

    final_geometric_residuals = (
        compute_constraint_residuals(
            optimized_corners,
            B_ref,
        )
    )

    shift_distances = np.linalg.norm(
        optimized_shifts,
        axis=2,
    )

    optimized_refinement_results = deepcopy(
        refinement_results
    )

    plate_results: list[PlateCornerOptimization] = []

    successful_position = 0

    for result in optimized_refinement_results:
        if not result.get("success", False):
            continue

        if "image_points" not in result:
            continue

        current_corners = np.asarray(
            result["image_points"],
            dtype=np.float64,
        )

        if current_corners.shape != (4, 2):
            continue

        plate_index = int(
            result.get(
                "index",
                successful_position,
            )
        )

        result["original_image_points"] = (
            original_corners[
                successful_position
            ].copy()
        )

        result["image_points"] = (
            optimized_corners[
                successful_position
            ].copy()
        )

        result["corner_optimization_shifts"] = (
            optimized_shifts[
                successful_position
            ].copy()
        )

        result["corner_optimization_distances"] = (
            shift_distances[
                successful_position
            ].copy()
        )

        residual_start = 2 * successful_position
        residual_end = residual_start + 2

        plate_results.append(
            PlateCornerOptimization(
                plate_index=plate_index,
                original_corners=original_corners[
                    successful_position
                ].copy(),
                optimized_corners=optimized_corners[
                    successful_position
                ].copy(),
                shifts=optimized_shifts[
                    successful_position
                ].copy(),
                shift_distances=shift_distances[
                    successful_position
                ].copy(),
                initial_constraint_residuals=(
                    initial_geometric_residuals[
                        residual_start:residual_end
                    ].copy()
                ),
                final_constraint_residuals=(
                    final_geometric_residuals[
                        residual_start:residual_end
                    ].copy()
                ),
            )
        )

        successful_position += 1

    return CornerOptimizationResult(
        success=bool(optimization.success),
        message=str(optimization.message),
        initial_cost=float(
            np.sum(
                residual_function(
                    initial_shift_vector
                ) ** 2
            )
        ),
        final_cost=float(
            np.sum(
                residual_function(
                    optimization.x
                ) ** 2
            )
        ),
        initial_constraint_rms=constraint_rms(
            initial_geometric_residuals
        ),
        final_constraint_rms=constraint_rms(
            final_geometric_residuals
        ),
        mean_corner_shift_px=float(
            np.mean(shift_distances)
        ),
        max_corner_shift_px=float(
            np.max(shift_distances)
        ),
        optimized_refinement_results=(
            optimized_refinement_results
        ),
        plate_results=plate_results,
        scipy_result=optimization,
    )


def print_corner_optimization_result(
    result: CornerOptimizationResult,
) -> None:
    print(
        "\n========== CORNER OPTIMIZATION =========="
    )

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    print(
        f"\nInitial objective cost: "
        f"{result.initial_cost:.8f}"
    )

    print(
        f"Final objective cost  : "
        f"{result.final_cost:.8f}"
    )

    print(
        f"\nInitial constraint RMS: "
        f"{result.initial_constraint_rms:.8f}"
    )

    print(
        f"Final constraint RMS  : "
        f"{result.final_constraint_rms:.8f}"
    )

    print(
        f"\nMean corner shift: "
        f"{result.mean_corner_shift_px:.3f} px"
    )

    print(
        f"Maximum corner shift: "
        f"{result.max_corner_shift_px:.3f} px"
    )

    for plate in result.plate_results:
        print(
            f"\nPlate {plate.plate_index}"
        )

        print(
            "Initial Zhang residuals: "
            f"{plate.initial_constraint_residuals}"
        )

        print(
            "Final Zhang residuals  : "
            f"{plate.final_constraint_residuals}"
        )

        for corner_index in range(4):
            original = plate.original_corners[
                corner_index
            ]

            optimized = plate.optimized_corners[
                corner_index
            ]

            shift = plate.shifts[
                corner_index
            ]

            distance = plate.shift_distances[
                corner_index
            ]

            print(
                f"  corner {corner_index}: "
                f"original={original}, "
                f"optimized={optimized}, "
                f"shift={shift}, "
                f"distance={distance:.3f} px"
            )

    print(
        "=========================================\n"
    )