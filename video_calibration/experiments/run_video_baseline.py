from pathlib import Path
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


from video_calibration.frame_sampler import sample_video_frames
from video_calibration.frame_processor import process_sampled_frame
from video_calibration.calibration import (
    CalibrationPipelineConfig,
    run_iterative_calibration,
    print_calibration_result,
)
from video_calibration.filtering import (
    filter_observations,
    get_accepted_observations,
    print_filter_summary,
    print_observation_geometry,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "cars_photos" / "test_video.mov"

API_KEY = os.getenv(
    "ROBOFLOW_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set."
    )

VIDEO_CALIBRATION_DIR = Path(__file__).resolve().parents[1]

debug_frames_dir = (
    VIDEO_CALIBRATION_DIR
    / "debug"
    / "sampled_frames"
)

debug_frames_dir.mkdir(
    parents=True,
    exist_ok=True,
)
def main() -> None:
    all_observations = []

    for sampled_frame in sample_video_frames(
        VIDEO_PATH,
        every_n_frames=10,
        max_sampled_frames=5,
    ):
        frame_path = (
                debug_frames_dir
                / f"frame_{sampled_frame.frame_index:06d}.jpg"
        )

        cv2.imwrite(
            str(frame_path),
            sampled_frame.image,
        )

        print(f"Saved sampled frame: {frame_path}")
        observations = process_sampled_frame(
            sampled_frame,
            api_key=API_KEY,
            video_name=VIDEO_PATH.stem,
            use_cached_json=True,
            refinement_method="robust_mask_lines",
            refinement_debug=False,
            refinement_panel_debug=True
        )

        all_observations.extend(observations)
    filter_observations(all_observations)

    print_filter_summary(all_observations)
    print_observation_geometry(all_observations)

    accepted_observations = get_accepted_observations(
        all_observations
    )

    print(
        f"Observations ready for selection: "
        f"{len(accepted_observations)}"
    )
    print("\n========== RESULT ==========")
    print(f"Total observations: {len(all_observations)}")

    # for observation in all_observations:
    #     print(
    #         f"Frame={observation.frame_index}, "
    #         f"time={observation.timestamp_sec:.3f}, "
    #         f"confidence={observation.detection_confidence:.3f}, "
    #         f"area={observation.area:.1f}"
    #     )

    calibration_config = CalibrationPipelineConfig(
        min_observations=3,
        max_observation_residual=0.08,
        max_rejection_fraction=0.35,
    )

    # calibration_result = run_iterative_calibration(
    #     accepted_observations,
    #     config=calibration_config,
    # )
    #
    # print_calibration_result(
    #     calibration_result
    # )
    from video_calibration.calibration import (
        estimate_calibration_once,
    )

    (
        K_est,
        B,
        V,
        singular_values,
        condition_number,
    ) = estimate_calibration_once(
        accepted_observations
    )

    print("\n========== SINGLE CALIBRATION ==========")

    print(f"Observations: {len(accepted_observations)}")

    print("\nB:")
    print(B)

    print("\nB eigenvalues:")
    print(np.linalg.eigvalsh(0.5 * (B + B.T)))

    print("\nConstraint singular values:")
    print(singular_values)

    print(f"\nCondition number: {condition_number:.6e}")

    print("\nEstimated K:")
    print(K_est)

    print("========================================")


if __name__ == "__main__":
    main()