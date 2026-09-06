from pathlib import Path

import cv2
import imageio.v2 as imageio


PROJECT_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = PROJECT_ROOT / "assets"

MAIN_VIDEO = PROJECT_ROOT / "test_video_pose_temporal_corner_refined.mp4"
ZOOM_VIDEO = PROJECT_ROOT / "test_video_pose_zoom_panel_temporal_corner_refined.mp4"

MAIN_GIF = ASSETS_DIR / "video_pose_demo.gif"
ZOOM_GIF = ASSETS_DIR / "zoom_panel_demo.gif"


def video_to_gif(
    input_path: Path,
    output_path: Path,
    target_fps: float = 10.0,
    target_width: int | None = None,
    start_frame: int = 0,
    playback_speed: float = 1.0,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {input_path}")

    if playback_speed <= 0:
        raise ValueError("playback_speed must be greater than 0")

    if not 0.0 <= crop_top < 1.0:
        raise ValueError("crop_top must be between 0 and 1")

    if not 0.0 <= crop_bottom < 1.0:
        raise ValueError("crop_bottom must be between 0 and 1")

    if crop_top + crop_bottom >= 1.0:
        raise ValueError("crop_top + crop_bottom must be less than 1")

    frame_step = max(1, round(source_fps / target_fps))
    actual_fps = source_fps / frame_step
    playback_fps = actual_fps * playback_speed

    frames = []
    frame_index = 0

    final_width = None
    final_height = None

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index >= start_frame:
            relative_index = frame_index - start_frame

            if relative_index % frame_step == 0:

                # ----------------------------------------------------
                # Vertical crop
                # ----------------------------------------------------
                height = frame.shape[0]

                y1 = round(height * crop_top)
                y2 = round(height * (1.0 - crop_bottom))

                frame = frame[y1:y2, :]

                # ----------------------------------------------------
                # Resize after cropping
                # ----------------------------------------------------
                if target_width is not None and frame.shape[1] != target_width:
                    scale = target_width / frame.shape[1]
                    target_height = round(frame.shape[0] * scale)

                    frame = cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )

                final_height, final_width = frame.shape[:2]

                # OpenCV BGR -> RGB
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frames.append(frame_rgb)

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames extracted from video: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    imageio.mimsave(
        output_path,
        frames,
        duration=1.0 / playback_fps,
        loop=0,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print()
    print(f"Created: {output_path}")
    print(f"Frames: {len(frames)}")
    print(f"Sampled FPS: {actual_fps:.2f}")
    print(f"Playback speed: {playback_speed:.2f}x")
    print(f"Effective GIF FPS: {playback_fps:.2f}")
    print(f"Start frame: {start_frame}")
    print(f"Crop top: {crop_top:.0%}")
    print(f"Crop bottom: {crop_bottom:.0%}")
    print(f"Resolution: {final_width}x{final_height}")
    print(f"File size: {size_mb:.2f} MB")

def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating main video pose GIF...")

    video_to_gif(
        input_path=MAIN_VIDEO,
        output_path=MAIN_GIF,
        target_fps=8,
        target_width=600,
        start_frame=12,
        playback_speed=0.8,
        crop_top=0.40,
        crop_bottom=0.30,
    )

    print()
    print("Creating zoom panel GIF...")

    video_to_gif(
        input_path=ZOOM_VIDEO,
        output_path=ZOOM_GIF,
        target_fps=10,
        target_width=1000,
        start_frame=0,
        playback_speed=1.0,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
