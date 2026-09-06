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
):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {input_path}")

    if playback_speed <= 0:
        raise ValueError("playback_speed must be greater than 0")

    frame_step = max(1, round(source_fps / target_fps))
    actual_fps = source_fps / frame_step
    playback_fps = actual_fps * playback_speed

    if target_width is not None and target_width < source_width:
        scale = target_width / source_width
        target_height = round(source_height * scale)
    else:
        target_width = source_width
        target_height = source_height

    frames = []
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index >= start_frame:
            relative_index = frame_index - start_frame

            if relative_index % frame_step == 0:
                if frame.shape[1] != target_width:
                    frame = cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from video: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"Resolution: {target_width}x{target_height}")
    print(f"File size: {size_mb:.2f} MB")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating main video pose GIF...")

    video_to_gif(
        input_path=MAIN_VIDEO,
        output_path=MAIN_GIF,
        target_fps=8,
        target_width=480,
        start_frame=12,
        playback_speed=0.8,
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
