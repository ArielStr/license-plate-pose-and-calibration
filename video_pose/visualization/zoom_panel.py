from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from video_pose.models import PlateFrameResult


@dataclass(frozen=True)
class ZoomPanelConfig:
    """
    Fixed zoom-panel layout.

    The panel always contains the same number of equally sized cells.
    Track IDs keep the same slot for the rest of the video.
    Empty/not-yet-visible tracks leave a black cell.
    """

    max_tracks: int = 6
    columns: int = 3

    cell_width: int = 640
    cell_height: int = 360

    header_height: int = 70
    border_thickness: int = 2

    padding_x_ratio: float = 0.75
    padding_top_ratio: float = 2.20
    padding_bottom_ratio: float = 1.00

    min_padding_x_px: int = 90
    min_padding_top_px: int = 120
    min_padding_bottom_px: int = 70

    @property
    def rows(self) -> int:
        return int(ceil(self.max_tracks / self.columns))

    @property
    def panel_width(self) -> int:
        return self.columns * self.cell_width

    @property
    def grid_height(self) -> int:
        return self.rows * self.cell_height

    @property
    def panel_height(self) -> int:
        return self.header_height + self.grid_height


class ZoomPanelRenderer:
    """
    Builds a fixed 6-cell panel from crops of the already annotated MVP frame.
    No pose/refinement/tracking is recomputed here.
    """

    def __init__(self, config: ZoomPanelConfig | None = None) -> None:
        self.config = config or ZoomPanelConfig()
        self._slot_by_track_id: dict[int, int] = {}
        self._warned_track_ids: set[int] = set()

    @property
    def width(self) -> int:
        return self.config.panel_width

    @property
    def height(self) -> int:
        return self.config.panel_height

    def _slot_for_track(self, track_id: int | None) -> int | None:
        if track_id is None:
            return None

        if track_id in self._slot_by_track_id:
            return self._slot_by_track_id[track_id]

        used_slots = set(self._slot_by_track_id.values())

        for slot in range(self.config.max_tracks):
            if slot not in used_slots:
                self._slot_by_track_id[track_id] = slot
                return slot

        if track_id not in self._warned_track_ids:
            print(
                f"Zoom panel full: Track {track_id} has no free slot "
                f"(max_tracks={self.config.max_tracks})"
            )
            self._warned_track_ids.add(track_id)

        return None

    def _crop_bounds(
        self,
        frame: np.ndarray,
        result: PlateFrameResult,
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = map(float, result.detection.bbox)

        bbox_width = max(1.0, x_max - x_min)
        bbox_height = max(1.0, y_max - y_min)

        padding_x = max(
            self.config.min_padding_x_px,
            int(round(self.config.padding_x_ratio * bbox_width)),
        )
        padding_top = max(
            self.config.min_padding_top_px,
            int(round(self.config.padding_top_ratio * bbox_height)),
        )
        padding_bottom = max(
            self.config.min_padding_bottom_px,
            int(round(self.config.padding_bottom_ratio * bbox_height)),
        )

        frame_height, frame_width = frame.shape[:2]

        crop_x_min = max(0, int(np.floor(x_min)) - padding_x)
        crop_y_min = max(0, int(np.floor(y_min)) - padding_top)
        crop_x_max = min(frame_width, int(np.ceil(x_max)) + padding_x)
        crop_y_max = min(frame_height, int(np.ceil(y_max)) + padding_bottom)

        return crop_x_min, crop_y_min, crop_x_max, crop_y_max

    def _fit_crop_to_cell(self, crop: np.ndarray) -> np.ndarray:
        """
        Fill the whole cell with the crop.

        This uses "cover" scaling instead of letterboxing:
        the crop is enlarged until the cell is completely filled, then the
        overflow is trimmed. Active cells therefore have no black bands.
        """
        cell_height = self.config.cell_height
        cell_width = self.config.cell_width

        if crop is None or crop.size == 0:
            return np.zeros(
                (cell_height, cell_width, 3),
                dtype=np.uint8,
            )

        crop_height, crop_width = crop.shape[:2]

        # Fill the cell completely.
        scale = max(
            cell_width / crop_width,
            cell_height / crop_height,
        )

        resized_width = max(
            cell_width,
            int(round(crop_width * scale)),
        )
        resized_height = max(
            cell_height,
            int(round(crop_height * scale)),
        )

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=interpolation,
        )

        # Center horizontally.
        x_start = max(
            0,
            (resized_width - cell_width) // 2,
        )

        # Slight upward bias helps preserve the existing Track / pose text
        # that is drawn above the plate in annotated_frame.
        overflow_y = max(
            0,
            resized_height - cell_height,
        )
        y_start = int(round(overflow_y * 0.35))

        return resized[
            y_start:y_start + cell_height,
            x_start:x_start + cell_width,
        ].copy()

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60.0)
        remaining = seconds - 60.0 * minutes
        return f"{minutes:02d}:{remaining:05.2f}"

    def _draw_header(
        self,
        panel: np.ndarray,
        *,
        timestamp_sec: float,
        duration_sec: float | None,
    ) -> None:
        current = self._format_time(timestamp_sec)

        if duration_sec is None:
            text = current
        else:
            text = f"{current} / {self._format_time(duration_sec)}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.15
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        text_x = max(
            0,
            (self.width - text_width) // 2,
        )

        text_y = (
            self.config.header_height
            + text_height
            - baseline
        ) // 2

        cv2.putText(
            panel,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_grid(self, panel: np.ndarray) -> None:
        top = self.config.header_height

        for slot in range(self.config.max_tracks):
            row = slot // self.config.columns
            column = slot % self.config.columns

            x0 = column * self.config.cell_width
            y0 = top + row * self.config.cell_height
            x1 = x0 + self.config.cell_width
            y1 = y0 + self.config.cell_height

            cv2.rectangle(
                panel,
                (x0, y0),
                (x1 - 1, y1 - 1),
                (90, 90, 90),
                self.config.border_thickness,
            )

    def render(
        self,
        annotated_frame: np.ndarray,
        frame_results: list[PlateFrameResult],
        *,
        timestamp_sec: float,
        duration_sec: float | None = None,
    ) -> np.ndarray:
        panel = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )

        self._draw_header(
            panel,
            timestamp_sec=timestamp_sec,
            duration_sec=duration_sec,
        )

        grid_top = self.config.header_height

        for result in frame_results:
            slot = self._slot_for_track(result.detection.track_id)

            if slot is None:
                continue

            x_min, y_min, x_max, y_max = self._crop_bounds(
                annotated_frame,
                result,
            )

            crop = annotated_frame[
                y_min:y_max,
                x_min:x_max,
            ]

            cell = self._fit_crop_to_cell(crop)

            row = slot // self.config.columns
            column = slot % self.config.columns

            panel_y = grid_top + row * self.config.cell_height
            panel_x = column * self.config.cell_width

            panel[
                panel_y:panel_y + self.config.cell_height,
                panel_x:panel_x + self.config.cell_width,
            ] = cell

        self._draw_grid(panel)

        return panel
