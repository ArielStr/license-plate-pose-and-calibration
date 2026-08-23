import cv2
import numpy as np


def show_resized(
    window_name,
    img,
    max_width=1200,
    max_height=800,
):
    h, w = img.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    resized = cv2.resize(
        img,
        (
            int(w * scale),
            int(h * scale),
        ),
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.imshow(
        window_name,
        resized,
    )

    cv2.waitKey(0)

    cv2.destroyWindow(
        window_name
    )


def rectify_plate(
    img,
    image_points,
    plate_w_cm=45.0,
    plate_h_cm=10.0,
    debug=True,
):
    scale = 10

    rect_w = int(
        plate_w_cm
        * scale
    )

    rect_h = int(
        plate_h_cm
        * scale
    )

    dst_rect = np.array(
        [
            [0, 0],
            [rect_w - 1, 0],
            [
                rect_w - 1,
                rect_h - 1,
            ],
            [0, rect_h - 1],
        ],
        dtype=np.float32,
    )

    H_rect = cv2.getPerspectiveTransform(
        image_points.astype(
            np.float32
        ),
        dst_rect,
    )

    rectified = cv2.warpPerspective(
        img,
        H_rect,
        (
            rect_w,
            rect_h,
        ),
    )

    if debug:
        cv2.namedWindow(
            "Rectified plate",
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            "Rectified plate",
            1000,
            300,
        )

        cv2.imshow(
            "Rectified plate",
            rectified,
        )

        cv2.waitKey(0)

        cv2.destroyWindow(
            "Rectified plate"
        )

    return (
        rectified,
        H_rect,
    )


def draw_pose_on_image(
    img,
    image_points,
    pose,
    plate_index=1,
):
    output = img.copy()

    colors = [
        (0, 255, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (0, 255, 255),
    ]

    color = colors[
        (plate_index - 1)
        % len(colors)
    ]

    pts = (
        image_points
        .astype(int)
        .reshape((-1, 1, 2))
    )

    cv2.polylines(
        output,
        [pts],
        isClosed=True,
        color=color,
        thickness=4,
    )

    distance_m = (
        pose["distance_to_center_cm"]
        / 100.0
    )

    yaw_deg = pose[
        "yaw_deg"
    ]

    text = (
        f"Car {plate_index}: "
        f"{distance_m:.2f} m, "
        f"yaw {yaw_deg:.1f} deg"
    )

    x_min = int(
        np.min(
            image_points[:, 0]
        )
    )

    y_min = int(
        np.min(
            image_points[:, 1]
        )
    )

    text_x = x_min

    text_y = max(
        y_min - 20,
        40,
    )

    cv2.putText(
        output,
        text,
        (
            text_x,
            text_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return output
