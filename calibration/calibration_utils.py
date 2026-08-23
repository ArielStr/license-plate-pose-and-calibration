import numpy as np


def load_calibration_results(input_path):
    data = np.load(input_path, allow_pickle=True)

    return {
        "K_cv": data["K_cv"],
        "dist_coeffs": data["dist_coeffs"],
        "image_size": tuple(data["image_size"])
    }