import numpy as np


def _v_ij(H, i, j):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ], dtype=np.float64)


def estimate_B_from_homographies(homographies):
    if len(homographies) < 3:
        raise ValueError("At least three homographies are required.")

    rows = []

    for item in homographies:
        H = np.asarray(item["H"], dtype=np.float64)

        if abs(H[2, 2]) < 1e-12:
            raise ValueError(
                "Cannot normalize homography by H[2,2]."
            )

        H = H / H[2, 2]

        v12 = _v_ij(H, 0, 1)
        v11 = _v_ij(H, 0, 0)
        v22 = _v_ij(H, 1, 1)

        rows.append(v12)
        rows.append(v11 - v22)

    V = np.vstack(rows)

    _, singular_values, vt = np.linalg.svd(V)
    b = vt[-1]

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]],
    ], dtype=np.float64)

    return B, V, singular_values


def compute_K_from_B(B):
    B = np.asarray(B, dtype=np.float64)
    B = 0.5 * (B + B.T)

    for sign in (1.0, -1.0):
        B_try = sign * B

        if abs(B_try[2, 2]) > 1e-12:
            B_try = B_try / B_try[2, 2]

        try:
            L = np.linalg.cholesky(B_try)
            K = np.linalg.inv(L.T)
            K = K / K[2, 2]
            return K
        except np.linalg.LinAlgError:
            continue

    return None


def estimate_intrinsics_from_homographies(homographies):
    """
    Core Zhang calibration result plus diagnostics useful for experiments.
    """
    B, V, singular_values = estimate_B_from_homographies(
        homographies
    )

    K_est = compute_K_from_B(B)
    print("V singular values:")
    print(singular_values)

    print("\nEstimated B:")
    print(B)

    print("\nB eigenvalues:")
    print(np.linalg.eigvalsh(0.5 * (B + B.T)))
    if K_est is None:
        raise RuntimeError(
            "Could not recover K: estimated B is not positive definite."
        )

    if singular_values[-1] > 0:
        condition_number = float(
            singular_values[0] / singular_values[-1]
        )
    else:
        condition_number = float("inf")

    return {
        "K": K_est,
        "B": B,
        "V": V,
        "singular_values": singular_values,
        "condition_number": condition_number,
        "num_homographies": len(homographies),
    }


def compare_K(K_est, K_ref):
    """Evaluation helper only; not used by the core pipeline."""
    print("\n========== K COMPARISON ==========\n")
    print("Estimated K:")
    print(K_est)

    print("\nReference K:")
    print(K_ref)

    print("\nDifference K_est - K_ref:")
    print(K_est - K_ref)

    print("\nRelative focal error:")
    print(
        f"fx error: "
        f"{(K_est[0, 0] - K_ref[0, 0]) / K_ref[0, 0] * 100:.2f}%"
    )
    print(
        f"fy error: "
        f"{(K_est[1, 1] - K_ref[1, 1]) / K_ref[1, 1] * 100:.2f}%"
    )

    print("\nPrincipal point error:")
    print(f"cx error: {K_est[0, 2] - K_ref[0, 2]:.2f} px")
    print(f"cy error: {K_est[1, 2] - K_ref[1, 2]:.2f} px")


def estimate_K_from_homographies(homographies, K_ref=None):
    """Backwards-compatible wrapper for older experiment scripts."""
    result = estimate_intrinsics_from_homographies(
        homographies
    )

    K_est = result["K"]

    print("\n========== ZHANG INTRINSICS ==========\n")
    print("Number of homographies:", result["num_homographies"])
    print("Constraint matrix shape:", result["V"].shape)

    print("Constraint singular values:")
    print(result["singular_values"])

    print(
        "Constraint condition number:",
        f"{result['condition_number']:.6e}",
    )

    print("\nEstimated B:")
    print(result["B"])

    print("\nEstimated K:")
    print(K_est)

    if K_ref is not None:
        compare_K(K_est, K_ref)

    return K_est
