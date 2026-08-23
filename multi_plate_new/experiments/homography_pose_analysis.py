import numpy as np


def extract_pose_from_homography(H, K):
    K_inv = np.linalg.inv(K)

    B = K_inv @ H

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    scale = 1.0 / np.linalg.norm(b1)

    r1 = scale * b1
    r2 = scale * b2
    t = scale * b3

    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)

    normal = np.cross(r1, r2)
    normal = normal / np.linalg.norm(normal)

    return {
        "r1": r1,
        "r2": r2,
        "normal": normal,
        "t": t,
        "scale": scale
    }


def angle_between_vectors_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def analyze_plate_normals(homographies, K):
    results = []

    print("\n========== PLATE NORMALS ==========\n")

    for h in homographies:
        pose = extract_pose_from_homography(h["H"], K)

        normal = pose["normal"]
        t = pose["t"]

        results.append({
            "index": h["index"],
            "normal": normal,
            "t": t,
            "pose": pose
        })

        print(f"Plate {h['index']}")
        print(f"normal: {normal}")
        print(f"t: {t}")
        print(f"distance approx: {np.linalg.norm(t):.3f}")
        print()

    print("\n========== NORMAL ANGLES ==========\n")

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            angle = angle_between_vectors_deg(
                results[i]["normal"],
                results[j]["normal"]
            )

            print(
                f"Plate {results[i]['index']} <-> "
                f"Plate {results[j]['index']}: "
                f"{angle:.2f} deg"
            )

    return results