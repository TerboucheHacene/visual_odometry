import math
from typing import List, Tuple

import cv2
import numpy as np


def estimate_stereo_motion(
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    k_left: np.ndarray,
    depth_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the relative motion between two consecutive stereo frames.

    The relative motion is estimated using the 3D points at time t and the 2D
    correspondences at time t+1 using the following equation:
    ```
        points_2D = K * [R | t] * points_3D
    ```
    where `points_2D` are the 2D points at time t+1, `K` is the intrinsic matrix
    of the left camera, `R` is the rotation matrix, `t` is the translation vector,
    and `points_3D` are the 3D points at time t. The problem is solved using the
    PnP RANSAC algorithm from OpenCV to estimate the rotation and translation.


    Parameters
    ----------
    matches : List[cv2.DMatch]
        List of matches between the two consecutive stereo frames.
    kp1 : List[cv2.KeyPoint]
        List of keypoints in the first stereo frame at time t.
    kp2 : List[cv2.KeyPoint]
        List of keypoints in the second stereo frame at time t+1.
    k_left : np.ndarray
        3x3 intrinsic matrix of the left camera.
    depth_map : np.ndarray
        Depth map of the left image at time t.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotation matrix and the translation vector.
    """

    # Extract 3D points
    points_3d = []
    points_2d = []
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        p1 = kp1[idx1].pt
        p2 = kp2[idx2].pt
        depth = depth_map[int(p1[1]), int(p1[0])]
        if math.isnan(depth) or math.isinf(depth):
            continue
        points_3d.append(
            [
                (p1[0] - k_left[0, 2]) * depth / k_left[0, 0],
                (p1[1] - k_left[1, 2]) * depth / k_left[1, 1],
                depth,
            ]
        )
        points_2d.append(p2)

    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # Estimate motion
    _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, points_2d, k_left, None)
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat, tvec


def estimate_mono_motion(
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    k_left: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the relative motion between two consecutive monocular frames.

    The relative motion is estimated using the essential matrix decomposition
    algorithm from OpenCV. The essential matrix is computed using the 2D
    correspondences between the two consecutive frames and the intrinsic matrix
    of the left camera. The essential matrix is then decomposed into the rotation
    and translation matrices.

    Parameters
    ----------
    matches : List[cv2.DMatch]
        List of matches between the two consecutive monocular frames.
    kp1 : List[cv2.KeyPoint]
        List of keypoints in the first monocular frame at time t.
    kp2 : List[cv2.KeyPoint]
        List of keypoints in the second monocular frame at time t+1.
    k_left : np.ndarray
        3x3 intrinsic matrix of the left camera.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotation matrix and the translation vector.
    """

    points1 = []
    points2 = []
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        p1 = kp1[idx1].pt
        p2 = kp2[idx2].pt
        points1.append(p1)
        points2.append(p2)

    points1 = np.array(points1)
    points2 = np.array(points2)

    E, _ = cv2.findEssentialMat(
        points1, points2, k_left, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    _, R, t, _ = cv2.recoverPose(E, points1, points2, k_left)
    return R, t
