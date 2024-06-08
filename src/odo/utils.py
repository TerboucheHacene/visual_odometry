from typing import Tuple

import cv2
import numpy as np


def decompose_projection_matrix(
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a projection matrix into its intrinsic and extrinsic parameters.

    Parameters
    ----------
    P : np.ndarray
        The projection matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The intrinsic matrix, the rotation matrix, and the translation vector.
    """
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    t = t / t[3]
    t = t[:3]
    return k, r, t


def compute_bf(t_left: np.ndarray, t_right: np.ndarray, k_left: np.ndarray) -> float:
    """Compute the baseline times the focal length.

    Parameters
    ----------
    t_left : np.ndarray
        3x1 translation vector of the left camera.
    t_right : np.ndarray
        3x1 translation vector of the right camera.
    k_left : np.ndarray
        3x3 intrinsic matrix of the left camera.

    Returns
    -------
    float
        The baseline times the focal length.
    """
    b = np.abs(t_left[0] - t_right[0]).item()
    f = k_left[0, 0]
    return b * f


def disp_to_depth(disp: np.ndarray, bf: float) -> np.ndarray:
    mask_nan = np.isnan(disp)
    mask_0 = disp == 0

    depth = np.zeros_like(disp)
    depth[~mask_nan & ~mask_0] = bf / disp[~mask_nan & ~mask_0]
    depth[mask_nan] = np.nan
    depth[mask_0] = np.inf

    return depth
