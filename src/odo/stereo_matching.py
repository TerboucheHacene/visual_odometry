from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from odo.utils import disp_to_depth


@dataclass
class StereoMatchingConfig:
    """Configuration for stereo matching."""

    method: Literal["bm", "sgbm"]
    num_disparities: int
    block_size: int
    min_disparity: int = 0


class StereoMatcher:
    """A stereo matcher that computes disparity maps and depth maps.

    It uses either the block matching method (BM) or the semi-global block matching
    method (SGBM) from OpenCV. The configuration of the stereo matcher is defined by
    a StereoMatchingConfig object.

    Parameters
    ----------
    config : StereoMatchingConfig
        Configuration for stereo matching.

    """

    def __init__(self, config: StereoMatchingConfig) -> None:
        self.config = config
        if config.method == "bm":
            self.matcher = cv2.StereoBM_create(
                numDisparities=config.num_disparities,
                blockSize=config.block_size,
            )
        elif config.method == "sgbm":
            self.matcher = cv2.StereoSGBM_create(
                numDisparities=config.num_disparities,
                blockSize=config.block_size,
                minDisparity=config.min_disparity,
                P1=8 * 3 * config.block_size**2,
                P2=32 * 3 * config.block_size**2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        else:
            raise ValueError("The method must be 'bm' or 'sgbm'")

    def _check_input(self, left_image: np.ndarray, right_image: np.ndarray) -> None:
        """Check that the input images are valid."""

        if left_image.shape != right_image.shape:
            raise ValueError("Left and right images must have the same shape")
        if left_image.ndim != 2 or right_image.ndim != 2:
            raise ValueError("Left and right images must be grayscale")

    def __call__(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """Compute the disparity map from a pair of stereo images.

        Parameters
        ----------
        left_image : np.ndarray
            HxW grayscale image from the left camera.
        right_image : np.ndarray
            HxW grayscale image from the right camera.

        Returns
        -------
        np.ndarray
            HxW disparity map.
        """
        self._check_input(left_image, right_image)
        disp_left = self.matcher.compute(left_image, right_image)
        disp_left = disp_left.astype(np.float32) / 16.0
        disp_left[disp_left == -1] = np.nan
        return disp_left

    def compute_depth(
        self, left_image: np.ndarray, right_image: np.ndarray, bf: float
    ) -> np.ndarray:
        """Compute the depth map from a pair of stereo images.

        Parameters
        ----------
        left_image : np.ndarray
            HxW grayscale image from the left camera.
        right_image : np.ndarray
            HxW grayscale image from the right camera.
        bf : float
            Baseline times the focal length.

        Returns
        -------
        np.ndarray
            HxW depth map.
        """

        disp = self(left_image, right_image)
        return disp_to_depth(disp, bf)
