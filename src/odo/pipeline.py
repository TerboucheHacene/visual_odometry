from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from odo.features import FeatureDetectorAndDescriptor, FeatureMatcher
from odo.motion import estimate_stereo_motion
from odo.stereo_matching import StereoMatcher, disp_to_depth


@dataclass
class Frame:
    """A class to represent a frame in a visual odometry pipeline."""

    image_left: np.ndarray
    image_right: np.ndarray
    disp_map: np.ndarray = None
    depth_map: np.ndarray = None
    matches: List[cv2.DMatch] = None
    kp1: List[cv2.KeyPoint] = None
    dp1: np.ndarray = None
    rmat: np.ndarray = None
    tvec: np.ndarray = None
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        if self.image_left.shape != self.image_right.shape:
            raise ValueError("Images must have the same shape")

        self.height, self.width = self.image_left.shape[:2]

    def compute_disparity_map(self, stereo_matcher: StereoMatcher) -> None:
        if self.disp_map is not None:
            return
        self.disp_map = stereo_matcher(self.image_left, self.image_right)

    def compute_depth_map(self, bf: float):
        if self.disp_map is None:
            raise ValueError("Disparity map must be computed first")
        if self.depth_map is not None:
            return
        self.depth_map = disp_to_depth(self.disp_map, bf)

    def detect_and_compute_features(
        self, feature_detector: FeatureDetectorAndDescriptor
    ):
        if (self.kp1 is not None) and (self.dp1 is not None):
            return
        self.kp1, self.dp1 = feature_detector.detect_and_compute(self.image_left)


class VisualOdometryEstimator:
    """A class to estimate the camera pose using stereo visual odometry.

    The visual odometry pipeline consists of the following steps:
    1. Detect and compute features in the left image at time t.
    2. Compute the disparity map using stereo matching at time t.
    3. Compute the depth map using the disparity map and bf.
    4. Detect and compute features in the left image at time t+1.
    5. Match features between the left images at time t and t+1.
    6. Estimate the relative motion between the two consecutive frames.
    7. Update the camera pose using the estimated motion.

    Parameters
    ----------
    feature_detector : FeatureDetectorAndDescriptor
        A feature detector and descriptor object.
    feature_matcher : FeatureMatcher
        A feature matcher object.
    stereo_matcher : StereoMatcher
        A stereo matcher object.
    camera_matrix : np.ndarray
        3x3 intrinsic camera matrix.
    bf : float
        Baseline times the focal length.

    """

    def __init__(
        self,
        feature_detector: FeatureDetectorAndDescriptor,
        feature_matcher: FeatureMatcher,
        stereo_matcher: StereoMatcher,
        camera_matrix: np.ndarray,
        bf: float,
    ) -> None:
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.stereo_matcher = stereo_matcher
        self.bf = bf
        self.camera_matrix = camera_matrix
        self._is_initialized = False

    def init(self, init_frame: Frame) -> None:
        """Initialize the visual odometry pipeline.

        Parameters
        ----------
        init_frame : Frame
            The initial frame to initialize the pipeline.
        """
        init_frame.detect_and_compute_features(self.feature_detector)
        init_frame.compute_disparity_map(self.stereo_matcher)
        init_frame.compute_depth_map(self.bf)
        self._prev_frame: Frame = init_frame
        self._current_frame: Frame = init_frame
        rmat = init_frame.rmat if init_frame.rmat is not None else np.eye(3)
        tvec = init_frame.tvec if init_frame.tvec is not None else np.zeros(3)
        tmat = np.eye(4)
        tmat[:3, :3] = rmat
        tmat[:3, 3] = tvec.squeeze()
        self._previous_pose = tmat
        self._current_pose: np.ndarray = None
        self._transformation: np.ndarray = None
        self._is_initialized = True

    def set_current_frame(self, frame: Frame) -> None:
        """Set the current frame in the visual odometry pipeline."""
        if self._current_frame is not None:
            self._prev_frame = self._current_frame
        self._current_frame = frame

    def run(self) -> bool:
        """Run the visual odometry pipeline.

        Returns
        -------
        bool
            True if the pipeline ran successfully, False otherwise.
        """
        if not self._is_initialized:
            raise ValueError("Pipeline must be initialized first")

        self._current_frame.detect_and_compute_features(self.feature_detector)
        self._current_frame.compute_disparity_map(self.stereo_matcher)
        self._current_frame.compute_depth_map(self.bf)

        matches = self.feature_matcher.match_and_filter(
            self._prev_frame.dp1, self._current_frame.dp1
        )
        self._current_frame.matches = matches

        if len(matches) < 10:
            return False

        # Estimate motion
        rmat, tvec = estimate_stereo_motion(
            matches=matches,
            kp1=self._prev_frame.kp1,
            kp2=self._current_frame.kp1,
            k_left=self.camera_matrix,
            depth_map=self._prev_frame.depth_map,
        )

        self._current_frame.rmat = rmat
        self._current_frame.tvec = tvec
        return True

    def compute_transformation(self):
        """Compute the transformation matrix between the current and previous frame."""
        if self._current_frame.rmat is None or self._current_frame.tvec is None:
            return np.eye(4)

        tmat = np.eye(4)
        tmat[:3, :3] = self._current_frame.rmat
        tmat[:3, 3] = self._current_frame.tvec.T
        tmat = np.linalg.inv(tmat)
        self._transformation = tmat

    def update_pose(self):
        """Update the camera pose using the estimated motion."""
        self._current_pose = self._previous_pose @ self._transformation
        self._previous_pose = self._current_pose

    @property
    def current_pose(self):
        """Return the current camera pose."""
        return self._current_pose

    @property
    def transformation(self):
        """Return the transformation matrix between the current and previous frame."""
        return self._transformation

    @property
    def current_translation(self):
        """Return the current translation vector."""
        return self._current_pose[:3, 3]

    @property
    def current_rotation(self):
        """Return the current rotation matrix."""
        return self._current_pose[:3, :3]
