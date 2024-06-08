from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FeatureDetectorAndDescriptorConfig:
    """Configuration for feature detection and description."""

    detector: Literal["fast", "harris", "shi-tomasi", "sift", "surf", "orb"]
    descriptor: Literal["sift", "surf", "orb"]
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.config = self.config or {}


class FeatureDetectorAndDescriptor:
    """A class to detect and describe features in images.

    It uses a feature detector and a feature descriptor from OpenCV. The configuration
    of the feature detector and descriptor is defined by a
    FeatureDetectorAndDescriptorConfig object.

    Parameters
    ----------
    config : FeatureDetectorAndDescriptorConfig
        Configuration for feature detection and description.

    """

    def __init__(self, config: FeatureDetectorAndDescriptorConfig) -> None:
        self.config = config
        self.detector = self._create_detector()
        self.descriptor = self._create_descriptor()

    def _create_detector(self) -> Optional[cv2.Feature2D]:
        """Create a feature detector.

        Returns
        -------
        Optional[cv2.Feature2D]
            The feature detector, or None if the descriptor does not require one.
        """

        if self.config.detector in ["sift", "surf", "orb"]:
            return None
        if self.config.detector == "fast":
            return cv2.FastFeatureDetector_create(**self.config.config)
        elif self.config.detector == "harris":
            return cv2.FeatureDetector_create("HARRIS", **self.config.config)

        elif self.config.detector == "shi-tomasi":
            return cv2.GFTTDetector_create(**self.config.config)
        else:
            raise ValueError(f"Invalid detector: {self.config.detector}")

    def _create_descriptor(self) -> cv2.Feature2D:
        """Create a feature descriptor.

        Returns
        -------
        cv2.Feature2D
            The feature descriptor.
        """
        if self.config.descriptor == "sift":
            return cv2.SIFT_create(**self.config.config)

        elif self.config.descriptor == "surf":
            return cv2.SURF_create(**self.config.config)

        elif self.config.descriptor == "orb":
            return cv2.ORB_create(**self.config.config)

        else:
            raise ValueError(f"Invalid descriptor: {self.config.descriptor}")

    def detect_and_compute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect and compute features in an image.

        Parameters
        ----------
        image : np.ndarray
            HxW grayscale image in uint8 format.
        mask : Optional[np.ndarray], optional
            HxW binary mask, by default None.

        Returns
        -------
        List[cv2.KeyPoint]
            List of keypoints objects detected in the image.
        np.ndarray
            NxM array of feature descriptors, where N is the number of keypoints and M is
            the descriptor dimensionality.
        """
        if self.detector is None:  # Descriptor does not require a detector
            kps, descs = self.descriptor.detectAndCompute(image, mask)

        else:  # Descriptor requires a detector
            kps = self.detector.detect(image, mask)
            kps, descs = self.descriptor.compute(image, kps)

        return kps, descs

    def detect(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> List[cv2.KeyPoint]:
        """Detect features in an image.

        Parameters
        ----------
        image : np.ndarray
            HxW grayscale image in uint8 format.
        mask : Optional[np.ndarray], optional
            HxW binary mask, by default None.

        Returns
        -------
        List[cv2.KeyPoint]
            List of keypoints objects detected in the image.
        """
        if self.detector is None:
            kps = self.descriptor.detect(image, mask)
        else:
            kps = self.detector.detect(image, mask)
        return kps

    def compute(self, image: np.ndarray, kps: List[cv2.KeyPoint]) -> np.ndarray:
        """Compute feature descriptors for a set of keypoints.

        Parameters
        ----------
        image : np.ndarray
            HxW grayscale image in uint8 format.

        kps : List[cv2.KeyPoint]
            List of keypoints objects detected in the image.

        Returns
        -------
        np.ndarray
            NxM array of feature descriptors, where N is the number of keypoints and M is
            the descriptor dimensionality.
        """
        return self.descriptor.compute(image, kps)


@dataclass
class FeatureMatcherConfig:
    """Configuration for feature matching."""

    method: Literal["bf", "flann"]
    lowe_ratio: float = 0.5
    sorted_matches: bool = True
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.config = self.config or {}


class FeatureMatcher:
    """A class to match features between two images.

    It uses a feature matcher from OpenCV. The configuration of the feature matcher is
    defined by a FeatureMatcherConfig object.

    Parameters
    ----------
    config : FeatureMatcherConfig
        Configuration for feature matching.
    """

    def __init__(self, config: FeatureMatcherConfig) -> None:
        self.config = config
        self.matcher = self.create_matcher()

    def create_matcher(self) -> cv2.DescriptorMatcher:
        if self.config.method == "bf":
            return cv2.BFMatcher_create(**self.config.config)

        elif self.config.method == "flann":
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)

        else:
            raise ValueError(f"Invalid matcher: {self.config.method}")

    def match(self, descs1: np.ndarray, descs2: np.ndarray) -> List[List[cv2.DMatch]]:
        """Match features between two sets of descriptors.

        Parameters
        ----------
        descs1 : np.ndarray
            NxM array of feature descriptors, where N is the number of keypoints and M is
            the descriptor dimensionality.
        descs2 : np.ndarray
            NxM array of feature descriptors, where N is the number of keypoints and M is
            the descriptor dimensionality.

        Returns
        -------
        List[List[cv2.DMatch]]
            List of matches between the two sets of descriptors.
        """
        matches = self.matcher.knnMatch(descs1, descs2, k=2)
        if self.config.sorted_matches:
            matches = sorted(matches, key=lambda x: x[0].distance)
        return matches

    def filter_matches(self, matches: List[cv2.DMatch]) -> List[cv2.DMatch]:
        """Filter matches using the Lowe's ratio test.

        Parameters
        ----------
        matches : List[cv2.DMatch]
            List of matches between two sets of descriptors.

        Returns
        -------
        List[cv2.DMatch]
            List of filtered matches.
        """
        good_matches = []
        for m, n in matches:
            if m.distance < self.config.lowe_ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def match_and_filter(
        self, descs1: np.ndarray, descs2: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match and filter features between two sets of descriptors.

        Parameters
        ----------
        descs1 : np.ndarray
            NxM array of feature descriptors, where N is the number of keypoints and M is
            the descriptor dimensionality.
        descs2 : np.ndarray
            NxM array of feature descriptors, where N is the number of keypoints and M is
            the descriptor dimensionality.

        Returns
        -------
        List[cv2.DMatch]
            List of filtered matches.
        """

        matches = self.match(descs1, descs2)
        good_matches = self.filter_matches(matches)
        return good_matches
