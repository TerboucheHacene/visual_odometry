from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from odo.dataset import KittiDataset
from odo.features import (
    FeatureDetectorAndDescriptor,
    FeatureDetectorAndDescriptorConfig,
    FeatureMatcher,
    FeatureMatcherConfig,
)
from odo.metrics import VisualOdometryEvaluator
from odo.pipeline import Frame, VisualOdometryEstimator
from odo.stereo_matching import StereoMatcher, StereoMatchingConfig
from odo.utils import compute_bf, decompose_projection_matrix
from odo.viz import Plotter, VideoSaver


def main():

    # Load the KITTI dataset
    db_path = Path("/home/hacene/Documents/workspace/visual_odometry/KITTI/dataset")
    dataset = KittiDataset(db_path)
    dataset.current_sequence_name = "00"

    # Initialize the StereoMatcher
    stereo_matching_config = StereoMatchingConfig(
        method="sgbm", num_disparities=96, block_size=11
    )
    stereo_matcher = StereoMatcher(stereo_matching_config)

    # Initialize the FeatureDetectorAndDescriptor
    feature_detector_config = FeatureDetectorAndDescriptorConfig(
        detector="sift", descriptor="sift"
    )
    feature_detector = FeatureDetectorAndDescriptor(feature_detector_config)

    # Initialize the FeatureMatcher
    feature_matching_config = FeatureMatcherConfig(
        method="bf",
        config={"normType": cv2.NORM_L2, "crossCheck": False},
        lowe_ratio=0.5,
    )
    feature_matcher = FeatureMatcher(feature_matching_config)

    # Decompose the projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(dataset.projections()["P0"])
    k_right, r_right, t_right = decompose_projection_matrix(dataset.projections()["P1"])

    # Compute the baseline * focal length
    bf = compute_bf(t_left, t_right, k_left)

    # Initialize the EvaluationMetrics
    evaluator = VisualOdometryEvaluator()

    # Initialize the VisualOdometryEstimator
    pipeline = VisualOdometryEstimator(
        stereo_matcher=stereo_matcher,
        feature_detector=feature_detector,
        feature_matcher=feature_matcher,
        bf=bf,
        camera_matrix=k_left,
    )

    # Initialize the pipeline
    init_frame = Frame(
        image_left=cv2.imread(dataset[0].left_image.__str__(), cv2.IMREAD_GRAYSCALE),
        image_right=cv2.imread(dataset[0].right_image.__str__(), cv2.IMREAD_GRAYSCALE),
    )
    pipeline.init(init_frame)

    # Initialize the visualization
    plot = True
    save_video = True
    if plot:
        plt.ion()
        gt_pose = np.array(
            [
                [item.gt_pose[0, 3], item.gt_pose[1, 3], item.gt_pose[2, 3]]
                for item in dataset
            ]
        )
        min_pose = gt_pose.min(axis=0)
        max_pose = gt_pose.max(axis=0)
        plotter = Plotter(min_pose, max_pose)

        if save_video:
            frame_size = (
                int(plotter.fig.get_size_inches()[0] * plotter.fig.dpi),
                int(plotter.fig.get_size_inches()[1] * plotter.fig.dpi),
            )
            video_saver = VideoSaver(
                "visual_odometry.avi", fps=10, frame_size=frame_size
            )

    # Initialize the pose lists
    gt_pose_list = [np.array([0, 0, 0])]
    pred_pose_list = [np.array([0, 0, 0])]

    # Run the pipeline
    for i in tqdm(range(1, len(dataset)), desc="Running the pipeline"):
        item = dataset[i]
        left_image = cv2.imread(str(item.left_image), cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(str(item.right_image), cv2.IMREAD_GRAYSCALE)
        frame = Frame(
            image_left=left_image,
            image_right=right_image,
        )
        pipeline.set_current_frame(frame)
        pipeline.run()
        pipeline.compute_transformation()
        pipeline.update_pose()

        evaluator.update(
            item.gt_pose[:3, 3],
            pipeline.current_translation,
        )
        metrics = evaluator.compute()

        gt_pose_list.append(item.gt_pose[:3, 3])
        pred_pose_list.append(pipeline.current_pose[:3, 3])

        if plot:
            plotter.update(
                left_image,
                right_image,
                frame.disp_map,
                np.array(gt_pose_list),
                np.array(pred_pose_list),
                metrics,
            )
            if save_video:
                video_saver.write_frame(plotter.fig)

    print(f"Mean Squared Error: {metrics.mse}")
    print(f"Root Mean Squared Error: {metrics.rmse}")
    print(f"Mean Absolute Error: {metrics.mae}")

    if save_video:
        video_saver.release()
        cv2.imwrite("pipeline_output.png", video_saver.current_frame)

    plt.close()


if __name__ == "__main__":
    main()
