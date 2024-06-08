from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from odo.metrics import EvaluationMetrics


class Plotter:
    """A class to visualize the stereo visual odometry pipeline using matplotlib.

    The plotter creates a 3x2 grid of subplots to visualize the following:
    | Left Image | Right Image |
    | Disparity  | 3D Trajectory spanning two rows |
    | Error Metrics | 3D Trajectory |

    Parameters
    ----------
    min_pose : np.ndarray
        A 3D array (x, y, z) representing the minimum pose values.
    max_pose : np.ndarray
        A 3D array (x, y, z) representing the maximum pose values.
    """

    def __init__(self, min_pose: np.ndarray, max_pose: np.ndarray) -> None:
        self.fig = plt.figure()
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1:, 1], projection="3d")
        self.ax5 = self.fig.add_subplot(gs[2, 0])

        self.ax5.set_xlabel("Frame")
        self.ax5.set_ylabel("Error")
        self.ax5.set_title("Visual Odometry Metrics")
        self.ax5.grid(True)

        self.ax4.set_xlabel("X")
        self.ax4.set_ylabel("Y")
        self.ax4.set_zlabel("Z")
        self.ax4.view_init(-20, 270)

        min_x, max_x = min_pose[0], max_pose[0]
        min_y, max_y = min_pose[1], max_pose[1]
        min_z, max_z = min_pose[2], max_pose[2]

        self.ax4.set_xlim(min_x - 10, max_x + 10)
        self.ax4.set_ylim(min_y - 10, max_y + 10)
        self.ax4.set_zlim(min_z - 10, max_z + 10)

        # Initialize the plots with empty data.
        (self.gt_plot,) = self.ax4.plot(
            [], [], [], color="blue", markersize=1, label="Ground Truth"
        )
        (self.pred_plot,) = self.ax4.plot(
            [], [], [], color="red", markersize=1, label="Prediction"
        )
        self.idx = 0

    def update(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        disp_map: np.ndarray,
        gt_pose: np.ndarray,
        pred_pose: np.ndarray,
        metrics: EvaluationMetrics,
    ) -> None:
        """Update the subplots with the latest images, disparity map, and poses.

        Parameters
        ----------
        left_image : np.ndarray
            A HxW grayscale image from the left camera.
        right_image : np.ndarray
            A HxW grayscale image from the right camera.
        disp_map : np.ndarray
            A HxW disparity map.
        gt_pose : np.ndarray
            A Nx3 array representing the ground truth poses.
        pred_pose : np.ndarray
            A Nx3 array representing the predicted poses.
        metrics : EvaluationMetrics
            A class containing error metrics (mse, rmse, mae).
        """
        self.idx += 1
        self.ax1.imshow(left_image, cmap="gray")
        self.ax2.imshow(right_image, cmap="gray")
        self.ax3.imshow(disp_map, cmap="jet")

        self.gt_plot.set_data(gt_pose[:, 0], gt_pose[:, 1])
        self.gt_plot.set_3d_properties(gt_pose[:, 2])
        self.pred_plot.set_data(pred_pose[:, 0], pred_pose[:, 1])
        self.pred_plot.set_3d_properties(pred_pose[:, 2])

        self.ax5.scatter(self.idx, metrics.mse, color="red", label="MSE")
        self.ax5.scatter(self.idx, metrics.rmse, color="blue", label="RMSE")
        self.ax5.scatter(self.idx, metrics.mae, color="green", label="MAE")
        if self.idx == 1:
            self.ax5.legend()

        self.ax1.axis("off")
        self.ax2.axis("off")
        self.ax3.axis("off")
        self.ax1.set_title("Left Image")
        self.ax2.set_title("Right Image")
        self.ax3.set_title("Disparity")
        plt.pause(0.000000000000000001)


class VideoSaver:
    """A class to save matplotlib figures as a video.

    Parameters
    ----------
    filename : str
        The name of the output video file.
    fps : int
        Frames per second of the output video.
    frame_size : Tuple[int, int]
        A tuple containing the width and height of the video frame.

    """

    def __init__(self, filename: str, fps: int, frame_size: Tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self._current_frame: np.ndarray = None

    @property
    def current_frame(self) -> np.ndarray:
        """Return the current frame."""
        return self._current_frame

    def write_frame(self, fig: plt.Figure) -> None:
        """Write the current matplotlib figure to the video.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure to write to the video.
        """
        # Draw the canvas, then convert it to an image.
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._current_frame = img
        self.video_writer.write(img)

    def release(self) -> None:
        """Release the video writer."""
        self.video_writer.release()
