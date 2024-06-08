from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class KittiItem:
    """A KITTI dataset item."""

    sequence_name: str
    left_image: Path
    right_image: Path
    disp_image: Optional[Path] = None
    velodyne_points: Optional[Path] = None
    timestamp: Optional[float] = None
    gt_pose: Optional[np.ndarray] = None
    estimated_pose: Optional[np.ndarray] = None


@dataclass
class KittiSequence:
    """A KITTI dataset sequence."""

    sequence_path: Path
    sequence_name: str
    projections: Dict[str, np.ndarray]
    items: List[KittiItem]
    num_items: int = field(init=False)

    def __post_init__(self):
        self.num_items = len(self.items)


class KittiDataset:
    """A class to load the KITTI dataset for visual odometry.

    The dataset is assumed to be organized as follows:
    |-- db_root
    |   |-- sequences
    |   |   |-- sequence_00
    |   |   |   |-- image_0
    |   |   |   |   |-- 000000.png
    |   |   |   |   |-- ...
    |   |   |   |-- image_1
    |   |   |   |   |-- 000000.png
    |   |   |   |   |-- ...
    |   |   |   |-- calib.txt
    |   |   |   |-- times.txt
    |   |   |-- sequence_01
    |   |   |   |-- ...
    |   |-- poses
    |   |   |-- sequence_00.txt
    |   |   |-- ...
    |   |-- velodyne
    |   |   |-- sequence_00
    |   |   |   |-- 000000.bin
    |   |   |   |-- ...
    |   |   |-- sequence_01
    |   |   |   |-- ...

    The `velodyne` directory is optional and is only loaded if `load_velodyne` is set to
    True. The `sequence_names` parameter can be used to load only a subset of the
    sequences. The dataset only iterates over the items in the current sequence, and hence
    the `current_sequence_name` property must be set before iterating.


    Parameters
    ----------
    db_root : Path
        The root directory of the KITTI dataset.
    load_velodyne : bool, optional
        Whether to load the velodyne points, by default False.
    sequence_names : List[str], optional
        The names of the sequences to load, by default None. If None, all sequences are
        loaded.


    Examples
    --------
    Load the KITTI dataset and iterate over the items in the first sequence:

        ```python
        from pathlib import Path
        from odo.dataset import KittiDataset

        db_root = Path("path/to/kitti")
        dataset = KittiDataset(db_root)
        print(dataset.sequence_names)
        # Set the current sequence
        dataset.current_sequence_name = "sequence_00"
        # Iterate over the items in the sequence
        for item in dataset:
            print(item.left_image)
        ```
    """

    def __init__(
        self,
        db_root: Path,
        load_velodyne: bool = False,
        sequence_names: List[str] = None,
    ):
        self.db_root = db_root
        self.load_velodyne = load_velodyne
        self._sequence_names = sequence_names
        self.sequences: Dict[str, KittiSequence] = {}
        self._load_sequences()
        self._current_sequence: Optional[KittiSequence] = None
        self._current_sequence_name: Optional[str] = None
        self._total_num_items = sum(
            [sequence.num_items for sequence in self.sequences.values()]
        )

    @property
    def total_num_items(self) -> int:
        """The total number of items in the dataset (all sequences)."""
        return self._total_num_items

    @property
    def current_sequence_name(self) -> str:
        """The name of the current sequence."""
        return self._current_sequence_name

    @current_sequence_name.setter
    def current_sequence_name(self, sequence_name: str) -> None:
        """Set the current sequence by name."""
        self._current_sequence = self.sequences[sequence_name]
        self._current_sequence_name = sequence_name

    @property
    def current_sequence(self) -> KittiSequence:
        """The current sequence object."""
        if self._current_sequence_name is None:
            raise ValueError("No current sequence set")
        else:
            self._current_sequence = self.sequences[self._current_sequence_name]
        return self._current_sequence

    def __len__(self) -> int:
        """The number of items in the current sequence."""
        if self._current_sequence is None:
            raise ValueError("No current sequence set")
        return self._current_sequence.num_items

    def __getitem__(self, idx: int) -> KittiItem:
        """Get an item from the current sequence."""
        if self._current_sequence is None:
            raise ValueError("No current sequence set")
        return self._current_sequence.items[idx]

    def _load_sequences(self):
        """Load the sequences from the dataset."""
        if self._sequence_names is None:
            self._sequence_names = [
                item.name
                for item in (self.db_root / "sequences").iterdir()
                if item.is_dir()
            ]
            self._sequence_names.sort()

        for sequence_name in tqdm(self._sequence_names):
            self.sequences[sequence_name] = self._load_sequence(sequence_name)

    @property
    def sequence_names(self) -> List[str]:
        """The names of the sequences in the dataset."""
        return list(self.sequences.keys())

    def _load_sequence(self, sequence_name: str) -> KittiSequence:
        """Load a sequence from the dataset."""
        sequence_path = self.db_root / "sequences" / sequence_name
        poses_path = self.db_root / "poses" / f"{sequence_name}.txt"
        calibration_path = sequence_path / "calib.txt"
        lidar_path = self.db_root / "velodyne" / sequence_name
        timestamps_path = sequence_path / "times.txt"

        # Load the calibration file
        df = pd.read_csv(calibration_path, sep=" ", header=None, index_col=0)
        projection = {
            "P0": df.loc["P0:"].values.reshape(3, 4),
            "P1": df.loc["P1:"].values.reshape(3, 4),
            "P2": df.loc["P2:"].values.reshape(3, 4),
            "P3": df.loc["P3:"].values.reshape(3, 4),
            "Tr": df.loc["Tr:"].values.reshape(3, 4) if "Tr:" in df.index else None,
        }

        # load the timestamps
        timestamps = pd.read_csv(
            timestamps_path, header=None, delimiter=" "
        ).values.squeeze()

        # Load the poses
        if poses_path.exists():
            poses = pd.read_csv(poses_path, header=None, delimiter=" ").values
        else:
            poses = None

        # Load the images
        left_images = sorted((sequence_path / "image_0").glob("*.png"))
        right_images = sorted((sequence_path / "image_1").glob("*.png"))
        if len(left_images) != len(right_images):
            raise ValueError("Number of left and right images do not match")

        # Load the velodyne points
        velodyne_points = (
            sorted(lidar_path.glob("*.bin")) if self.load_velodyne else None
        )

        items = []
        for i, (left_image, right_image) in enumerate(zip(left_images, right_images)):
            item = KittiItem(
                sequence_name=sequence_name,
                left_image=left_image,
                right_image=right_image,
                velodyne_points=velodyne_points[i] if self.load_velodyne else None,
                timestamp=timestamps[i],
                gt_pose=poses[i].reshape(3, 4) if poses is not None else None,
            )
            items.append(item)

        return KittiSequence(
            sequence_path=sequence_path,
            sequence_name=sequence_name,
            projections=projection,
            items=items,
        )

    def __iter__(self):
        """Iterate over the items in the current sequence."""
        self._current_sequence = self.sequences[self._current_sequence_name]
        self._current_sequence_idx = 0
        return self

    def __next__(self):
        """Get the next item in the current sequence."""
        if self._current_sequence_idx < self._current_sequence.num_items:
            item = self._current_sequence.items[self._current_sequence_idx]
            self._current_sequence_idx += 1
            return item
        else:
            raise StopIteration

    def projections(self) -> Dict[str, np.ndarray]:
        """Get the projection matrices for the current sequence."""
        if self._current_sequence is None:
            raise ValueError("No current sequence set")
        return self.current_sequence.projections
