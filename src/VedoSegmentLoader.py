import re
import nrrd
import numpy as np
from vedo import Volume, Mesh, colors
from pathlib import Path


class SegmentationNameNotFoundError(Exception):
    """Raised when a segmentation name is not found in SegmentationWrapper."""

    pass


class SegmentationLoaderManager:
    """
    Class to manage the loading and processing of segmentation data.
    - Load to seg.nrrd files exported from 3D Slicer into Vedo Volumes.
    - Manages multi-layer nrrd files (segmentations where segments overlap).

    """

    def __init__(self, segmentation_path: Path):
        self.segmentation_path = segmentation_path
        if not segmentation_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")

        self.data, self.header = nrrd.read(str(segmentation_path))

        self.spacing, self.origin = self.parse_header()
        self.dimension = self.header["dimension"]

        # Cache for loaded volumes - (LabelValue, Volume)
        self.cache_volume_dict: dict[str, tuple[int, Volume]] = {}

    def parse_header(self):
        # voxel spacing (x,y,z) in mm
        spacing = np.array(
            [
                float(self.header["space directions"][1][0]),  # X
                float(self.header["space directions"][2][1]),  # Y
                float(self.header["space directions"][3][2]),  # Z
            ]
        )

        origin = self.header["space origin"]

        return spacing, origin

    def get_volume_from_segment_name(self, label_name: str) -> Volume:
        if label_name not in self.cache_volume_dict:
            label_value = self.get_segment_label_value_from_name(label_name)
            raw_data = self.get_data_from_segment_name(label_name)
            v = Volume(raw_data, spacing=self.spacing, origin=self.origin)
            self.cache_volume_dict[label_name] = (label_value, v)

        return self.cache_volume_dict[label_name][1]

    def load_volumes_to_cache(self, segment_names: list[str]) -> None:
        for name in segment_names:
            self.get_volume_from_segment_name(name)

    def get_cache_volume_dict(self) -> dict[str, tuple[int, Volume]]:
        return self.cache_volume_dict

    def get_data_from_segment_name(self, label_name: str):
        """
        In case of overlapping segments, data might be split in multiple volumes.
        You need to use header information to figure out the right data volume.
        """
        if len(self.data.shape) > 3:
            layer = self.get_segment_layer(label_name)
            return self.data[layer]
        elif len(self.data.shape) == 3:
            return self.data
        else:
            raise ValueError("Unsupported data shape")

    def get_segment_layer(self, label_name: str) -> int:
        """For multi-layer nrrd files"""
        segment_id = self.find_segment_index(label_name)
        segment_layer = self.header[f"Segment{segment_id}_Layer"]
        return int(segment_layer)

    def get_segment_label_value_from_name(self, label_name: str) -> int:
        segment_index = self.find_segment_index(label_name)
        return self.get_segment_label_value(segment_index)

    def get_segment_label_value(self, segment_index: int) -> int:
        return int(self.header[f"Segment{segment_index}_LabelValue"])

    def parse_id_from_header_entry(self, key_name: str) -> int:
        match = re.search(r"Segment(\d+)_(Name|ID)", key_name)
        if match:
            id = int(match.group(1))
            return id
        else:
            raise ValueError(f"Invalid segment format. Provided key: {key_name}")

    def find_segment_index(self, label_name: str) -> int:
        for key, value in self.header.items():
            if not isinstance(value, str):
                continue

            # Only process SegmentX_ID or SegmentX_Name
            if "Segment" not in key or ("ID" not in key and "Name" not in key):
                continue

            if value == label_name:
                segment_id = self.parse_id_from_header_entry(key)
                return segment_id

        raise SegmentationNameNotFoundError(
            f"Label '{label_name}' not found in {self.segmentation_path}"
        )

    def get_slice(self, segment_name: str, index: int, plane: str, color: str) -> Mesh:
        assert plane in ["x", "y", "z"], "Plane must be 'x', 'y' or 'z'"

        segment_index = self.find_segment_index(segment_name)
        segment_label_value = self.get_segment_label_value(segment_index)
        segment_volume = self.get_volume_from_segment_name(segment_name)

        # print(f"segment name: {segment_name} segment index: {segment_index}")
        # print(f"Segment Label Value: {segment_label_value}")

        if plane == "x":
            seg_slice = segment_volume.xslice(index)
        elif plane == "y":
            seg_slice = segment_volume.yslice(index)
        elif plane == "z":
            seg_slice = segment_volume.zslice(index)
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Set color of segment
        lut = colors.build_lut(
            [
                (0, (0, 0, 1), 0.0),  # everything else transparent
                (segment_label_value, color),
            ],
            vmin=0,
            vmax=segment_label_value,
        )
        seg_slice.cmap(lut)
        seg_slice.alpha(0.3)

        return seg_slice
