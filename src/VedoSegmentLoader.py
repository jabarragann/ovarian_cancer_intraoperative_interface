import re
import nrrd
import numpy as np
from vedo import Volume, Mesh
from pathlib import Path


class SegmentationNameNotFoundError(Exception):
    """Raised when a segmentation name is not found in SegmentationWrapper."""
    pass


class VedoSegmentLoader:
    def __init__(self, segmentation_path: Path):
        self.segmentation_path = segmentation_path
        if not segmentation_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")

        self.data, self.header = nrrd.read(
            str(segmentation_path)
        )

        self.spacing, self.origin = self.parse_header()
        self.dimension = self.header["dimension"]

        self.volume_dict : dict[str, Volume] = {}

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

    def get_volume_from_id(self, label_name: str) -> Volume:
        if label_name not in self.volume_dict:
            raw_data = self.get_data_from_id(label_name)
            self.volume_dict[label_name] = Volume(raw_data, spacing=self.spacing, origin=self.origin)

        return self.volume_dict[label_name]

    def get_data_from_id(self, label_name: str):
        """
        In case of overlapping segments, data might be split in multiple volumes.
        You need to use header information to figure out the right data volume.
        """
        if len(self.data.shape) > 3:
            layer = self.get_layer(label_name)
            return self.data[layer]
        elif len(self.data.shape) == 3:
            return self.data
        else:
            raise ValueError("Unsupported data shape")

    def get_layer(self, label_name: str) -> int:
        """For multi-layer nrrd files"""
        segment_id = self.find_segment_id(label_name)
        segment_layer = self.header[f"Segment{segment_id}_Layer"]
        return int(segment_layer)

    def get_id_from_string(self, key_name: str) -> int:
        match = re.search(r"Segment(\d+)_(Name|ID)", key_name)
        if match:
            id = int(match.group(1))
            return id
        else:
            raise ValueError(f"Invalid segment format. Provided key: {key_name}")

    def find_segment_id(self, label_name: str) -> int:
        for key, value in self.header.items():
            if not isinstance(value, str):
                continue

            # Only process SegmentX_ID or SegmentX_Name
            if "Segment" not in key or ("ID" not in key and "Name" not in key):
                continue

            if value == label_name:
                segment_id = self.get_id_from_string(key)
                return segment_id

        raise SegmentationNameNotFoundError(
            f"Label '{label_name}' not found in {self.segmentation_path}"
        )

