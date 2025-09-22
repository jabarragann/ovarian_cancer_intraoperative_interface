import pickle
from enum import Enum
from pathlib import Path

import numpy as np
import seaborn as sns

regions_color_palette = sns.color_palette("Set2", 7).as_hex()


class QuadrantsInformation(Enum):
    CENTRAL_AND_BOWEL = (
        0,
        "Central and Bowel Resection (BR)",
        "central_quadrant",
        regions_color_palette[0],
    )
    LEFT_UPPER_QUADRANT = (
        1,
        "Left Upper Quadrant (LUQ)",
        "left_upper_quadrant",
        regions_color_palette[1],
    )
    UPPER_RIGHT_QUADRANT = (
        2,
        "Upper Right Quadrant (URQ)",
        "upper_right_quadrant",
        regions_color_palette[2],
    )
    LEFT_FLANK_AND_BOWEL = (
        3,
        "Left Flank and Bowel Resection (BR)",
        "left_flank_quadrant",
        regions_color_palette[3],
    )
    RIGHT_FLANK_AND_BOWEL = (
        4,
        "Right Flank and Bowel Resection",
        "right_flank_quadrant",
        regions_color_palette[4],
    )
    SMALL_BOWEL = (5, "Small bowel", "small_bowel_quadrant", regions_color_palette[5])
    PELVIC_REGION = (
        6,
        "Pelvic Region",
        "pelvic_region_quadrant",
        regions_color_palette[6],
    )

    def __init__(self, id: int, long_name: str, short_name: str, color: str):
        self._id = id
        self._name = long_name
        self._short_name = short_name
        self._color = color

    @classmethod
    def from_id(cls, id: int):
        for member in cls:
            if member.id == id:
                return member
        raise ValueError(f"Unknown quadrant ID: {id}")

    @classmethod
    def from_file_name(cls, filename: Path):
        """
        Assumes file extension is .seg.nrrd
        """
        name_no_suffix = filename.with_suffix("").with_suffix("").name
        for member in cls:
            if member.short_name == name_no_suffix:
                return member
        raise ValueError(f"Unknown quadrant: {name_no_suffix}")

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def short_name(self) -> str:
        return self._short_name

    @property
    def color(self) -> str:
        return self._color


def load_centers(path: Path):
    complete_path = path / "regions_center.pkl"
    if not Path(complete_path).exists():
        raise FileNotFoundError(f"Center file not found: {complete_path}")

    return pickle.load(open(complete_path, "rb"))


def save_centers(centers: dict[QuadrantsInformation, np.ndarray], runtime_path: Path):
    path = runtime_path / "regions_center.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(centers, f)


def compute_center(volume):
    seg_np = volume.tonumpy()

    indices = np.argwhere(seg_np == 1)
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)
    center_idx = ((min_idx + max_idx) // 2).astype(int)
    print(f"Center idx {center_idx}")

    return center_idx

    # # seg is a vedo.Volume
    # ijk = center_idx

    # # VTK has a method for this:
    # world_coords: MutableSequence[float]
    # world_coords = [0, 0, 0]
    # volume.dataset.TransformContinuousIndexToPhysicalPoint(ijk, world_coords)

    # return np.array(world_coords)


## Unused

# @dataclass
# class QuadrantInstance:
#     quadrant_info: QuadrantsInformation
#     volume: Volume
#     center: Union[np.ndarray, None] = None
#     slice: Union[Mesh, None] = None

#     def __post_init__(self):
#         self.compute_center()

#     def compute_center(self):
#         seg_np = self.volume.tonumpy()

#         indices = np.argwhere(seg_np == 1)
#         min_idx = indices.min(axis=0)
#         max_idx = indices.max(axis=0)
#         center_idx = ((min_idx + max_idx) // 2).astype(int)
#         print(f"Center idx {center_idx}")

#         # seg is a vedo.Volume
#         ijk = center_idx

#         # VTK has a method for this:
#         world_coords: MutableSequence[float]
#         world_coords = [0, 0, 0]
#         self.volume.dataset.TransformContinuousIndexToPhysicalPoint(ijk, world_coords)

#         self.center = np.array(world_coords)


# @dataclass
# class QuadrantManager:
#     instances: list[QuadrantInstance] = field(default_factory=list)

#     def add_instance(self, instance: QuadrantInstance):
#         self.instances.append(instance)

#     def remove_instance(self, instance: QuadrantInstance):
#         self.instances.remove(instance)

#     def get_instance(
#         self, quadrant_info: QuadrantsInformation
#     ) -> Union[QuadrantInstance, None]:
#         for instance in self.instances:
#             if instance.quadrant_info == quadrant_info:
#                 return instance
#         return None
