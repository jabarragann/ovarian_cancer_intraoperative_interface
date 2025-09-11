from enum import Enum
from pathlib import Path
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
