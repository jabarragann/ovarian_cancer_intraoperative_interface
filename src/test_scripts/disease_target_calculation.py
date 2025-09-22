import sys
from typing import Any

sys.path.append("./src")  # to import from parent dir

from QuadrantInformation import QuadrantsInformation
from pathlib import Path
import numpy as np
from vedo import Volume
from scipy import ndimage


from SegmentationManager import SegmentationManager


def load_ct_scans_regions(
    data_path: Path,
) -> dict[QuadrantsInformation, Volume]:
    regions_path = data_path / "regions"
    regions_dict: dict[QuadrantsInformation, Volume] = {}
    for region_file in regions_path.glob("*.seg.nrrd"):
        quadrant_info = QuadrantsInformation.from_file_name(region_file)
        regions_dict[quadrant_info] = Volume(region_file)

        # print(f"Loaded {quadrant_info.name}")
        # if quadrant_info == QuadrantsInformation.PELVIC_REGION:
        #     break

    return regions_dict


def compute_target(
    disease_volume: np.ndarray, region_volume: np.ndarray, lymph_node_label_value: int
) -> list[dict[str, Any]]:
    # print("Computing target")
    # print(f"label value {lymph_node_label_value}")
    # print(f"disease volume shape {disease_volume.shape}")
    # print(f"region volume shape {region_volume.shape}")
    # print(f"max val disease volume {disease_volume.max()}")
    # print(f"max val region volume {region_volume.max()}")

    region_volume = region_volume.astype(bool)
    result_masked = disease_volume * region_volume == lymph_node_label_value

    structure = ndimage.generate_binary_structure(3, 2)  # 26-connectivity
    labeled, n = ndimage.label(result_masked, structure)  # type: ignore

    # print(labeled)
    # print(f"n clusters {n}")
    clusters: list[dict[str, Any]] = []

    for idx in range(1, n + 1):
        coords = np.argwhere(labeled == idx)  # voxel coords (z,y,x)
        centroid = coords.mean(axis=0)  # centroid in voxels
        radius = np.linalg.norm(coords - centroid, axis=1).max()
        clusters.append(
            {
                "cluster_id": idx,
                "centroid_vox": centroid,
                "radius_vox": radius,
                "voxel_count": len(coords),
            }
        )
        # print(clusters[-1])

    sorted_clusters = sorted(clusters, key=lambda c: c["radius_vox"], reverse=True)

    return sorted_clusters


def main():
    ## Setup
    data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

    patient_id = 6
    complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
    seg_path = complete_path / "radiologist_annotations.seg.nrrd"

    vedo_segment_loader = SegmentationManager(seg_path)

    regions_dict = load_ct_scans_regions(complete_path)

    label_values: dict[str, tuple[int, Volume]] = {}
    for segment_name in ["primary", "lymph node", "carcinosis"]:
        label_value = vedo_segment_loader.get_segment_label_value_from_name(
            segment_name
        )
        volume = vedo_segment_loader.get_volume_from_segment_name(segment_name)
        label_values[segment_name] = (label_value, volume)

        print(f"Segment {segment_name} has label value {label_value}")

    central_region_volume = regions_dict[QuadrantsInformation.LEFT_FLANK_AND_BOWEL]
    print(f"lymph node label value {label_values['lymph node'][0]}")
    compute_target(
        label_values["lymph node"][1].tonumpy(),
        central_region_volume.tonumpy(),
        label_values["lymph node"][0],
    )

    print("EXIT!")

    dict_clusters: dict[QuadrantsInformation, dict[str, Any]] = {}

    for region, region_vol in regions_dict.items():
        print(f"Calculating clusters in region {region.name}...")
        dict_clusters[region] = {}
        for disease in ["primary", "lymph node", "carcinosis"]:
            clusters = compute_target(
                label_values[disease][1].tonumpy(),
                region_vol.tonumpy(),
                label_values[disease][0],
            )

            biggest_cluster = clusters[0] if len(clusters) > 0 else None
            print(f"Biggest cluster {disease} - {biggest_cluster}")
            dict_clusters[region][disease] = biggest_cluster


if __name__ == "__main__":
    main()
