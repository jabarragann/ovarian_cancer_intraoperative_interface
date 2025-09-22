from dataclasses import dataclass, field
from functools import wraps
import pickle
import time


from QuadrantInformation import QuadrantsInformation
from pathlib import Path
import numpy as np
from vedo import Volume
from scipy import ndimage


from VedoSegmentLoader import SegmentationLoaderManager


@dataclass
class ClusterInfo:
    cluster_id: int
    centroid_vox: np.ndarray
    radius_vox: float
    voxel_count: int


@dataclass
class DiseaseClusterManager:
    root_path: Path
    regions_dict: dict[QuadrantsInformation, Volume]
    disease_dict: dict[str, tuple[int, Volume]]
    dict_clusters: dict[QuadrantsInformation, dict[str, list[ClusterInfo]]] = field(
        init=False
    )

    def __post_init__(self):
        try:
            self.dict_clusters = self.load_clusters()
            print(
                f"Loading existing clusters from {self.root_path / 'dict_clusters.pkl'}"
            )
        except FileNotFoundError:
            print("No existing clusters found, calculating clusters.")
            self.dict_clusters = self.calculate_clusters()
            self.save_clusters()

    def calculate_clusters(
        self,
    ) -> dict[QuadrantsInformation, dict[str, list[ClusterInfo]]]:
        dict_clusters: dict[QuadrantsInformation, dict[str, list[ClusterInfo]]] = {}
        for region, region_vol in self.regions_dict.items():
            dict_clusters[region] = {}
            for disease, (label_value, disease_vol) in self.disease_dict.items():
                dict_clusters[region][disease] = []

                clusters = self.compute_target(
                    disease_vol.tonumpy(),
                    region_vol.tonumpy(),
                    label_value,
                )
                dict_clusters[region][disease].extend(clusters)

        return dict_clusters

    def load_clusters(self) -> dict[QuadrantsInformation, dict[str, list[ClusterInfo]]]:
        with open(self.root_path / "dict_clusters.pkl", "rb") as f:
            return pickle.load(f)

    def save_clusters(self):
        if self.root_path.exists() is False:
            self.root_path.mkdir(parents=True, exist_ok=True)

        with open(self.root_path / "dict_clusters.pkl", "wb") as f:
            pickle.dump(self.dict_clusters, f)

    def compute_target(
        self,
        disease_volume: np.ndarray,
        region_volume: np.ndarray,
        segment_label_value: int,
    ) -> list[ClusterInfo]:
        """
        Computes disease cluster within each region. Returns sorted list of clusters by size.
        """

        region_volume = region_volume.astype(bool)
        result_masked = disease_volume * region_volume == segment_label_value

        structure = ndimage.generate_binary_structure(3, 2)  # 26-connectivity
        labeled, n = ndimage.label(result_masked, structure)  # type: ignore

        clusters: list[ClusterInfo] = []
        if n > 0:
            for idx in range(1, n + 1):
                coords = np.argwhere(labeled == idx)  # voxel coords (z,y,x)
                centroid = coords.mean(axis=0)  # centroid in voxels
                radius = np.linalg.norm(coords - centroid, axis=1).max()
                clusters.append(
                    ClusterInfo(
                        cluster_id=idx,
                        centroid_vox=centroid,
                        radius_vox=radius,
                        voxel_count=len(coords),
                    )
                )

            sorted_clusters = sorted(clusters, key=lambda c: c.radius_vox, reverse=True)
        else:
            sorted_clusters = []

        return sorted_clusters

    def cluster_report(self):
        for region in self.dict_clusters.keys():
            for disease in self.disease_dict.keys():
                current_clusters = self.dict_clusters[region][disease]
                print(
                    f"Region: {region.name} has {len(current_clusters)} {disease} clusters"
                )
                for cluster in current_clusters:
                    print(
                        f"Radius (vox): {cluster.radius_vox:.2f}, Centroid (vox): {cluster.centroid_vox}"
                    )

    def disease_prescence(self, region: QuadrantsInformation, disease: str) -> bool:
        return len(self.dict_clusters[region][disease]) > 0

    def get_centroid_of_largest_cluster(
        self, region: QuadrantsInformation, disease: str
    ) -> np.ndarray | None:
        clusters = self.dict_clusters[region][disease]
        if len(clusters) == 0:
            return None

        return clusters[0].centroid_vox.astype(int)


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


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


@timeit
def main():
    ## Setup
    data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

    patient_id = 6
    complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
    seg_path = complete_path / "radiologist_annotations.seg.nrrd"

    vedo_segment_loader = SegmentationLoaderManager(seg_path)
    vedo_segment_loader.load_volumes_to_cache(["primary", "lymph node", "carcinosis"])

    regions_dict = load_ct_scans_regions(complete_path)
    disease_dict = vedo_segment_loader.get_cache_volume_dict()

    cluster_path = complete_path / "interface_runtime"

    cluster_manager = DiseaseClusterManager(
        regions_dict=regions_dict, disease_dict=disease_dict, root_path=cluster_path
    )

    cluster_manager.cluster_report()

    test_region = QuadrantsInformation.LEFT_UPPER_QUADRANT
    print(
        f"Disease present in {test_region.name}: {cluster_manager.disease_prescence(test_region, 'lymph node')}"
    )
    biggest_cluster = cluster_manager.get_centroid_of_largest_cluster(
        test_region, "lymph node"
    )
    print(f"Centroid of largest cluster in {test_region.name}: {biggest_cluster}")


if __name__ == "__main__":
    main()
