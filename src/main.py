import time
from collections import namedtuple
from collections.abc import MutableSequence
from functools import wraps
from pathlib import Path

import numpy as np
from vedo import Light, Line, Mesh, Plotter, Text2D, Volume, colors

from DiseaseClusterManager import DiseaseClusterManager
from QuadrantInformation import (
    QuadrantsInformation,
    compute_center,
    load_centers,
    save_centers,
)
from SegmentationManager import SegmentationManager


def time_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        print(f"{self.__class__.__name__}.__init__ took {end - start:.6f} seconds")
        return result

    return wrapper


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


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


# def text_generator(
#     quadrant_name: str, carcinosis_count: int, lymph_node_count: int
# ) -> str:
#     usage_text = (
#         f"Anatomical region: {quadrant_name}\n"
#         f"Carcinosis count: {carcinosis_count}\n"
#         f"Lymph node count: {lymph_node_count}\n"
#         "                                                       \n"
#         "\n"
#         # "Anatomical region: Left Flank and Bowel Resection (BR) \n" # longest text
#     )

#     return usage_text


class CT_Viewer(Plotter):
    @time_init
    def __init__(self, enable_3d_view: bool = True):
        self.enable_3d_view = enable_3d_view

        kwargs = {"sharecam": False, "size": (1200, 800)}
        kwargs.update({"bg": "black", "bg2": "black"})

        super().__init__(shape=(2, 4), title="CT Viewer", **kwargs)
        self.interactor.RemoveObservers("KeyPressEvent")  # type: ignore
        self.set_layout()

        ## State variables
        self.target_voxel = [248, 268, 176]
        self.active_quadrant_id = 6
        self.quadrant_volumes_dict: dict[QuadrantsInformation, Volume] = {}
        self.quadrant_slices_dict: dict[QuadrantsInformation, Mesh] = {}

        self.slices_viewport_objects: dict[str, list[Mesh]] = {}
        self.all_slices, self.all_objects = self.setup_viewer()

        self.add_callback("KeyPress", self.on_key_press)

        self.update_slices_viewports(self.target_voxel)
        # self.at(1).show(
        #     self.disease_slice["coronal"], camera=self.camera_params_slices["coronal"]
        # )
        # self.at(2).show(
        #     self.disease_slice["sagittal"], camera=self.camera_params_slices["sagittal"]
        # )
        # self.at(3).show(
        #     self.disease_slice["axial"], camera=self.camera_params_slices["axial"]
        # )

        self.at(4).show(self.all_slices, camera=self.camera_params_regions)
        self.at(5).show(self.all_objects, camera=self.camera_params_3d)

        ## TEXT LABELS
        self.setup_text_labels()

    def set_layout(self):
        # ratios
        col_ratios = [3, 3, 3]
        # row_ratios = [1, 4, 4, 1] --> For reference only
        col_fracs = [c / sum(col_ratios) for c in col_ratios]
        col_fracs_sum = [sum(col_fracs[:i]) for i in range(len(col_fracs))]

        ## Define viewports helper to compute viewport (xmin, ymin, xmax, ymax)
        b = 0.005  # small border between viewports
        top_border = 0.90

        # row 1
        self.renderers[0].SetViewport([0.0, top_border, 1.0, 1.0])
        # row 2
        self.renderers[1].SetViewport(
            [col_fracs_sum[0] + b, 0.5, col_fracs_sum[1] - b, top_border]
        )
        self.renderers[2].SetViewport(
            [col_fracs_sum[1] + b, 0.5, col_fracs_sum[2] - b, top_border]
        )
        self.renderers[3].SetViewport([col_fracs_sum[2] + b, 0.5, 1.0 - b, top_border])
        # row 3
        self.renderers[4].SetViewport([0.0, 0.1, 0.5, 0.5])
        self.renderers[5].SetViewport([0.5, 0.1, 1.0, 0.5])
        # row 4
        self.renderers[6].SetViewport([0.0, 0.0, 1.0, 0.1])
        # Move out of the way the unused viewport
        self.renderers[7].SetViewport([1.0, 1.0, 2.0, 2.0])

    def create_slices_cameras(self, voxel_pos_in_world, dist_to_plane=400):
        # camera for slice panes
        # (("coronal", "y"), ("sagittal", "x"), ("axial", "z"))
        camera_params_slices: dict[str, dict[str, list[float]]] = {}
        camera_params_slices["coronal"] = {
            "pos": [
                voxel_pos_in_world[0],
                voxel_pos_in_world[1] - dist_to_plane,
                voxel_pos_in_world[2],
            ],
            "focalPoint": voxel_pos_in_world,
            "viewup": [0, 0, 1],
        }
        camera_params_slices["sagittal"] = {
            "pos": [
                voxel_pos_in_world[0] + dist_to_plane,
                voxel_pos_in_world[1],
                voxel_pos_in_world[2],
            ],
            "focalPoint": voxel_pos_in_world,
            "viewup": [0, 0, 1],
        }
        ## Warning:
        ## For some reason precisely aligning with the volume center will the camera go black.
        camera_params_slices["axial"] = {
            "pos": [
                voxel_pos_in_world[0] + 0.001,
                voxel_pos_in_world[1],
                voxel_pos_in_world[2] - dist_to_plane,
            ],
            "focalPoint": voxel_pos_in_world,
            "viewup": [0, -1, 0],
        }

        return camera_params_slices

    def update_slices_viewports(self, target_in_voxel):
        viewport_to_view = {1: "coronal", 2: "sagittal", 3: "axial"}
        view_to_ax = {"coronal": "y", "sagittal": "x", "axial": "z"}

        ## Calculate target world
        target_in_world = [0.0, 0.0, 0.0]
        self.ct_volume.dataset.TransformContinuousIndexToPhysicalPoint(
            target_in_voxel,
            target_in_world,  # type: ignore
        )
        ## calculate camera params
        self.slices_camera_params = self.create_slices_cameras(target_in_world)

        ## Remove old actors from each viewport
        for i in range(1, 4):
            self.at(i).remove(*self.slices_viewport_objects[viewport_to_view[i]])

        ## Create new slices
        self.slices_viewport_objects = self.create_disease_slice(
            self.ct_volume, target_in_voxel
        )

        ## Add new objects to each viewport
        for i, view_name in viewport_to_view.items():
            this_view_objs = self.slices_viewport_objects[viewport_to_view[i]]
            crosshair = create_crosshair(
                target_in_world, this_view_objs[0], view_to_ax[view_name]
            )
            self.slices_viewport_objects[viewport_to_view[i]].extend(crosshair)

            self.at(i).add(*self.slices_viewport_objects[viewport_to_view[i]])
            self.at(i).camera = self.slices_camera_params[view_name]

    def setup_text_labels(self):
        ## Static labels.
        offset = 0.33 / 2
        coronal_text = Text2D(
            "Coronal",
            pos=(0.33 - offset, 0.5),  # type: ignore
            s=1.4,
            c="white",
        )
        sagittal_text = Text2D(
            "Sagittal",
            pos=(0.66 - offset, 0.5),  # type: ignore
            s=1.4,
            c="white",
        )
        axial_text = Text2D("Axial", pos=(0.99 - offset, 0.5), s=1.4, c="white")  # type: ignore
        self.at(0).add(coronal_text)
        self.at(0).add(sagittal_text)
        self.at(0).add(axial_text)

        ## Dynamic text labels
        self.station_text = (
            "Region: " + QuadrantsInformation.from_id(self.active_quadrant_id).name
        )
        self.station_text_vedo = Text2D(
            self.station_text,
            pos="top-left",
            s=1.1,
            c="white",
            # bg="yellow",
            # alpha=0.25,
        )
        self.at(4).add(self.station_text_vedo)

        # Disease text
        self.disease_text_vedo = Text2D(
            "Disease findings in region", pos="top-left", s=1.1, c="white"
        )
        self.at(6).add(self.disease_text_vedo)

        offset = 0.33 / 4

        active_quadrant = QuadrantsInformation.from_id(self.active_quadrant_id)
        carcinosis_present = self.disease_cluster_manager.disease_prescence(
            active_quadrant, "carcinosis"
        )
        lymph_node_present = self.disease_cluster_manager.disease_prescence(
            active_quadrant, "lymph node"
        )
        primary_present = self.disease_cluster_manager.disease_prescence(
            active_quadrant, "primary"
        )

        self.carcinosis_text_vedo = Text2D(
            "Carcinosis: Yes" if carcinosis_present else "Carcinosis: No",
            pos=(0.0 + offset, 0.5),  # type: ignore
            s=1.1,
            c="white",
        )
        self.lymph_node_text_vedo = Text2D(
            "Lymph Nodes: Yes" if lymph_node_present else "Lymph Nodes: No",
            pos=(0.33 + offset, 0.5),  # type: ignore
            s=1.1,
            c="white",
        )
        self.primary_text_vedo = Text2D(
            "Primary Tumor: Yes" if primary_present else "Primary Tumor: No",
            pos=(0.66 + offset, 0.5),  # type: ignore
            s=1.1,
            c="white",
        )
        self.at(6).add(
            self.carcinosis_text_vedo, self.lymph_node_text_vedo, self.primary_text_vedo
        )

    def update_disease_text_labels(self, active_quadrant: QuadrantsInformation):
        self.station_text = "Region: " + active_quadrant.name
        self.station_text_vedo.text(self.station_text)

        carcinosis_present = self.disease_cluster_manager.disease_prescence(
            active_quadrant, "carcinosis"
        )
        lymph_node_present = self.disease_cluster_manager.disease_prescence(
            active_quadrant, "lymph node"
        )
        primary_present = self.disease_cluster_manager.disease_prescence(
            active_quadrant, "primary"
        )

        self.carcinosis_text_vedo.text(
            "Carcinosis: Yes" if carcinosis_present else "Carcinosis: No"
        )
        self.lymph_node_text_vedo.text(
            "Lymph Nodes: Yes" if lymph_node_present else "Lymph Nodes: No"
        )
        self.primary_text_vedo.text(
            "Primary Tumor: Yes" if primary_present else "Primary Tumor: No"
        )

    def load_volumes(self, data_path, patient_id):
        ct_path = data_path / f"raw_scans_patient_{patient_id:02d}.nrrd"
        ct = Volume(ct_path)

        # seg = Volume(complete_path / "regions" / "pelvic_region_quadrant.seg.nrrd")
        segmentation_manager = SegmentationManager(
            data_path / "radiologist_annotations.seg.nrrd"
        )
        segmentation_manager.load_volumes_to_cache(
            ["primary", "lymph node", "carcinosis"]
        )

        region_seg_dict = load_ct_scans_regions(data_path)

        return ct, segmentation_manager, region_seg_dict

    def setup_viewer(self):
        data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")
        patient_id = 6
        data_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
        runtime_path = data_path / "interface_runtime"

        ## Panel 1 assets - Load slices
        index = 270
        self.ct_volume, self.segmentation_manager, self.quadrant_volumes_dict = (
            self.load_volumes(data_path, patient_id)
        )
        self.disease_cluster_manager = DiseaseClusterManager(
            regions_dict=self.quadrant_volumes_dict,
            disease_dict=self.segmentation_manager.get_cache_volume_dict(),
            root_path=runtime_path,
        )

        # Create slices
        ct_slice = self.slice_intensity_volume(self.ct_volume, index=index)
        self.quadrant_slices_dict = self.create_quadrant_slices(index=index)
        self.slices_viewport_objects = self.create_disease_slice(
            self.ct_volume, target_voxel=self.target_voxel
        )

        seg_slices_list = [s for s in self.quadrant_slices_dict.values()]

        self.quadrant_center_dict = self.load_region_centers(runtime_path)

        # Set active quadrant
        quadrant = QuadrantsInformation.from_id(self.active_quadrant_id)
        self.quadrant_slices_dict[quadrant].alpha(0.4)

        # ct_vis = (
        #     ct_volume.clone().cmap("gray").alpha([[-1000, 0], [200, 0.2], [1000, 0.7]])
        # )

        ## Panel 2 assets
        vol_bounds = self.ct_volume.bounds()  # xmin,xmax, ymin,ymax, zmin,zmax
        vol_center = self.ct_volume.center()
        self.camera_params_3d, self.camera_params_slices, self.camera_params_regions = (
            create_camera_params(vol_center, vol_bounds)
        )
        # print(f"vol_bounds: {vol_bounds}")
        # print(f"vol_center: {vol_center}")
        # print(f"vol origin {ct_volume.origin()}")

        lights_list = create_lights(vol_center, vol_bounds)

        ## 3D rendering window
        all_objects = []
        meshes_list = []
        meshes_dict = {}
        if self.enable_3d_view:  # Turn off 3D rendering to sped up development
            meshes_list, meshes_dict = load_meshes(
                "/home/juan95/research/3dreconstruction/slicer_scripts/output"
            )

            meshes_disease_list, meshes_disease_dict = load_meshes(
                "/home/juan95/research/3dreconstruction/slicer_scripts/output_disease"
            )

            set_mesh_visual_properties(meshes_dict)
            set_disease_visual_properties(meshes_disease_dict)

            all_objects = meshes_list + lights_list
            all_objects.append(meshes_disease_dict["lymph node"])
            all_objects.append(meshes_disease_dict["carcinosis"])

        all_slices = [ct_slice, seg_slices_list]

        return all_slices, all_objects

    def slice_intensity_volume(self, ct_volume, index, plane="y", W=400, L=50):
        """
        Window level for soft tissue
        W=400
        L=50
        """
        assert plane in ["x", "y", "z"], "Plane must be 'x', 'y' or 'z'"
        if plane == "x":
            ct_slice = ct_volume.xslice(index)
        elif plane == "y":
            ct_slice = ct_volume.yslice(index)
        elif plane == "z":
            ct_slice = ct_volume.zslice(index)

        vmin = L - W / 2
        vmax = L + W / 2

        ct_slice.cmap("gray", vmin=vmin, vmax=vmax)

        return ct_slice

    def create_disease_slice(
        self, ct_volume: Volume, target_voxel: list[int]
    ) -> dict[str, list[Mesh]]:
        slices_dict: dict[str, list[Mesh]] = {}

        SegmentInfo = namedtuple("Segment", ["name", "color"])
        lymph_segment = SegmentInfo("lymph node", "#9725e8")
        primary_segment = SegmentInfo("primary", "#45e825")
        carcinosis_segment = SegmentInfo("carcinosis", "#f0e964")

        all_segments = (lymph_segment, primary_segment, carcinosis_segment)
        orthogonal_planes = (
            ("sagittal", "x", target_voxel[0]),
            ("coronal", "y", target_voxel[1]),
            ("axial", "z", target_voxel[2]),
        )
        for plane_name, plane, index in orthogonal_planes:
            slices_dict[plane_name] = []
            ct_slice = self.slice_intensity_volume(ct_volume, index=index, plane=plane)
            slices_dict[plane_name].append(ct_slice)

            for segment in all_segments:
                segment_slice = self.segmentation_manager.get_slice(
                    segment.name, index, plane=plane, color=segment.color
                )
                slices_dict[plane_name].append(segment_slice)

        return slices_dict

    def create_quadrant_slices(self, index) -> dict[QuadrantsInformation, Mesh]:
        slices_dict = {}
        for region, volume in self.quadrant_volumes_dict.items():
            seg_slice = volume.yslice(index)
            lut = colors.build_lut(
                [
                    (0, (0, 0, 0), 0.0),  # background (scalar, color, alpha)
                    (1, region.color),  # segmentation
                ],
                vmin=0.7,
                vmax=1.2,
                below_alpha=0.0,  # type: ignore
                above_alpha=1.0,  # type: ignore
            )
            seg_slice.cmap(lut)
            seg_slice.alpha(0.0)

            slices_dict[region] = seg_slice

        return slices_dict

        # ## METHOD 1
        # seg_slice = seg_volume.yslice(index)
        # lut = colors.build_lut(
        #     [
        #         (0, "black"),  # background
        #         (1, regions_color_palette[0]),  # segmentation
        #     ],
        # )

        # # Apply LUT to your slice
        # seg_slice.cmap(lut)
        # seg_slice.alpha(0.4)

        # return [seg_slice]

    def load_region_centers(
        self, runtime_path
    ) -> dict[QuadrantsInformation, np.ndarray]:
        centers_dict = {}

        try:
            centers_dict = load_centers(runtime_path)
        except FileNotFoundError:
            print("Centers file not found. Computing region centers...")
            for region, volume in self.quadrant_volumes_dict.items():
                centers_dict[region] = compute_center(volume)
            save_centers(centers_dict, runtime_path)

        return centers_dict

    def on_key_press(self, evt):
        """Handle keyboard events"""
        key = evt.keypress
        if key == "q":
            self.break_interaction()
        elif key.lower() == "t":
            print("Help key pressed")

        elif key.lower() == "h":
            print("position", self.at(4).camera.GetPosition())  # type: ignore
            print("focal", self.at(4).camera.GetFocalPoint())  # type: ignore
            print("viewup", self.at(4).camera.GetViewUp())  # type: ignore

        elif is_int(key):
            idx = int(key)
            if idx < 7 and idx >= 0:
                ## Adapt region view port
                # print(f"activating {QuadrantsInformation.from_id(idx).name}")
                current_quadrant = QuadrantsInformation.from_id(self.active_quadrant_id)
                self.quadrant_slices_dict[current_quadrant].alpha(0.0)

                new_quadrant = QuadrantsInformation.from_id(idx)
                self.quadrant_slices_dict[new_quadrant].alpha(0.4)
                self.active_quadrant_id = idx

                self.station_text = "Region: " + new_quadrant.name
                self.station_text_vedo.text(self.station_text)

                ## Adapt 3D view port
                self.position_3d_camera_in_region(new_quadrant)

                ## Adapt slices view port
                self.target_voxel = self.calculate_new_target(new_quadrant)
                # print(f"New target voxel {self.target_voxel}")
                self.update_slices_viewports(self.target_voxel)

                self.update_disease_text_labels(new_quadrant)

            self.render()

    def calculate_new_target(self, new_quadrant: QuadrantsInformation) -> list[int]:
        disease_to_highlight = ["lymph node", "primary", "carcinosis"]
        for disease in disease_to_highlight:
            vol_center = self.disease_cluster_manager.get_centroid_of_largest_cluster(
                new_quadrant, disease
            )
            if vol_center is not None:
                return vol_center.tolist()

        # If no annotations are found use region center
        print(f"no disease detected in {new_quadrant.name}, using region center")
        vol_center = self.quadrant_center_dict[new_quadrant]
        vol_center[1] = 264  # TODO: fix hardcoded value
        target_voxel = vol_center.tolist()

        return target_voxel

    def position_3d_camera_in_region(self, new_quadrant: QuadrantsInformation):
        """
        Change 3D camera position
        index 0 --> Left-right
        index 1 --> Anterior-posterior
        index 2 --> inferior-superior
        """

        volume = self.quadrant_volumes_dict[new_quadrant]
        vol_center = self.quadrant_center_dict[new_quadrant]

        anterior_plane_center = vol_center
        anterior_plane_center[1] = int(volume.shape[1])

        center_in_world: MutableSequence[float]
        center_in_world = [0, 0, 0]
        volume.dataset.TransformContinuousIndexToPhysicalPoint(
            anterior_plane_center,  # type: ignore
            center_in_world,
        )
        center_in_world_np = np.array(center_in_world)

        # New camera position
        new_cam_pos1 = center_in_world_np.copy()
        vol_bounds = volume.bounds()
        new_cam_pos1[1] = vol_bounds[2] - 200

        self.at(5).camera.SetPosition(new_cam_pos1)  # type: ignore
        self.at(5).camera.SetFocalPoint(center_in_world)  # type: ignore
        self.at(5).camera.SetViewUp([0, 0, 1])  # type: ignore
        self.at(5).renderer.ResetCameraClippingRange()  # type: ignore


def create_camera_params(vol_center, vol_bounds):
    # Bounding box --> [xmin,xmax, ymin,ymax, zmin,zmax].

    slice_focal_point = [vol_center[0], vol_center[1], vol_center[2]]
    camera_params_3d = {
        "pos": [vol_center[0], vol_bounds[2] - 550, vol_center[2]],
        "focalPoint": slice_focal_point,
        "viewup": [0, 0, 1],
    }

    camera_params_regions = {
        "pos": [-238.57000083409514, -798.5, 1178.9672953406998],
        "focalPoint": [-238.57000083409514, -0.48538350000004016, 1178.9672953406998],
        "viewup": [0, 0, 1],
    }

    # camera for slice panes
    # (("coronal", "y"), ("sagittal", "x"), ("axial", "z"))
    camera_params_slices: dict[str, dict[str, list[float]]] = {}
    camera_params_slices["coronal"] = {
        "pos": [vol_center[0], vol_bounds[2] - 550, vol_center[2]],
        "focalPoint": slice_focal_point,
        "viewup": [0, 0, 1],
    }
    camera_params_slices["sagittal"] = {
        "pos": [vol_bounds[1] + 550, vol_center[1], vol_center[2]],
        "focalPoint": slice_focal_point,
        "viewup": [0, 0, 1],
    }
    ## Warning:
    ## For some reason precisely aligning with the volume center will the camera go black.
    camera_params_slices["axial"] = {
        "pos": [vol_center[0] + 0.001, vol_center[1], vol_bounds[4] - 500],
        "focalPoint": slice_focal_point,
        "viewup": [0, -1, 0],
    }
    # print("axial pose:", camera_params_slices["axial"]["pos"])
    # print("coronal pose:", camera_params_slices["coronal"]["pos"])
    # print("sagittal pose:", camera_params_slices["sagittal"]["pos"])

    return camera_params_3d, camera_params_slices, camera_params_regions


def parse_mtl(mtl_file: Path):
    """Parse .mtl file and return a dict of {material_name: (r,g,b)} diffuse colors."""
    materials = {}
    current = None
    with open(mtl_file) as f:
        for line in f:
            if line.startswith("newmtl"):
                current = line.split()[1]
            elif line.startswith("Kd") and current:
                r, g, b = map(float, line.split()[1:4])
                materials[current] = (r, g, b)
    return materials


def load_mesh(obj_path: Path, load_mtl: bool) -> Mesh:
    # path = Path("/home/juan95/research/3dreconstruction/slicer_scripts/output")
    # mesh_path = path / "liver.obj"

    mesh = Mesh(obj_path)

    # Load material
    mtl_path = obj_path.with_suffix(".mtl")
    if mtl_path.exists() and load_mtl:
        mats = parse_mtl(mtl_path)

        if mats:
            color = next(iter(mats.values()))
            mesh.c(color)
    else:
        # mesh.c("random").alpha(1.0)
        pass

    mesh.lighting("plastic")
    mesh.properties.SetAmbient(0.4)
    mesh.properties.SetSpecular(0.8)
    mesh.properties.SetSpecularPower(10)

    return mesh


def load_meshes(
    folder: str, pattern: str = "*.obj"
) -> tuple[list[Mesh], dict[str, Mesh]]:
    folder_path = Path(folder)
    meshes_list = []
    meshes_dict = {}
    for mesh_path in folder_path.glob(pattern):
        m = load_mesh(mesh_path, load_mtl=True)

        # m = Mesh(str(mesh_path))
        # m.c("random").alpha(0.6)  # random color, semi-transparent
        meshes_list.append(m)
        mesh_name = mesh_path.with_suffix("").name
        meshes_dict[mesh_name] = m

    return meshes_list, meshes_dict


def create_crosshair(
    target_in_world: list[float], slice_mesh: Mesh, plane: str, size: float = 0.02
):
    """
    target: (x,y,z) in world coords
    slice_mesh: vedo.Mesh of the slice (to get bounds)
    plane: 'x', 'y', or 'z' – which axis is constant for this slice
    """
    x0, x1, y0, y1, z0, z1 = slice_mesh.bounds()

    cx, cy, cz = target_in_world

    if plane == "z":  # axial -> lines in X,Y
        dx = (x1 - x0) * size
        dy = (y1 - y0) * size
        hline = Line((cx - dx, cy, cz), (cx + dx, cy, cz), c="yellow", lw=2)
        vline = Line((cx, cy - dy, cz), (cx, cy + dy, cz), c="yellow", lw=2)

    elif plane == "y":  # coronal -> lines in X,Z
        dx = (x1 - x0) * size
        dz = (z1 - z0) * size
        hline = Line((cx - dx, cy, cz), (cx + dx, cy, cz), c="yellow", lw=2)
        vline = Line((cx, cy, cz - dz), (cx, cy, cz + dz), c="yellow", lw=2)

    elif plane == "x":  # sagittal -> lines in Y,Z
        dy = (y1 - y0) * size
        dz = (z1 - z0) * size
        hline = Line((cx, cy - dy, cz), (cx, cy + dy, cz), c="yellow", lw=2)
        vline = Line((cx, cy, cz - dz), (cx, cy, cz + dz), c="yellow", lw=2)

    else:
        raise ValueError("plane must be 'x', 'y' or 'z'")

    return [hline, vline]


def create_lights(vol_center, vol_bounds):
    light1_pos = [vol_center[0], vol_bounds[2] - 100, vol_center[2]]
    light2_pos = [vol_bounds[0] - 50, vol_center[1], vol_center[2]]
    light3_pos = [vol_bounds[1] + 50, vol_center[1], vol_center[2]]

    light1 = Light(pos=light1_pos, focal_point=vol_center, c="white", intensity=1.0)  # type: ignore
    light2 = Light(pos=light2_pos, focal_point=vol_center, c="white", intensity=1.0)  # type: ignore
    light3 = Light(pos=light3_pos, focal_point=vol_center, c="white", intensity=1.0)  # type: ignore

    return [light1, light2, light3]


def set_mesh_visual_properties(meshes_dict):
    for key, mesh in meshes_dict.items():
        mesh.lighting("plastic")
        mesh.properties.SetAmbient(0.3)
        mesh.properties.SetSpecular(0.6)
        mesh.properties.SetSpecularPower(5)

    meshes_dict["liver"].alpha(0.4)


def set_disease_visual_properties(mesh_dict):
    for key, mesh in mesh_dict.items():
        mesh.lighting("glossy")
        mesh.properties.SetAmbient(0.6)
        mesh.properties.SetSpecular(0.9)
        mesh.properties.SetSpecularPower(14)


def main():
    viewer = CT_Viewer()
    viewer.interactive().close()


if __name__ == "__main__":
    main()
