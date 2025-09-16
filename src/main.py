from collections.abc import MutableSequence
from functools import wraps
import time
import numpy as np
from vedo import Volume, colors, Mesh, Light, Text2D
from pathlib import Path
from vedo import Plotter
from QuadrantInformation import (
    QuadrantsInformation,
    compute_center,
    load_centers,
    save_centers,
)


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


def text_generator(
    quadrant_name: str, carcinosis_count: int, lymph_node_count: int
) -> str:
    usage_text = (
        f"Anatomical region: {quadrant_name}\n"
        f"Carcinosis count: {carcinosis_count}\n"
        f"Lymph node count: {lymph_node_count}\n"
        "                                                       \n"
        "\n"
        # "Anatomical region: Left Flank and Bowel Resection (BR) \n" # longest text
    )

    return usage_text


class CT_Viewer(Plotter):
    @time_init
    def __init__(self):
        kwargs = {"sharecam": False, "size": (1200, 800)}
        kwargs.update({"bg": "black", "bg2": "black"})

        super().__init__(shape=(2, 4), title="CT Viewer", **kwargs)
        self.interactor.RemoveObservers("KeyPressEvent")  # type: ignore
        self.set_layout()

        ## State variables
        self.active_quadrant = 6
        self.quadrant_volumes_dict: dict[QuadrantsInformation, Volume] = {}
        self.quadrant_slices_dict: dict[QuadrantsInformation, Mesh] = {}

        self.all_slices, self.all_objects, camera_params = self.setup_viewer()
        self.usage_text = text_generator(
            QuadrantsInformation.from_id(self.active_quadrant).name, 0, 0
        )

        self.add_callback("KeyPress", self.on_key_press)

        self.at(1).show(self.all_slices, camera=camera_params)
        self.at(4).show(self.all_slices, camera=camera_params)
        self.at(5).show(self.all_objects, camera=camera_params)

        self.text_handle = Text2D(
            self.usage_text,
            font="Calco",
            pos="top-left",
            s=1.2,
            bg="yellow",
            alpha=0.25,
        )
        self.at(6).add(self.text_handle)

    def set_layout(self):
        # ratios
        col_ratios = [3, 3, 3]
        # row_ratios = [1, 4, 4, 1] --> For reference only
        col_fracs = [c / sum(col_ratios) for c in col_ratios]
        col_fracs_sum = [sum(col_fracs[:i]) for i in range(len(col_fracs))]

        ## Define viewports helper to compute viewport (xmin, ymin, xmax, ymax)
        top_border = 0.90

        # row 1 
        self.renderers[0].SetViewport([0.0, top_border, 1.0, 1.0])
        # row 2
        self.renderers[1].SetViewport([ col_fracs_sum[0] , 0.5, col_fracs_sum[1] , top_border  ])
        self.renderers[2].SetViewport([ col_fracs_sum[1] , 0.5, col_fracs_sum[2] , top_border])
        self.renderers[3].SetViewport([ col_fracs_sum[2] , 0.5, 1.0, top_border])
        # row 3
        self.renderers[4].SetViewport([0.0, 0.1, 0.5, 0.5])
        self.renderers[5].SetViewport([0.5, 0.1, 1.0, 0.5])
        # row 4
        self.renderers[6].SetViewport([0.0, 0.0, 1.0, 0.1])
        # Move out of the way the unused viewport
        self.renderers[7].SetViewport([1.0, 1.0, 2.0, 2.0])

    def load_volumes(self, data_path):
        patient_id = 6
        complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
        ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"

        # print(complete_path)
        # print(complete_path.exists())
        # print(ct_path)
        # print(ct_path.exists())

        ct = Volume(ct_path)
        # seg = Volume(complete_path / "regions" / "pelvic_region_quadrant.seg.nrrd")
        disease_annotations = Volume(complete_path / "radiologist_annotations.seg.nrrd")
        region_seg_dict = load_ct_scans_regions(complete_path)

        return ct, disease_annotations, region_seg_dict

    def setup_viewer(self):
        data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

        ## Panel 1 assets - Load slices
        index = 270
        self.ct_volume, self.seg_volume, self.quadrant_volumes_dict = self.load_volumes(
            data_path
        )

        ct_slice = self.slice_intensity_volume(self.ct_volume, index=index)
        self.quadrant_slices_dict = self.create_quadrant_slices(index=index)
        # self.disease_annotations_slice = self.create_disease_slice(index=282)
        seg_slices_list = [s for s in self.quadrant_slices_dict.values()]

        self.quadrant_center_dict = self.calculate_centers()

        # Set active quadrant
        quadrant = QuadrantsInformation.from_id(self.active_quadrant)
        self.quadrant_slices_dict[quadrant].alpha(0.4)

        # ct_vis = (
        #     ct_volume.clone().cmap("gray").alpha([[-1000, 0], [200, 0.2], [1000, 0.7]])
        # )

        ## Panel 2 assets
        vol_bounds = self.ct_volume.bounds()  # xmin,xmax, ymin,ymax, zmin,zmax
        vol_center = self.ct_volume.center()
        camera_params = create_camera_params(vol_center, vol_bounds)
        # print(f"vol_bounds: {vol_bounds}")
        # print(f"vol_center: {vol_center}")
        # print(f"vol origin {ct_volume.origin()}")

        lights_list = create_lights(vol_center, vol_bounds)
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

        return all_slices, all_objects, camera_params

    def slice_intensity_volume(self, ct_volume, index, W=400, L=50):
        """
        Window level for soft tissue
        W=400
        L=50
        """
        ct_slice = ct_volume.yslice(index)

        vmin = L - W / 2
        vmax = L + W / 2

        ct_slice.cmap("gray", vmin=vmin, vmax=vmax)

        return ct_slice

    # def create_disease_slice(self, index: int) -> list[Mesh]:
    #     volume_slice = self.ct_volume.yslice(index) 
    #     disease_slice = self.seg_volume.yslice(index)

    #     lut = colors.build_lut(
    #         [
    #             (0, (0, 0, 0), 0.0),  # background (scalar, color, alpha)
    #             (1, region.color),  # segmentation
    #         ],
    #         vmin=0.7,
    #         vmax=1.2,
    #         below_alpha=0.0,  # type: ignore
    #         above_alpha=1.0,  # type: ignore
    #     )
    #     pass

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

    def calculate_centers(self) -> dict[QuadrantsInformation, np.ndarray]:
        centers_dict = {}

        try:
            centers_dict = load_centers()
        except FileNotFoundError:
            for region, volume in self.quadrant_volumes_dict.items():
                centers_dict[region] = compute_center(volume)
            save_centers(centers_dict)

        return centers_dict

    def on_key_press(self, evt):
        """Handle keyboard events"""
        key = evt.keypress
        if key == "q":
            self.break_interaction()
        elif key.lower() == "t":
            print("Help key pressed")

        elif is_int(key):
            idx = int(key)
            if idx < 7 and idx >= 0:
                # print(f"activating {QuadrantsInformation.from_id(idx).name}")
                current_quadrant = QuadrantsInformation.from_id(self.active_quadrant)
                self.quadrant_slices_dict[current_quadrant].alpha(0.0)

                new_quadrant = QuadrantsInformation.from_id(idx)
                self.quadrant_slices_dict[new_quadrant].alpha(0.4)
                self.active_quadrant = idx

                self.position_camera_in_region(new_quadrant)

                new_text = text_generator(new_quadrant.name, 0, 0)
                self.text_handle.text(new_text)

            self.render()

    def position_camera_in_region(self, new_quadrant: QuadrantsInformation):
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

        self.at(1).camera.SetPosition(new_cam_pos1)  # type: ignore
        self.at(1).camera.SetFocalPoint(center_in_world)  # type: ignore
        self.at(1).camera.SetViewUp([0, 0, 1])  # type: ignore
        self.at(1).renderer.ResetCameraClippingRange()  # type: ignore


def create_camera_params(vol_center, vol_bounds):
    slice_focal_point = [vol_center[0], vol_center[1], vol_center[2]]
    camera_params = {
        "pos": [vol_center[0], vol_bounds[2] - 550, vol_center[2]],
        "focalPoint": slice_focal_point,
        "viewup": [0, 0, 1],
    }
    return camera_params


def parse_mtl(mtl_file: Path):
    """Parse .mtl file and return a dict of {material_name: (r,g,b)} diffuse colors."""
    materials = {}
    current = None
    with open(mtl_file, "r") as f:
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
