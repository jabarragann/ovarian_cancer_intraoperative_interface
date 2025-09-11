from functools import wraps
import time
from vedo import Volume,  colors, Mesh, Light
from pathlib import Path
from vedo import Plotter
from QuadrantInformation import QuadrantsInformation


def time_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        print(f"{self.__class__.__name__}.__init__ took {end - start:.6f} seconds")
        return result

    return wrapper


def load_ct_scans_regions(
    data_path: Path,
) -> dict[QuadrantsInformation, Volume]:
    regions_path = data_path / "regions"
    regions_dict: dict[QuadrantsInformation, Volume] = {}
    for region_file in regions_path.glob("*.seg.nrrd"):
        quadrant_info = QuadrantsInformation.from_file_name(region_file)
        regions_dict[quadrant_info] = Volume(region_file)

    return regions_dict


class CT_Viewer(Plotter):
    @time_init
    def __init__(self):
        super().__init__(N=2, bg="black", bg2="black", sharecam=False)
        self.interactor.RemoveObservers("KeyPressEvent")  # type: ignore

        self.all_slices, self.all_objects, camera_params = self.setup_viewer()

        self.add_callback("KeyPress", self.on_key_press)

        self.at(0).show(self.all_slices, title="2D Slice", camera=camera_params)
        self.at(1).show(self.all_objects, title="3D Mesh", camera=camera_params)

    def load_volumes(self, data_path):
        patient_id = 6
        complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
        ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"

        # print(complete_path)
        # print(complete_path.exists())
        # print(ct_path)
        # print(ct_path.exists())

        ct = Volume(ct_path)
        seg = Volume(complete_path / "regions" / "pelvic_region_quadrant.seg.nrrd")
        region_seg_dict = load_ct_scans_regions(complete_path)

        return ct, seg, region_seg_dict

    def setup_viewer(self):
        data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

        ## Panel 1 assets - Load slices
        index = 270
        self.ct_volume, self.seg_volume, self.region_seg_dict = self.load_volumes(
            data_path
        )

        ct_slice = self.slice_intensity_volume(self.ct_volume, index=index)
        seg_slice = self.slice_segmentation_volume(self.seg_volume, index=index)

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

        all_slices = [ct_slice, seg_slice]

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

    def slice_segmentation_volume(self, seg_volume, index) -> list[Mesh]:
        all_slices = []
        for region, volume in self.region_seg_dict.items():
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
            seg_slice.alpha(0.4)

            all_slices.append(seg_slice)

        return all_slices

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

    def on_key_press(self, evt):
        """Handle keyboard events"""
        if evt.keypress == "q":
            self.break_interaction()
        elif evt.keypress.lower() == "t":
            print("Help key pressed")
        elif evt.keypress.lower() == "1":
            print("region 1 selected")
            for slice in self.all_slices:
                slice.alpha(0.0)
            self.render()
        elif evt.keypress.lower() == "2":
            print("region 2 selected")
            for slice in self.all_slices:
                slice.alpha(0.8)

            self.render()


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
