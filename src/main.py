from vedo import Volume, show, colors, Mesh
from pathlib import Path
from vedo import Plotter


def load_data(data_path):
    patient_id = 6
    complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
    ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"

    # print(complete_path)
    # print(complete_path.exists())
    # print(ct_path)
    # print(ct_path.exists())

    ct = Volume(ct_path)
    seg = Volume(complete_path / "regions" / "pelvic_region_quadrant.seg.nrrd")
    return ct, seg


def create_slice(ct_volume, seg_volume, index):
    ct_slice = ct_volume.yslice(index)
    ct_slice.cmap("gray", vmin=-500, vmax=1000)

    seg_slice = seg_volume.yslice(index)
    lut = colors.build_lut(
        [
            (0, "black"),  # background
            (1, "red"),  # segmentation
        ],
    )

    # Apply LUT to your slice
    seg_slice.cmap(lut)
    seg_slice.alpha(0.4)

    print(type(seg_slice))
    print(type(ct_slice))

    return ct_slice, seg_slice


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

    # mesh.lighting("plastic").ambient(0.4).specular(0.9).specularPower(30)
    mesh.lighting("plastic")

    return mesh


def load_meshes(folder: str, pattern: str = "*.obj"):
    folder_path = Path(folder)
    meshes = []
    for mesh_path in folder_path.glob(pattern):
        m = load_mesh(mesh_path, load_mtl=True)

        # m = Mesh(str(mesh_path))
        # m.c("random").alpha(0.6)  # random color, semi-transparent
        meshes.append(m)
    return meshes


def main():
    data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")
    index = 270
    ct_volume, seg_volume = load_data(data_path)
    ct_slice, seg_slice = create_slice(ct_volume, seg_volume, index=index)

    ct_vis = ct_volume.clone().cmap("gray").alpha([[-1000, 0], [200, 0.2], [1000, 0.7]])

    plt = Plotter(N=2, bg="black", bg2="black", sharecam=False)

    bounds = ct_volume.bounds()
    vol_center = ct_volume.center()
    camera_params = create_camera_params(vol_center, bounds)

    meshes = load_meshes("/home/juan95/research/3dreconstruction/slicer_scripts/output")
    # meshes.append(ct_vis)

    plt.at(0).show(
        ct_slice, seg_slice, title=f"2D Slice (index {index})", camera=camera_params
    )
    # plt.at(1).show(ct_vis, title="3D Volume Rendering")
    plt.at(1).show(meshes, title="3D Mesh", camera=camera_params)

    # Show the Plotter window with both renderers
    plt.interactive().close()

    print("Viewer closed.")


if __name__ == "__main__":
    main()
