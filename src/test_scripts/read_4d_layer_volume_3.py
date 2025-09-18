import sys

sys.path.append("./src")  # to import from parent dir

from pathlib import Path
import numpy as np
from vedo import Volume, colors, show, Plotter, Line
import nrrd

from VedoSegmentLoader import VedoSegmentLoader


def create_camera_params(voxel_pos_in_world, dist_to_plane=600):
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


def slice_factory(ct, vedo_segment_loader):
    def get_slices(plane, slice_index):
        if plane == "x":
            ct_slice = ct.xslice(slice_index)
        elif plane == "y":
            ct_slice = ct.yslice(slice_index)
        elif plane == "z":
            ct_slice = ct.zslice(slice_index)

        W = 400
        L = 50
        vmin = L - W / 2
        vmax = L + W / 2
        ct_slice.cmap("gray", vmin=vmin, vmax=vmax)

        seg_slice1 = vedo_segment_loader.get_slice(
            "lymph node", slice_index, plane=plane, color="#9725e8"
        )
        seg_slice2 = vedo_segment_loader.get_slice(
            "primary", slice_index, plane=plane, color="#45e825"
        )
        seg_slice3 = vedo_segment_loader.get_slice(
            "carcinosis", slice_index, plane=plane, color="#f0e964"
        )

        return [ct_slice, seg_slice1, seg_slice2, seg_slice3]

    return get_slices


def create_cb(plotter):
    def on_key_press(event):
        key = event.keypress
        print(f"Key pressed: {key}")
        if key == "q":
            plotter.close()
        elif key == "h":
            # print(plotter.at(2).camera.GetPosition())
            print(f"actual camera pose - {plotter.at(2).camera.GetPosition()}")

    return on_key_press

# def create_cross_hair(target_in_world, slice_mesh):
#     c = target_in_world
#     xmin, xmax, ymin, ymax, zmin, zmax = slice_mesh.bounds()

#     # Amount to extend from the center (here ±30 % of full size)
#     x_half = (xmax - xmin) * 0.1
#     y_half = (ymax - ymin) * 0.1

#     # Short cross-hair lines
#     hline = Line((c[0] - x_half, c[1], c[2]),
#                 (c[0] + x_half, c[1], c[2]),
#                 c="yellow", lw=2)
#     vline = Line((c[0], c[1] - y_half, c[2]),
#                 (c[0], c[1] + y_half, c[2]),
#                 c="yellow", lw=2)

#     return [hline, vline]

def create_crosshair(target, slice_mesh, plane, size=0.02):
    """
    target: (x,y,z) in world coords
    slice_mesh: vedo.Mesh of the slice (to get bounds)
    plane: 'x', 'y', or 'z' – which axis is constant for this slice
    """
    x0, x1, y0, y1, z0, z1 = slice_mesh.bounds()

    cx, cy, cz = target

    if plane == "z":   # axial -> lines in X,Y
        dx = (x1 - x0) * size
        dy = (y1 - y0) * size
        hline = Line((cx - dx, cy, cz), (cx + dx, cy, cz), c="yellow", lw=2)
        vline = Line((cx, cy - dy, cz), (cx, cy + dy, cz), c="yellow", lw=2)

    elif plane == "y": # coronal -> lines in X,Z
        dx = (x1 - x0) * size
        dz = (z1 - z0) * size
        hline = Line((cx - dx, cy, cz), (cx + dx, cy, cz), c="yellow", lw=2)
        vline = Line((cx, cy, cz - dz), (cx, cy, cz + dz), c="yellow", lw=2)

    elif plane == "x": # sagittal -> lines in Y,Z
        dy = (y1 - y0) * size
        dz = (z1 - z0) * size
        hline = Line((cx, cy - dy, cz), (cx, cy + dy, cz), c="yellow", lw=2)
        vline = Line((cx, cy, cz - dz), (cx, cy, cz + dz), c="yellow", lw=2)

    else:
        raise ValueError("plane must be 'x', 'y' or 'z'")

    return [hline, vline]

def main():
    ## Setup
    data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

    patient_id = 6
    complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
    ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"
    seg_path = complete_path / "radiologist_annotations.seg.nrrd"

    vedo_segment_loader = VedoSegmentLoader(seg_path)

    ct = Volume(ct_path)

    ## Slices and visualization
    get_slice = slice_factory(ct, vedo_segment_loader)

    # ("sagittal", "x", 248), ("coronal", "y", 268), ("axial", "z", 176)
    planes_names_dict = {"coronal": "y", "sagittal": "x", "axial": "z"}
    plane_to_name = {v: k for k, v in planes_names_dict.items()}

    ## CHANGE HERE:
    ## Change the plane to get different slices.

    # target_in_voxel = (248, 268, 176)
    target_in_voxel = (283, 281, 175)
    target_in_world = [0, 0, 0]
    ct.dataset.TransformContinuousIndexToPhysicalPoint(
        target_in_voxel,
        target_in_world,  # type: ignore
    )

    # Look anterior-posterior direction
    camera_params_slices = create_camera_params(target_in_world)

    # Set the plotter
    plotter = Plotter(
        title="CT Viewer", N=3, sharecam=False, bg="black", bg2="black", resetcam=False
    )
    on_key_press = create_cb(plotter)
    plotter.interactor.RemoveObservers("KeyPressEvent")  # type: ignore
    plotter.add_callback("KeyPress", on_key_press)

    slices = get_slice("x", target_in_voxel[0])
    crosshair = create_crosshair(target_in_world, slices[0], "x") 
    plotter.at(0).add(slices+crosshair)

    slices = get_slice("y", target_in_voxel[1])
    crosshair = create_crosshair(target_in_world, slices[0], "y") 
    plotter.at(1).add(slices+crosshair)

    slices = get_slice("z", target_in_voxel[2])
    crosshair = create_crosshair(target_in_world, slices[0], "z") 
    plotter.at(2).add(slices+crosshair)

    plotter.at(0).camera = camera_params_slices[plane_to_name["x"]]
    plotter.at(1).camera = camera_params_slices[plane_to_name["y"]]
    plotter.at(2).camera = camera_params_slices[plane_to_name["z"]]

    # print(
    #     f"camera pos - {plane_to_name['z']}",
    #     camera_params_slices[plane_to_name["z"]]["pos"],
    # )
    # print(f"target in world - {target_in_world}")
    # print(f"actual camera pose - {plotter.at(2).camera.GetPosition()}")


    plotter.show(interactive=True)


if __name__ == "__main__":
    main()
