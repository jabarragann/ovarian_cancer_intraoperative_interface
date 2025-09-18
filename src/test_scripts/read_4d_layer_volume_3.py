import sys

sys.path.append("./src")  # to import from parent dir

from pathlib import Path
import numpy as np
from vedo import Volume, colors, show, Plotter, Line, Mesh
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
    camera_params_slices["axial"] = {
        "pos": [
            voxel_pos_in_world[0],
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


def create_cb(plotter, get_slice, create_crosshair, plane_to_name, ct):
    u_index = 0
    target_in_voxel = [(283, 281, 175), (248, 268, 176), (277,242, 129)]
    target_in_world = []

    for voxel in target_in_voxel:
        temp_world = [0.0, 0.0, 0.0]
        ct.dataset.TransformContinuousIndexToPhysicalPoint(
            voxel,
            temp_world,  # type: ignore
        )
        target_in_world.append(temp_world)

    def on_key_press(event):
        nonlocal u_index
        key = event.keypress
        print(f"Key pressed: {key}")
        if key == "q":
            plotter.close()
        elif key == "h":
            # print(plotter.at(2).camera.GetPosition())
            print(f"actual camera pose - {plotter.at(2).camera.GetPosition()}")

        elif key == "u":
            u_index = (u_index + 1) % 3
            camera_params_slices = create_camera_params(target_in_world[u_index])
            update_plotters(
                target_in_voxel[u_index], target_in_world[u_index], camera_params_slices
            )
        
        plotter.render()


    def update_plotters(target_voxel, target_world, camera_params_slices):
        # print(f"target_voxel: {target_voxel}")
        # print(f"target_world: {target_world}")

        # Remove old actors from each viewport
        for i in range(3):
            plotter.at(i).remove(*current_actors[i])

        # Build new actors and add them
        axes = ("x", "y", "z")
        for i, ax in enumerate(axes):
            slices = get_slice(ax, target_voxel["xyz".index(ax)])
            crosshair = create_crosshair(target_world, slices[0], ax)
            current_actors[i] = slices + crosshair
            plotter.at(i).add(*current_actors[i])

            plotter.at(i).camera = camera_params_slices[plane_to_name[ax]]

    return on_key_press, update_plotters

def create_crosshair(target_in_world: list[float], slice_mesh: Mesh, plane:str, size:float=0.02):
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


current_actors = [[], [], []]


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

    on_key_press, update_ploters = create_cb(
        plotter, get_slice, create_crosshair, plane_to_name, ct
    )

    plotter.interactor.RemoveObservers("KeyPressEvent")  # type: ignore
    plotter.add_callback("KeyPress", on_key_press)

    update_ploters(target_in_voxel, target_in_world, camera_params_slices)

    # print(
    #     f"camera pos - {plane_to_name['z']}",
    #     camera_params_slices[plane_to_name["z"]]["pos"],
    # )
    # print(f"target in world - {target_in_world}")
    # print(f"actual camera pose - {plotter.at(2).camera.GetPosition()}")

    plotter.show(interactive=True)


if __name__ == "__main__":
    main()
