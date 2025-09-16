import sys

sys.path.append("./src")  # to import from parent dir

from pathlib import Path
import numpy as np
from vedo import Volume, colors, show, Plotter
import nrrd

from VedoSegmentLoader import VedoSegmentLoader


def create_camera_params(vol_center, vol_bounds):
    # [xmin,xmax, ymin,ymax, zmin,zmax].
    print(vol_center)
    print(vol_bounds)
    slice_focal_point = [vol_center[0], vol_center[1], vol_center[2]]
    camera_params_3d = {
        "pos": [vol_center[0], vol_bounds[2] - 550, vol_center[2]],
        "focalPoint": slice_focal_point,
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
    print("axial pose:", camera_params_slices["axial"]["pos"])
    print("coronal pose:", camera_params_slices["coronal"]["pos"])
    print("sagittal pose:", camera_params_slices["sagittal"]["pos"])

    return camera_params_3d, camera_params_slices


def create_cb(plotter):
    def on_key_press(event):
        key = event.keypress
        print(f"Key pressed: {key}")
        if key == "q":
            plotter.close()
        elif key == "h":
            print(plotter.at(0).camera.GetPosition())

    return on_key_press


def main():
    ## Setup
    data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

    patient_id = 6
    complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
    ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"
    seg_path = complete_path / "radiologist_annotations.seg.nrrd"

    vedo_segment_loader = VedoSegmentLoader(seg_path)

    ct = Volume(ct_path)
    # Look anterior-posterior direction
    camera_params, camera_params_slices = create_camera_params(ct.center(), ct.bounds())

    ## Slices and visualization
    slice_index = 270
    # slice_index = 116
    # slice_index = 282
    # slice_index = 227

    # CT Slice -
    planes_names_dict = {"coronal": "y", "sagittal": "x", "axial": "z"}
    plane_to_name = {v: k for k, v in planes_names_dict.items()}

    ## CHANGE HERE:
    ## Change the plane to get different slices.
    plane = "z"
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

    plotter = Plotter(title="CT Viewer")
    on_key_press = create_cb(plotter)
    plotter.interactor.RemoveObservers("KeyPressEvent")  # type: ignore
    plotter.add_callback("KeyPress", on_key_press)

    plotter.show(
        seg_slice1,
        seg_slice2,
        seg_slice3,
        ct_slice,
        bg="black",
        camera=camera_params_slices[plane_to_name[plane]],
    )
    # plotter.show(
    #     seg_slice1,
    #     seg_slice2,
    #     seg_slice3,
    #     ct_slice,
    #     bg="black",
    #     camera=camera_params_slices["coronal"],
    # )

    plotter.interactive()


if __name__ == "__main__":
    main()
