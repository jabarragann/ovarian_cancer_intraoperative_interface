from vedo import Volume, show, colors
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


def main():
    data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")
    index = 270
    ct_volume, seg_volume = load_data(data_path)
    ct_slice, seg_slice = create_slice(ct_volume, seg_volume, index=index)

    ct_vis = ct_volume.clone().cmap("gray").alpha([[-1000, 0], [200, 0.2], [1000, 0.7]])

    plt = Plotter(N=2, bg="black", bg2="black", sharecam=False)

    print(ct_volume.center())
    bounds = ct_volume.bounds()
    vol_center = ct_volume.center()
    camera_params = create_camera_params(vol_center, bounds)
    plt.at(0).show(
        ct_slice, seg_slice, title=f"2D Slice (index {index})", camera=camera_params
    )
    plt.at(1).show(ct_vis, title="3D Volume Rendering")

    # Show the Plotter window with both renderers
    plt.interactive().close()

    print("Viewer closed.")


if __name__ == "__main__":
    main()
