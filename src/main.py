from vedo import Volume, show, colors
from pathlib import Path
from vedo import Plotter

def load_data(data_path):
    ct = Volume(data_path / "raw_scans_patient_08.nrrd")
    seg = Volume(data_path / "regions" / "pelvic_region_quadrant.seg.nrrd")
    return ct, seg

def create_slice(ct_volume, seg_volume, index):
    ct_slice = ct_volume.yslice(index)
    ct_slice.cmap("gray", vmin=-500, vmax=1000)

    seg_slice = seg_volume.yslice(index)
    lut = colors.build_lut(
        [
            (0, "black"),  # background
            (1, "red"),    # segmentation
        ],
    )

    # Apply LUT to your slice
    seg_slice.cmap(lut)
    seg_slice.alpha(0.4)  

    return ct_slice, seg_slice


def main():
    data_path = Path(
        "/home/juan95/JuanData/OvarianCancerDataset/CT_scans/Patient08/3d_slicer/"
    )
    index = 270  
    ct_volume, seg_volume = load_data(data_path)
    ct_slice, seg_slice = create_slice(ct_volume, seg_volume, index=index)


    ct_vis = ct_volume.clone().cmap("gray").alpha([[-1000, 0], [200, 0.2], [1000, 0.7]])

    plt = Plotter(N=2, bg="black", bg2="black", sharecam=False)
    plt.at(0).show(ct_slice, seg_slice, title=f"2D Slice (index {index})")
    plt.at(1).show(ct_vis, title="3D Volume Rendering")

    # Show the Plotter window with both renderers
    plt.interactive().close()

    print("Viewer closed.")

if __name__ == "__main__":
    main()
