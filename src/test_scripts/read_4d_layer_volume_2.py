
import sys
sys.path.append("./src")  # to import from parent dir

from pathlib import Path
import numpy as np
from vedo import Volume, colors, show
import nrrd

from VedoSegmentLoader import VedoSegmentLoader

def create_camera_params(vol_center, vol_bounds):
    slice_focal_point = [vol_center[0], vol_center[1], vol_center[2]]
    camera_params = {
        "pos": [vol_center[0], vol_bounds[2] - 550, vol_center[2]],
        "focalPoint": slice_focal_point,
        "viewup": [0, 0, 1],
    }
    return camera_params

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
    camera_params = create_camera_params(ct.center(), ct.bounds())

    ## Slices and visualization
    slice_index = 282  # choose the slice along z-axis

    # CT Slice
    ct_slice = ct.yslice(slice_index)
    W=400
    L=50
    ct_slice = ct.yslice(slice_index)
    vmin = L - W / 2
    vmax = L + W / 2
    ct_slice.cmap("gray", vmin=vmin, vmax=vmax)

    # Seg slice
    label_id = 2 #todo
    lymph_node_volume = vedo_segment_loader.get_volume_from_id("lymph node")
    seg_slice = lymph_node_volume.yslice(slice_index)
    lut = colors.build_lut(
        [
            (0, (0, 0, 1), 0.0),  # everything else transparent
            (label_id, "#9725e8"),  # label 2 → red semi-transparent
        ],
        vmin=0,
        vmax=label_id,
    )
    seg_slice.cmap(lut)
    seg_slice.alpha(0.3)

    label_id = 1 #todo
    primary_volume = vedo_segment_loader.get_volume_from_id("primary")
    seg_slice2 = primary_volume.yslice(slice_index)
    lut = colors.build_lut(
        [
            (0, (0, 0, 1), 0.0),  # everything else transparent
            (label_id, "#45e825"),  # label 1 → green semi-transparent
        ],
        vmin=0,
        vmax=label_id,
    )
    seg_slice2.cmap(lut)
    seg_slice2.alpha(0.3)

    # show slice
    show(seg_slice, seg_slice2, ct_slice, bg="black", camera=camera_params)

if __name__ == "__main__":
    main()