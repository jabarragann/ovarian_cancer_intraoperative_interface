
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
    slice_index = 282 
    slice_index = 227 

    # CT Slice
    ct_slice = ct.yslice(slice_index)
    W=400
    L=50
    ct_slice = ct.yslice(slice_index)
    vmin = L - W / 2
    vmax = L + W / 2
    ct_slice.cmap("gray", vmin=vmin, vmax=vmax)

    seg_slice1 = vedo_segment_loader.get_slice("lymph node", slice_index, plane='y', color="#9725e8")
    seg_slice2 = vedo_segment_loader.get_slice("primary", slice_index, plane='y', color="#45e825")
    seg_slice3 = vedo_segment_loader.get_slice("carcinosis", slice_index, plane='y', color="#f0e964")

    # show slice
    show(seg_slice1, seg_slice2, seg_slice3, ct_slice, bg="black", camera=camera_params)
    # show(seg_slice3, ct_slice, bg="black", camera=camera_params)

if __name__ == "__main__":
    main()