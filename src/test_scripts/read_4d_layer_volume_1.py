from pathlib import Path

import nrrd
import numpy as np
from vedo import Volume, colors, show


def create_camera_params(vol_center, vol_bounds):
    slice_focal_point = [vol_center[0], vol_center[1], vol_center[2]]
    camera_params = {
        "pos": [vol_center[0], vol_bounds[2] - 550, vol_center[2]],
        "focalPoint": slice_focal_point,
        "viewup": [0, 0, 1],
    }
    return camera_params


data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

patient_id = 6
complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"
seg_path = complete_path / "radiologist_annotations.seg.nrrd"


ct = Volume(ct_path)
ct_vol_np = ct.tonumpy()

camera_params = create_camera_params(ct.center(), ct.bounds())

## print info
# seg_bad_way = Volume(seg_path)
# seg_vol_np = seg_bad_way.tonumpy()
# print("shape of seg_vol_np:", seg_vol_np.shape)
# print(type(seg_vol_np))
# print(f"shape of ct_vol_np:", ct_vol_np.shape)
# print(type(ct_vol_np))


data, hdr = nrrd.read(str(complete_path / "radiologist_annotations.seg.nrrd"))
# voxel spacing (x,y,z) in mm
spacing = np.array(
    [
        float(hdr["space directions"][1][0]),  # X
        float(hdr["space directions"][2][1]),  # Y
        float(hdr["space directions"][3][2]),  # Z
    ]
)

# world origin (x,y,z)
print(hdr["space origin"])
print(type(hdr["space origin"]))
origin = hdr["space origin"]
print("spacing:", spacing)
print("origin :", origin)

layer0 = data[0]  # shape (512, 512, 293)
layer1 = data[1]  # shape (512, 512, 293)
seg_layer0_vol = Volume(layer0, spacing=spacing, origin=origin)
seg_layer1_vol = Volume(layer1, spacing=spacing, origin=origin)

print(f"shape of seg_layer1_vol: {seg_layer1_vol.tonumpy().shape}")
print(f"shape of ct_vol_np:  {ct_vol_np.shape}")


## Slices and visualization

# parameters
segment_value = 2
layer_index = 0
slice_index = 282  # choose the slice along z-axis


# CT slice
ct_slice = ct.yslice(slice_index)

W = 400
L = 50
ct_slice = ct.yslice(slice_index)
vmin = L - W / 2
vmax = L + W / 2
ct_slice.cmap("gray", vmin=vmin, vmax=vmax)


# seg slice
seg_slice = seg_layer0_vol.yslice(slice_index)
lut = colors.build_lut(
    [
        (0, (0, 0, 1), 0.0),  # everything else transparent
        # (1, (0, 1, 0)),  # label 2 → red semi-transparent
        (2, "#9725e8"),  # label 2 → red semi-transparent
    ],
    vmin=0,
    vmax=segment_value,
)
seg_slice.cmap(lut)
seg_slice.alpha(0.3)


# show slice
show(seg_slice, ct_slice, bg="black", camera=camera_params)
# show(ct_slice, bg="black")
