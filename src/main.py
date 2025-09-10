from vedo import Volume, show, colors
from pathlib import Path

data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans/Patient08/3d_slicer/") 

# Load CT as a Volume
ct = Volume(data_path/"raw_scans_patient_08.nrrd")
seg = Volume(data_path / "regions" / "pelvic_region_quadrant.seg.nrrd")

# Extract a single slice (for example, axial slice index 150)
index = 270 
ct_slice = ct.yslice(index)
ct_slice.cmap("gray", vmin=-500, vmax=1000)

seg_slice = seg.yslice(index)
lut = colors.build_lut([
    (0, "black"),   # background
    (1, "red"),     # segmentation
],
# vmin=0, vmax=1,  
#     below_alpha=0.0,   # anything below vmin (0) is fully transparent
#     above_alpha=0.4    # label 1 semi-transparent
)

# Apply LUT to your slice
seg_slice.cmap(lut)
seg_slice.alpha(0.4) # You can adjust this value for desired transparency

# seg_slice.backface_culling(False)
# seg_slice.pos(0, 0.1, 0)
# show(seg_slice,ct_slice, bg="black")

ct3d = ct.clone().cmap("gray").alpha([[-1000,0], [200,0.2], [1000,0.7]])
# show(seg_slice, ct_slice, ct3d, N=2,at=[0,0,1], bg="black", axes=1)

from vedo import Plotter
plt = Plotter(N=2, bg='black', bg2='black', sharecam=False)

# Add the 2D slices to the first renderer (at index 0)
plt.at(0).show(ct_slice, seg_slice, title=f"2D Slice (index {index})")

# Add the 3D volumes and meshes to the second renderer (at index 1)
plt.at(1).show(ct3d, title="3D Volume Rendering")

# Show the Plotter window with both renderers
plt.interactive().close()

print("Viewer closed.")

# show(ct_slice, seg_slice, at=0, N=2, bg="black")
# show(ct3d, at=1, N=2, bg="black")

# Show side-by-side: left=2D slice, right=3D volume
# show(slice_img, seg_slice, ct3d, N=2, bg="black", axes=1)

