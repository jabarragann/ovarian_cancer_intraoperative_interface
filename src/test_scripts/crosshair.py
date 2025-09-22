from pathlib import Path

from vedo import Line, Volume, show

data_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

patient_id = 6
complete_path = data_path / f"Patient{patient_id:02d}/3d_slicer/"
ct_path = complete_path / f"raw_scans_patient_{patient_id:02d}.nrrd"
seg_path = complete_path / "radiologist_annotations.seg.nrrd"

# vedo_segment_loader = VedoSegmentLoader(seg_path)

ct = Volume(ct_path)
zidx = 150
slice_mesh = ct.zslice(zidx).cmap("gray", vmin=-500, vmax=1000)

# Center of slice
c = slice_mesh.center_of_mass()
print(f"center of mass {c}")
xmin, xmax, ymin, ymax, zmin, zmax = slice_mesh.bounds()

# Amount to extend from the center (here ±30 % of full size)
x_half = (xmax - xmin) * 0.1
y_half = (ymax - ymin) * 0.1

# Short cross-hair lines
hline = Line((c[0] - x_half, c[1], c[2]), (c[0] + x_half, c[1], c[2]), c="yellow", lw=2)
vline = Line((c[0], c[1] - y_half, c[2]), (c[0], c[1] + y_half, c[2]), c="yellow", lw=2)

show(slice_mesh, hline, vline, bg="black")
