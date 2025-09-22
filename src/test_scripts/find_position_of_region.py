from collections.abc import MutableSequence
from pathlib import Path

import numpy as np
from vedo import Mesh, Volume

root_path = Path("/home/juan95/JuanData/OvarianCancerDataset/CT_scans")

patient_id = 6
complete_path = root_path / f"Patient{patient_id:02d}/3d_slicer/"
seg = Volume(complete_path / "regions" / "pelvic_region_quadrant.seg.nrrd")

seg_np = seg.tonumpy()

indices = np.argwhere(seg_np == 1)
min_idx = indices.min(axis=0)
max_idx = indices.max(axis=0)
center_idx = ((min_idx + max_idx) // 2).astype(int)
print(f"Center idx {center_idx}")

# seg is a vedo.Volume
ijk = center_idx

# VTK has a method for this:
world_coords: MutableSequence[float]
world_coords = [0, 0, 0]
seg.dataset.TransformContinuousIndexToPhysicalPoint(ijk, world_coords)

print("Voxel world coords:", world_coords)
