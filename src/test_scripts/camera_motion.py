import time

import numpy as np
from vedo import Plotter, Sphere

vp = Plotter(bg="black")

# add some object to visualize
vp.show(Sphere(r=30).c("red5"))

# get current camera position & focal point
pos1 = np.array(vp.camera.GetPosition())
fp1 = np.array(vp.camera.GetFocalPoint())

# target position/focal point
pos2 = pos1 + np.array([200, 100, 50])  # move camera away
fp2 = fp1 + np.array([0, 0, 0])  # keep looking at same point

# interpolate in N steps
N = 50
for t in np.linspace(0, 1, N):
    pos = (1 - t) * pos1 + t * pos2
    fp = (1 - t) * fp1 + t * fp2
    vp.camera.SetPosition(pos)
    vp.camera.SetFocalPoint(fp)
    vp.render()
    time.sleep(0.1)  # ~50 FPS

vp.interactive()
