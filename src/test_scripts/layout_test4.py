from vedo import Plotter, Sphere, Box

vp = Plotter(shape=(2, 4), sharecam=False, size=(1200, 800))

# ratios
col_ratios = [3, 3, 3]
row_ratios = [8, 2]
# normalise
col_fracs = [c / sum(col_ratios) for c in col_ratios]
row_fracs = [r / sum(row_ratios) for r in row_ratios]

col_fracs_sum = [sum(col_fracs[:i]) for i in range(len(col_fracs))]
print(col_fracs_sum)

## Define viewports helper to compute viewport (xmin, ymin, xmax, ymax)
# top row.
top_border = 0.90
big_canvas_height = 0.6
vp.renderers[0].SetViewport([0.0, top_border, 1.0, 1.0])
vp.renderers[1].SetViewport([ col_fracs_sum[0] , 0.5, col_fracs_sum[1] , top_border  ])
vp.renderers[2].SetViewport([ col_fracs_sum[1] , 0.5, col_fracs_sum[2] , top_border])
vp.renderers[3].SetViewport([ col_fracs_sum[2] , 0.5, 1.0, top_border])
# second row: single renderer spanning entire width
vp.renderers[4].SetViewport([0.0, 0.1, 0.5, 0.5])
vp.renderers[5].SetViewport([0.5, 0.1, 1.0, 0.5])
vp.renderers[6].SetViewport([0.0, 0.0, 1.0, 0.1])
# Move out of the way the unused viewport
vp.renderers[7].SetViewport([1.0, 1.0, 2.0, 2.0])

# --- show some actors ---
vp.at(0).show(Box().c("red"), resetcam=False)
vp.at(1).show(Box().c("green"), resetcam=False)
vp.at(2).show(Box().c("blue"), resetcam=False)
vp.at(3).show(Box().c("orange"), resetcam=False)
vp.at(4).show(Box().c("cyan"), resetcam=False)
vp.at(5).show(Box().c("yellow"), resetcam=False)
vp.at(6).show(Box().c("purple"), resetcam=False)
# vp.at(7).clear()


vp.show().interactive().show()
