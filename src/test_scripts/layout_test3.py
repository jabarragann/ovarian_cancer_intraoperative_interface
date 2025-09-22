from vedo import Box, Plotter, Sphere

vp = Plotter(shape=(2, 3), sharecam=False, size=(1200, 800))

# ratios
col_ratios = [2, 4, 4]
row_ratios = [8, 2]
# normalise
col_fracs = [c / sum(col_ratios) for c in col_ratios]
row_fracs = [r / sum(row_ratios) for r in row_ratios]

# top row.
top_border = 0.90
big_canvas_height = 0.6
vp.renderers[0].SetViewport([0.025, top_border - 0.3, 0.175, top_border])
vp.renderers[1].SetViewport([0.225, top_border - big_canvas_height, 0.575, top_border])
vp.renderers[2].SetViewport([0.625, top_border - big_canvas_height, 0.975, top_border])
# second row: single renderer spanning entire width
vp.renderers[3].SetViewport([0.0, 0.0, 2.0, row_fracs[1]])
# remaining two canvaces are moved out.
vp.renderers[4].SetViewport([1.0, 0.0, 1.0, 2.0])
vp.renderers[5].SetViewport([1.0, 0.0, 1.0, 2.0])

# --- show some actors ---
vp.at(0).show(Box().c("red"), resetcam=False)
vp.at(1).show(Box().c("green"), resetcam=False)
vp.at(2).show(Box().c("blue"), resetcam=False)
vp.at(3).show(Box(length=4).c("orange"), resetcam=False)
vp.at(4).show(Box().c("cyan"), resetcam=False)
vp.at(5).show(Box().c("yellow"), resetcam=False)


vp.show().interactive().show()
