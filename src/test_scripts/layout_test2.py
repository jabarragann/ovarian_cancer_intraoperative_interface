from vedo import Plotter, Sphere, Box

# ratios
col_ratios = [1, 4, 4]
row_ratios = [5, 1]

# normalise
col_fracs = [c / sum(col_ratios) for c in col_ratios]
row_fracs = [r / sum(row_ratios) for r in row_ratios]

vp = Plotter(shape=(2, 3), sharecam=False, size=(1200, 800))


# --- helper to compute viewport (xmin, ymin, xmax, ymax) ---
def viewport(col, row):
    # row 0 is top
    y0 = 1 - sum(row_fracs[: row + 1])
    y1 = y0 + row_fracs[row]
    x0 = sum(col_fracs[:col])
    x1 = x0 + col_fracs[col]
    return [x0, y0, x1, y1]


# set custom viewports for first-row renderers
for c in range(3):
    print(viewport(c, 0))
    vp.renderers[c].SetViewport(viewport(c, 0))

print(f"col_frac {col_fracs}")
print(f"row_frac {row_fracs}")

# # second row: single renderer spanning entire width
bottom = vp.renderers[3]  # flat index row1,col0
# bottom.SetViewport([0.0, 0.0, 0.5, row_fracs[1]])
bottom.SetViewport([0.0, 0.0, 2.0, row_fracs[1]])
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
