# plot_mode0.plt — clean pm3d sombrero

reset
set terminal pngcairo size 800,600 enhanced font "Arial,12"
set output 'mode0_surface.png'

# 1) Interpolate onto a grid
set dgrid3d 300,300        # finer grid for smoother surface

# 2) Axes and view
unset key
set size square
set tics scale 0.5
set view 60,30
set xlabel "x"
set ylabel "y"
set zlabel "u(x,y)"

# 3) Color surface only
unset hidden3d             # do NOT draw mesh
set pm3d                   # enable pm3d
set palette rgbformulae 33,13,10
set style data pm3d        # default draw style

# 4) Draw just the surface (no wireframe, no points)
splot 'mode0.dat' using 1:2:3

# EOF
