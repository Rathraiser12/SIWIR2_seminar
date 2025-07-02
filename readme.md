mpirun -n 4 python fem.py --delta 0.01 --nev 8 -- \
       -eps_type krylovschur -st_type sinvert -st_ksp_type preonly \
       -st_pc_type lu -eps_target 90 -eps_monitor_conv
