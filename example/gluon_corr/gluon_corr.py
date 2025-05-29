"""
This is a script to measure gluon propagator on the coulomb gauge configs.
"""
# %%
import gpt as g
import numpy as np
import gvar as gv

# Configuration
rng = g.random("T")

conf_path = "../../conf/S16T16_cg/gauge"
conf_n_ls = np.arange(50)
Ls = Lt = 16 # spatial and temporal lattice size

zmax = 16 # z shift between two F

# Main loop
corr_conf_ls = []
for conf_n in conf_n_ls:
    U_fixed = g.convert(g.load(f"{conf_path}/wilson_b6.cg.1e-08.{conf_n}"), g.double)

    fs1 = g.qcd.gauge.field_strength(U_fixed, mu=0, nu=3) # first F_\mu\nu 
    fs2 = g.qcd.gauge.field_strength(U_fixed, mu=0, nu=3) # second F_\mu\nu 

    # * if point source to point sink
    g_corr = []
    for dz in range(zmax):
        fs1_shift = g.cshift(fs1, 2, dz) # 2 means z direction, 0 1 2 3 for x y z t
        temp = np.mean( g.eval(g.color_trace(fs1_shift * fs2))[:] )
        g_corr.append(temp)

    corr_conf_ls.append(g_corr)

gv.dump(corr_conf_ls, f"dump/gluon_corr_conf_ls.dat")

# %%
