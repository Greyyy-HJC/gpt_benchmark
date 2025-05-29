'''
This is a script to run the CGPT code to do the gauge fixing.
'''

# %%
import gpt as g
from coulomb_gauge import coulomb

read_path = f"../../conf/S16T16"
precision = 1e-8
write_path = f"../../conf/S16T16_cg"

import sys

if len(sys.argv) != 2:
    print("Usage: python gfix_cgpt.py <n_conf>")
    sys.exit(1)

n_conf = int(sys.argv[1])

#! gauge fixing
U_read = g.convert( g.load(f"{read_path}/wilson_b6.{n_conf}"), g.double )

rng = g.random("E")
V0 = g.identity(U_read[1])

rng.element(V0)
U_read = g.qcd.gauge.transformed(U_read, V0) # random gauge transformation
c = coulomb(U_read)

U_fixed, V_trans = g.gauge_fix(U_read, maxiter=30000, prec=precision, use_fourier=False, orthog_dir=3) # Coulomb gauge

g.message(">>> Final Functional Value: ")
g.message(c([V_trans]))
g.message(">>> Final Gradient Value: ")
g.message(g.norm2(c.gradient(V_trans, V_trans)))

g.save(f"{write_path}/gauge/wilson_b6.cg.{precision}.{n_conf}", U_fixed, g.format.nersc())

g.save(f"{write_path}/Vtrans/V_trans.{precision}.{n_conf}", V_trans)

# %%