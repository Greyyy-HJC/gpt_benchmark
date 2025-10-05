# %%
import gpt as g
import numpy as np
from pyquda import init
from pyquda_utils import core, io, source, gamma
from opt_einsum import contract

# Configuration
rng = g.random("T")

conf_path = f"/home/jinchen/git/lat-software/gpt_benchmark/conf/S8T32_cg/gauge"
Ls = 8
Lt = 32
conf_n = 0

init([1, 1, 1, 1], resource_path=".cache")
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
kappa = 0.12623
csw_r = 1.0336
csw_t = 1.0336
multigrid = None # [[4, 4, 4, 4], [2, 2, 2, 8]]

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], 1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t, multigrid)

# %%
#! GPT
g.message(f"NOTE: Processing conf {conf_n}...")
U_fixed = g.convert(g.load(f"{conf_path}/wilson_b6.cg.1e-08.{conf_n}"), g.double)

# Quark and solver setup (same for all source positions)
grid = U_fixed[0].grid
L = np.array(grid.fdimensions)

w = g.qcd.fermion.wilson_clover(
    U_fixed,
    {
        "mass": mass,
        "csw_r": csw_r,
        "csw_t": csw_t,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-10, "maxiter": 10000})
propagator = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# Source positions
src = g.mspincolor(grid)
g.create.point(src, [0, 1, 0, 2]) # x y z t
dst = g.mspincolor(grid)
dst @= propagator * src
correlator_gpt = g(g.trace(dst * g.gamma["Z"]))[0, 0, :, 0].flatten() # x y z t

#! PyQDA
gauge = io.readNERSCGauge(f"{conf_path}/wilson_b6.cg.1e-08.{conf_n}")
dirac.loadGauge(gauge)

point_source = source.propagator(latt_info, "point", [0, 1, 0, 2])
point_propag = core.invertPropagator(dirac, point_source)

correlator_pyqda = core.gatherLattice(
    core.lexico(contract("wtzyxijaa,ji -> wtzyx", point_propag.data, gamma.gamma(4)).real.get(), [0,1,2,3,4]), 
    [0, 1, 2, 3]
)[0, :, 0, 0] # t z y x

g.message("Quark Propagator with insertion: Gamma Z")
g.message(f"GPT correlator:")
g.message(np.real(correlator_gpt))
g.message(f"PyQDA correlator:")
g.message(np.real(correlator_pyqda))
    
    
# %%

