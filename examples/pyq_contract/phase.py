# load python modules
import numpy as np

# load gpt modules
import gpt as g 

# load pyquda modules
from pyquda import init
from pyquda_utils import gpt, phase


GEN_SIMD_WIDTH = 64

def make_mom_phases_3pt(plist, grid, origin=None):    
    one = g.identity(g.complex(grid))
    pp = [2 * np.pi * np.array(p) / grid.fdimensions for p in plist] # plist is the q for TMD

    P = g.exp_ixp(pp, origin)
    mom = [g.eval(pp*one) for pp in P]
    return mom

mpi_geometry = [1, 1, 1, 1]
init(mpi_geometry, resource_path=".cache")

origin = [1,2,3,4]
plist = [ [0,0,0,0] ]

###################### load gauge ######################
Ls = 4
Lt = 4
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
rng = g.random("seed")
U_read = g.qcd.gauge.random(grid, rng)
latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U_read[0].grid, GEN_SIMD_WIDTH)


phases_3pt = make_mom_phases_3pt(plist, grid, origin)
phases_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(plist, origin)

print(">>> shape of phases_3pt")
print(np.shape(phases_3pt[0][:]))
print(">>> type of phases_3pt")
print(type(phases_3pt[0][:]))

print(">>> shape of phases_3pt_pyq")
print(np.shape(phases_3pt_pyq[0].get()))
print(">>> type of phases_3pt_pyq")
print(type(phases_3pt_pyq[0].get()))

print(phases_3pt[0][:] - phases_3pt_pyq[0].get().reshape(-1, 1))




# %%
#! gpt
######################### gpt #########################

import numpy

def relative_coordinates(x, o, l):
    l = numpy.array(l, dtype=numpy.int32)
    lhalf = l // 2
    o = numpy.array(o, dtype=numpy.int32)
    r = numpy.mod(x + (l - o + lhalf), l) - lhalf
    return r


def apply_exp_ixp(dst, src, p, origin, cache):
    cache_key = f"{src.grid}_{src.checkerboard().__name__}_{origin}_{p}"
    if cache_key not in cache:
        x = gpt.coordinates(src)
        phase = gpt.complex(src.grid)
        phase.checkerboard(src.checkerboard())
        x_relative = x
        if origin is not None:
            x_relative = relative_coordinates(x, origin, src.grid.fdimensions)
        phase[x] = cgpt.coordinates_momentum_phase(x_relative, p, src.grid.precision)
        cache[cache_key] = phase

    dst @= cache[cache_key] * src


def exp_ixp(p, origin=None):
    if isinstance(p, list):
        return [exp_ixp(x, origin) for x in p]
    elif isinstance(p, numpy.ndarray):
        p = p.tolist()

    cache = {}

    def mat(dst, src):
        return apply_exp_ixp(dst, src, p, origin, cache)

    def inv_mat(dst, src):
        return apply_exp_ixp(dst, src, [-x for x in p], origin, cache)

    # do not specify grid or otype, i.e., accept all
    return gpt.matrix_operator(mat=mat, adj_mat=inv_mat, inv_mat=inv_mat, adj_inv_mat=mat)

def make_mom_phases_3pt(plist, grid, origin=None):    
    one = g.identity(g.complex(grid))
    pp = [2 * np.pi * np.array(p) / grid.fdimensions for p in plist] # plist is the q for TMD

    P = g.exp_ixp(pp, origin)
    mom = [g.eval(pp*one) for pp in P]
    return mom


#! PyQUDA
######################### pyquda #########################
from typing import Sequence
from math import pi

def getPhase(self, mom_mode: Sequence[int], x0: Sequence[int] = [0, 0, 0, 0]):
    x = self.x
    global_size = self.latt_info.global_size

    if len(mom_mode) == 3:
        ip = [2j * pi * mom_mode[i] / global_size[i] for i in range(3)]
        ipx = ip[0] * x[0] + ip[1] * x[1] + ip[2] * x[2]
        ipx0 = ip[0] * x0[0] + ip[1] * x0[1] + ip[2] * x0[2]
    elif len(mom_mode) == 4:
        ip = [2j * pi * mom_mode[i] / global_size[i] for i in range(4)]
        ipx = ip[0] * x[0] + ip[1] * x[1] + ip[2] * x[2] + ip[3] * x[3]
        ipx0 = ip[0] * x0[0] + ip[1] * x0[1] + ip[2] * x0[2] + ip[3] * x0[3]
    else:
        getLogger().critical(f"mom should be a sequence of int with length 3 or 4, but get {mom_mode}", ValueError)

    backend = getCUDABackend()
    return arrayExp(ipx - ipx0, backend)

def getPhases(self, mom_mode_list: Sequence[Sequence[int]], x0: Sequence[int] = [0, 0, 0, 0]):
    Lx, Ly, Lz, Lt = self.latt_info.size

    backend = getCUDABackend()
    phases = arrayZeros((len(mom_mode_list), 2, Lt, Lz, Ly, Lx // 2), "<c16", backend)
    for idx, mom in enumerate(mom_mode_list):
        phases[idx] = self.getPhase(mom, x0)

    return phases