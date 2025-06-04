import gpt as g 
import os
import sys
import numpy as np
import math
from proton_qTMD_draft import proton_TMD, proton_contr


from tools import *
from io_corr import *

data_dir = "output"

def save_correlator_txt_ref(corr_data, filename, z_sep, polarization, flavor):
    """Save correlation data to txt file for comparison"""
    os.makedirs("output", exist_ok=True)
    filepath = f"output/{filename}.txt"
    
    with open(filepath, 'w') as f:
        f.write(f"# Proton PDF correlator data (Reference)\n")
        f.write(f"# Flavor: {flavor}\n")
        f.write(f"# Polarization: {polarization}\n")
        f.write(f"# Z separation: {z_sep}\n")
        f.write(f"# Format: gamma_idx momentum_idx time_slice real_part imag_part\n")
        f.write(f"#\n")
        
        # corr_data has shape [momentum][gamma][time]
        for mom_idx, mom_data in enumerate(corr_data[0]):
            for gamma_idx, gamma_data in enumerate(mom_data):
                for t_idx, value in enumerate(gamma_data):
                    real_part = value.real
                    imag_part = value.imag
                    f.write(f"{gamma_idx:2d} {mom_idx:2d} {t_idx:2d} {real_part:15.8e} {imag_part:15.8e}\n")
    
    g.message(f"Saved reference correlator data to {filepath}")


# tags
sm_tag = "gauss_source"
lat_tag = "TEST"

# parameters for the TMD calculation
parameters = {
    
    # NOTE: eta > 12 will only run bz=0: check qTMD.proton_qTMD_draft.create_TMD_Wilsonline_index_list
    "eta": [12],
    "b_z": 2,
    "b_T": 0,

    "qext": [0], # momentum transfer for TMD
    "qext_PDF": [0], # momentum transfer for PDF
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [0] for y in [0] for z in [0]], # 2pt momentum

    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "width" : 2.0, # width of Gaussian source

    "pol": ["PpSzp"], # polarization of the proton, PpSzp: positive helicity, PpSzm: negative helicity
    "t_insert": 8, # time separation between source and sink

    "save_propagators": False,
}

# load gauge configuration
g.message("Loading gauge configuration")
Ls = 8
Lt = 32
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)
L = U[0].grid.fdimensions
U, trafo = g.gauge_fix(U, maxiter=5000)
g.message("Finished loading gauge config")

# prepare inverter
Measurement = proton_TMD(parameters)
p = {
    "kappa": 0.1255997387525434, # 0.12623 for 300 MeV pion; 0.1256 for 670 MeV pion
    "csw_r": 1.0336,
    "csw_t": 1.0336,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}
w = g.qcd.fermion.wilson_clover(U, p)
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-8, "maxiter": 10000})
prop_exact = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# create the source and the propagator
pos = [0,0,0,0] # position of the source
srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
prop_exact_f = g.eval(prop_exact * srcDp)
    
g.message("Contraction: Starting 2pt (includes sink smearing)")

corr = g.slice( proton_contr(prop_exact_f, prop_exact_f), 3 )
save_correlator_txt_ref([[corr]], "proton_TMDs_2pt_short", 0, "PpSzp", "down") # * add two dimensions of pol and mom

g.message("Contraction: Done 2pt (includes sink smearing)")


#todo: test
WL = Measurement.create_PDF_Wilsonline(U, [0,3,0,0])
print("shape of WL: ", np.shape(WL[:]))
tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_exact_f, [WL], [[0,3,0,0]])
print("shape of tmd_forward_prop: ", np.shape(tmd_forward_prop[:]))