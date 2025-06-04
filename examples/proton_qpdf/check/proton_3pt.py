import gpt as g 
import os
import sys
import numpy as np
import math
from proton_qTMD_draft import proton_TMD


from tools import *
from io_corr import *

root_output = "."
src_shift = np.array([0,0,0,0])
data_dir = "/lustre/orion/nph159/proj-shared/xgao/prod/64I/qTMD_proton/p0"

# tags
sm_tag = "gauss_source"
lat_tag = "TEST"

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

# prepare wilson line indices
W_index_list_PDF = []
for bz in range(0, 2):
    W_index_list_PDF.append([0, bz, 0, 0]) # [bT, bz, eta, T direction], for PDF only need non-zero bz
g.message("W_index_list_PDF:", np.shape(W_index_list_PDF))
g.message(W_index_list_PDF)

# create the source and the propagator
pos = [0,0,0,0] # position of the source
phases = Measurement.make_mom_phases_PDF(U[0].grid, pos)
srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
prop_exact_f = g.eval(prop_exact * srcDp)
sequential_bw_prop_down = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 2, pos)
sequential_bw_prop_up = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 1, pos)
for iW, WL_indices in enumerate(W_index_list_PDF):
    g.message("Processing Wilson line index:", WL_indices)
    W = Measurement.create_PDF_Wilsonline(U, WL_indices)
    tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_exact_f, [W], [WL_indices])

    proton_TMDs_down = g.slice_trDA(sequential_bw_prop_down, tmd_forward_prop, phases,3)
    proton_TMDs_up = g.slice_trDA(sequential_bw_prop_up, tmd_forward_prop, phases,3)

    g.message("Proton TMDs down:", np.shape(proton_TMDs_down)) # shape (2, 1, 16, 32), for 2 polarizations, 1 momentum, 16 gamma insertion, 32 t
    g.message("Proton TMDs up:", np.shape(proton_TMDs_up))
    
    current_z_sep = WL_indices[1]
    save_correlator_txt_ref(proton_TMDs_down, "proton_TMDs_down" + f"_z{current_z_sep}", 0, "PpSzp", "down")
    save_correlator_txt_ref(proton_TMDs_up, "proton_TMDs_up" + f"_z{current_z_sep}", 0, "PpSzp", "up")
    
g.message("Contraction: Starting 2pt (includes sink smearing)")
tag = get_c2pt_file_tag(data_dir, lat_tag, 0, "ex", pos, sm_tag)
phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
corr_2pt = Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag)
save_correlator_txt_ref(corr_2pt, "proton_TMDs_2pt", 0, "PpSzp", "down")
g.message("Contraction: Done 2pt (includes sink smearing)")