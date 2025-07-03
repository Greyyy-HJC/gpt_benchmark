# load python modules
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
import os
import time
from mpi4py import MPI
import math

# load gpt modules
import gpt as g 
from Proton_qTMD_utils import proton_TMD
from tools import *
from io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma
import subprocess

# Gobal parameters
data_dir="/home/jinchen/git/lat-software/gpt_benchmark/examples/pyq_contract/data" # NOTE
lat_tag = "l64c64a076" # NOTE
sm_tag = "1HYP_GSRC_W90_k3_Z5" # NOTE
interpolation = "Z5" # NOTE, new interpolation operator
GEN_SIMD_WIDTH = 64
g.message(f"--lat_tag {lat_tag}")
g.message(f"--sm_tag {sm_tag}")


# --------------------------
# initiate quda
# --------------------------
mpi_geometry = [1, 1, 1, 1]
init(mpi_geometry)



# --------------------------
# Setup parameters
# --------------------------
parameters = {
    
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 1,
    "b_T": 1,

    "qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [-2,-1,0] for y in [-2,-1,0] for z in [0]}], # momentum transfer for TMD, pf = pi + q
    "qext_PDF": [[x,y,z,0] for x in [-2,-1,0] for y in [-2,-1,0] for z in [-2,-1,0]], # momentum transfer for PDF, not used 
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [0] for y in [0] for z in [0]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "width" : 9.0,

    "pol": ["PpUnpol"],
    "t_insert": 4, # time separation for TMD

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2]) + "dt" + str(parameters["t_insert"])
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
Measurement = proton_TMD(parameters)




# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 16
Lt = 16
conf = 0
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"/home/jinchen/git/lat-software/gpt_benchmark/conf/S16T16_cg/gauge/wilson_b6.cg.1e-08.{conf}"), g.double )


L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-8) # CG fix, to get trafo
latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U_prime[0].grid, GEN_SIMD_WIDTH)

w = g.qcd.fermion.wilson_clover(
        U_prime,
        {
            "kappa": 0.126,
            "csw_r": 0,
            "csw_t": 0,
            "xi_0": 1,
            "nu": 1,
            "isAnisotropic": False,
            "boundary_phases": [1.0, 1.0, 1.0, -1.0],
        },
    )
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-10, "maxiter": 1000})
propagator = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))

src = g.mspincolor(grid)
g.create.point(src, [0,0,0,0])
prop_exact_f = g.mspincolor(grid)
prop_exact_f @= propagator * src

phases_2pt = Measurement.make_mom_phases_2pt(U_prime[0].grid, [0,0,0,0])
tag = "test"

corr = Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation)
print(np.shape(corr))

# Save correlation function to txt file
if g.rank() == 0:  # Only write from rank 0 process
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate output filename
    outfile = f"{data_dir}/corr_2pt_{tag}.txt"
    
    # Write correlation function data
    with open(outfile, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(corr)}\n")
        f.write("# Format: [momentum_index][timeslice] value\n\n")
        
        # Write data
        for mom_idx in range(len(corr)):
            for t in range(len(corr[mom_idx])):
                f.write(f"{mom_idx} {t} {corr[mom_idx][t]}\n")
            f.write("\n")  # Blank line between momentum indices

g.message(f"Correlation function saved to {outfile}")






# ###################### setup source positions ######################
# src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
# src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
# src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source 4*4*4*4
# src_production = src_positions[0: 8] # take the number of sources needed for this project NOTE

# ###################### create multigrid inverter ######################
# latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
# dirac = core.getDirac(latt_info, -0.049, 1e-10,  5000, 1.0, 1.0372, 1.0372, [[8, 8, 4, 4]]) # remove the last two arguments for BiCGStab; S mass -0.015, U/D mass -0.049
# gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)
# g.message("DEBUG plaquette U_hyp:", g.qcd.gauge.plaquette(U_hyp))
# g.message("DEBUG plaquette gauge:", gauge.plaquette())
# dirac.loadGauge(gauge)
# g.message("Multigrid inverter ready.")
# g.mem_report(details=False)




# # --------------------------
# # Start measurements
# # --------------------------

# ###################### record the finished source position ######################
# sample_log_file = data_dir + "/sample_log_qtmd/" + str(conf) + '_' + sm_tag + "_" + pf_tag
# if g.rank() == 0:
#     f = open(sample_log_file, "a+")
#     f.close()

# ###################### loop over sources ######################
# for pos in src_production:
    
#     sample_log_tag = get_sample_log_tag(str(conf), pos, sm_tag + "_" + pf_tag)
#     g.message(f"START: {sample_log_tag}")
#     with open(sample_log_file, "a+") as f:
#         if sample_log_tag in f.read():
#             g.message("SKIP: " + sample_log_tag)
#             continue # NOTE comment this out for test otherwise it will skip all the sources that are already done

#     # get forward propagator boosted source
#     cp.cuda.runtime.deviceSynchronize()
#     t0 = time.time()
#     srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
#     b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
#     b.toDevice()
#     cp.cuda.runtime.deviceSynchronize()
#     g.message("TIME GPT-->Pyquda: Generatring boosted src", time.time() - t0)

#     # get forward propagator: smeared-point
#     cp.cuda.runtime.deviceSynchronize()
#     t0 = time.time()
#     propag = core.invertPropagator(dirac, b, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
#     prop_exact_f = g.mspincolor(grid)
#     gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)
#     cp.cuda.runtime.deviceSynchronize()
#     g.message("TIME Pyquda-->GPT: Forward propagator inversion", time.time() - t0)

#     # contraction for 2pt
#     cp.cuda.runtime.deviceSynchronize()
#     t0 = time.time()
#     tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
#     phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
#     Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation) # NOTE, new interpolation operator
#     cp.cuda.runtime.deviceSynchronize()
#     g.message("TIME GPT: Contraction 2pt (includes sink smearing)", time.time() - t0)

#     # get backward propagator through sequential source for U and D
#     cp.cuda.runtime.deviceSynchronize()
#     t0 = time.time()
#     sequential_bw_prop_down = Measurement.create_bw_seq_Pyquda(dirac, prop_exact_f, trafo, 2, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
#     sequential_bw_prop_up = Measurement.create_bw_seq_Pyquda(dirac, prop_exact_f, trafo, 1, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
#     cp.cuda.runtime.deviceSynchronize()
#     g.message("TIME GPT-->Pyquda-->GPT: Backward propagator through sequential source for U and D", time.time() - t0)

#     # prepare phases for qext
#     phases_3pt = Measurement.make_mom_phases_3pt(U[0].grid, pos)

#     # prepare the TMD separate indices for CG
#     W_index_list_CG = Measurement.create_TMD_Wilsonline_index_list_CG(U[0].grid)
        
#     g.message("\ncontract_TMD loop: CG no links")
#     proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
#     proton_TMDs_up = []
#     for iW, WL_indices in enumerate(W_index_list_CG):
#         cp.cuda.runtime.deviceSynchronize()
#         t0 = time.time()
#         g.message(f"TIME GPT: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
#         tmd_forward_prop = Measurement.create_fw_prop_TMD_CG(prop_exact_f, [WL_indices])
#         cp.cuda.runtime.deviceSynchronize()
#         g.message(f"TIME GPT: cshift", time.time() - t0)
#         cp.cuda.runtime.deviceSynchronize()
#         t0 = time.time()
#         proton_TMDs_down += [list(g.slice_trDA(sequential_bw_prop_down, tmd_forward_prop, phases_3pt,3))]
#         proton_TMDs_up += [list(g.slice_trDA(sequential_bw_prop_up, tmd_forward_prop, phases_3pt,3))]
#         cp.cuda.runtime.deviceSynchronize()
#         g.message(f"TIME GPT: contract TMD for U and D", time.time() - t0)
#         del tmd_forward_prop
#     proton_TMDs_down = np.array(proton_TMDs_down)
#     proton_TMDs_up = np.array(proton_TMDs_up)
#     g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)}")

#     for i, pol in enumerate(parameters["pol"]):
#         cp.cuda.runtime.deviceSynchronize()
#         t0 = time.time()
#         qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.D.ex", pos, f"{sm_tag}.{pf_tag}.{pol}")
#         qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.U.ex", pos, f"{sm_tag}.{pf_tag}.{pol}")
#         if g.rank() == 0:
#             save_qTMD_proton_hdf5(proton_TMDs_down[:,i,:,:,:], qtmd_tag_exact_D, gammalist, parameters["qext"], W_index_list_CG, parameters["t_insert"])
#             save_qTMD_proton_hdf5(proton_TMDs_up[:,i,:,:,:], qtmd_tag_exact_U, gammalist, parameters["qext"], W_index_list_CG, parameters["t_insert"])
#         cp.cuda.runtime.deviceSynchronize()
#         g.message(f"TIME: save TMDs for {pol}", time.time() - t0)

#     with open(sample_log_file, "a+") as f:
#         if g.rank() == 0:
#             f.write(sample_log_tag+"\n")
#     g.message("DONE: " + sample_log_tag)
