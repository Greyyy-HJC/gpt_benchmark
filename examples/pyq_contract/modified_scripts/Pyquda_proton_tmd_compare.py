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
from qTMD.PyQUDA_proton_qTMD_draft import proton_TMD, pyquda_gamma_ls #! import pyquda_gamma_ls for 3pt
from tools import *
from io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma, phase
import subprocess

# Gobal parameters
data_dir="/lustre/orion/nph158/proj-shared/jinchen/run/nucleon_TMD/data" # NOTE
lat_tag = "l64c64a076" # NOTE
sm_tag = "1HYP_GSRC_W90_k3_Z5" # NOTE
interpolation = "5" # NOTE, new interpolation operator
GEN_SIMD_WIDTH = 64
conf = g.default.get_int("--config_num", 0)
g.message(f"--lat_tag {lat_tag}")
g.message(f"--sm_tag {sm_tag}")
g.message(f"--config_num {conf}")


# --------------------------
# initiate quda
# --------------------------
mpi_geometry = [2, 2, 2, 4]
init(mpi_geometry, enable_mps=True)



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
    "pf": [0,0,7,0],
    "p_2pt": [[x,y,z,0] for x in [-2, -1, 0] for y in [-2, -1, 0] for z in [7]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,3],
    "boost_out": [0,0,3],
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
Ls = 64
Lt = 64
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
# U = g.convert( g.load(f"/lustre/orion/nph158/proj-shared/lattices/l64c64a076/gauge_nersc_gfix_a/l6464f21b7130m00119m0322a.{conf}.Coulomb.nersc"), g.double )
U = g.convert( g.load(f"/lustre/orion/nph158/proj-shared/jinchen/debug/nucleon_TMD/fixed_GLU/l6464f21b7130m00119m0322a.{conf}.coulomb.1e-14"), g.double )

g.mem_report(details=False)
L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=5000, prec=1e-12) # CG fix, to get trafo
del U_prime
U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3])) # hyp smearing
latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U[0].grid, GEN_SIMD_WIDTH)
gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)
g.mem_report(details=False)

###################### setup source positions ######################
src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source 4*4*4*4
src_production = src_positions[0: 1] # take the number of sources needed for this project NOTE

###################### create multigrid inverter ######################
latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
dirac = core.getDirac(latt_info, -0.049, 1e-10,  5000, 1.0, 1.0372, 1.0372, [[8, 8, 4, 4]]) # remove the last two arguments for BiCGStab; S mass -0.015, U/D mass -0.049
gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)
g.message("DEBUG plaquette U_hyp:", g.qcd.gauge.plaquette(U_hyp))
g.message("DEBUG plaquette gauge:", gauge.plaquette())
# gauge.projectSU3(1e-15) #todo: modified by Jinchen, for the new version of pyquda
dirac.loadGauge(gauge)
g.message("Multigrid inverter ready.")
g.mem_report(details=False)




# --------------------------
# Start measurements
# --------------------------

###################### record the finished source position ######################
sample_log_file = data_dir + "/sample_log_qtmd/" + str(conf) + '_' + sm_tag + "_" + pf_tag
if g.rank() == 0:
    f = open(sample_log_file, "a+")
    f.close()

###################### loop over sources ######################
for pos in src_production:
    
    sample_log_tag = get_sample_log_tag(str(conf), pos, sm_tag + "_" + pf_tag)
    g.message(f"START: {sample_log_tag}")
    with open(sample_log_file, "a+") as f:
        if sample_log_tag in f.read():
            g.message("SKIP: " + sample_log_tag)
            continue # NOTE comment this out for test otherwise it will skip all the sources that are already done

    # get forward propagator boosted source
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
    b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
    b.toDevice()
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME GPT-->Pyquda: Generatring boosted src", time.time() - t0)

    # get forward propagator: smeared-point
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    propag = core.invertPropagator(dirac, b, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    prop_exact_f = g.mspincolor(grid)
    gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME Pyquda-->GPT: Forward propagator inversion", time.time() - t0)

    #! GPT: contract 2pt TMD
    # cp.cuda.runtime.deviceSynchronize()
    # t0 = time.time()
    # tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    # phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
    # Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation) # NOTE, new interpolation operator
    # cp.cuda.runtime.deviceSynchronize()
    # g.message("TIME GPT: Contraction 2pt (includes sink smearing)", time.time() - t0)
    
    
    #! Pyquda: contract 2pt TMD
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]] #! [-x, -y, -z], has an extra minus sign compared to "p_2pt" by convention
    phases_2pt = phase.MomentumPhase(latt_info).getPhases(p_2pt_xyz) 
    Measurement.contract_2pt_TMD_pyquda(prop_exact_f, phases_2pt, trafo, tag, interpolation) # NOTE, new interpolation operator
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME Pyquda: Contraction 2pt (includes sink smearing)", time.time() - t0)
    

    #! GPT: get backward propagator through sequential source for U and D
    # cp.cuda.runtime.deviceSynchronize()
    # t0 = time.time()
    # sequential_bw_prop_down = Measurement.create_bw_seq_Pyquda(dirac, prop_exact_f, trafo, 2, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
    # sequential_bw_prop_up = Measurement.create_bw_seq_Pyquda(dirac, prop_exact_f, trafo, 1, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
    # cp.cuda.runtime.deviceSynchronize()
    # g.message("TIME GPT-->Pyquda-->GPT: Backward propagator through sequential source for U and D", time.time() - t0)
    
    #! PyQUDA: get backward propagator through sequential source for U and D
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    sequential_bw_prop_down_pyq = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 2, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
    sequential_bw_prop_up_pyq = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 1, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME GPT-->Pyquda: Backward propagator through sequential source for U and D", time.time() - t0)

    #! GPT: prepare phases for qext
    # phases_3pt = Measurement.make_mom_phases_3pt(U[0].grid, pos)
    
    #! PyQUDA: prepare phases for qext
    qext_xyz = [v[:3] for v in parameters["qext"]] #! [x, y, z] to be consistent with "qext"
    phases_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(qext_xyz)

    # prepare the TMD separate indices for CG
    W_index_list_CG = Measurement.create_TMD_Wilsonline_index_list_CG(U[0].grid)
        
    #! GPT: contract TMD
    # g.message("\ncontract_TMD loop: CG no links")
    # proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
    # proton_TMDs_up = []
    # for iW, WL_indices in enumerate(W_index_list_CG):
    #     cp.cuda.runtime.deviceSynchronize()
    #     t0 = time.time()
    #     g.message(f"TIME GPT: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
    #     tmd_forward_prop = Measurement.create_fw_prop_TMD_CG(prop_exact_f, [WL_indices])
    #     cp.cuda.runtime.deviceSynchronize()
    #     g.message(f"TIME GPT: cshift", time.time() - t0)
    #     cp.cuda.runtime.deviceSynchronize()
    #     t0 = time.time()
    #     proton_TMDs_down += [list(g.slice_trDA(sequential_bw_prop_down, tmd_forward_prop, phases_3pt,3))]
    #     proton_TMDs_up += [list(g.slice_trDA(sequential_bw_prop_up, tmd_forward_prop, phases_3pt,3))]
    #     cp.cuda.runtime.deviceSynchronize()
    #     g.message(f"TIME GPT: contract TMD for U and D", time.time() - t0)
    #     del tmd_forward_prop
    # proton_TMDs_down = np.array(proton_TMDs_down)
    # proton_TMDs_up = np.array(proton_TMDs_up)
    # g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)}")
    
    
    #! PyQUDA: contract TMD
    g.message("\ncontract_TMD loop: CG no links")
    proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
    proton_TMDs_up = []
    
    sequential_bw_joint_down_pyq = contract(
                "qwtzyx, pwtzyxjicf, gim -> pqgwtzyxjmcf",
                phases_3pt_pyq, sequential_bw_prop_down_pyq, pyquda_gamma_ls
            )
    sequential_bw_joint_up_pyq = contract(
                    "qwtzyx, pwtzyxjicf, gim -> pqgwtzyxjmcf",
                    phases_3pt_pyq, sequential_bw_prop_up_pyq, pyquda_gamma_ls
                )
    
    for iW, WL_indices in enumerate(W_index_list_CG):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        g.message(f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
        tmd_forward_prop = Measurement.create_fw_prop_TMD_CG_pyquda(prop_exact_f, WL_indices) #! note here [WL_indices] is changed to WL_indices for PyQUDA
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME PyQUDA: cshift", time.time() - t0)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        proton_TMDs_down += [contract("pqgwtzyxjmcf, wtzyxmjfc -> pqgt", sequential_bw_prop_down_pyq, tmd_forward_prop.data).get()]
        proton_TMDs_up += [contract("pqgwtzyxjmcf, wtzyxmjfc -> pqgt", sequential_bw_prop_up_pyq, tmd_forward_prop.data).get()]
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
        del tmd_forward_prop
    proton_TMDs_down = np.array(proton_TMDs_down)
    proton_TMDs_up = np.array(proton_TMDs_up)
    g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)}")
    

    for i, pol in enumerate(parameters["pol"]):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.D.ex", pos, f"{sm_tag}.{pf_tag}.{pol}")
        qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.U.ex", pos, f"{sm_tag}.{pf_tag}.{pol}")
        if g.rank() == 0:
            save_qTMD_proton_hdf5(proton_TMDs_down[:,i,:,:,:], qtmd_tag_exact_D, gammalist, parameters["qext"], W_index_list_CG, parameters["t_insert"])
            save_qTMD_proton_hdf5(proton_TMDs_up[:,i,:,:,:], qtmd_tag_exact_U, gammalist, parameters["qext"], W_index_list_CG, parameters["t_insert"])
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME: save TMDs for {pol}", time.time() - t0)

    with open(sample_log_file, "a+") as f:
        if g.rank() == 0:
            f.write(sample_log_tag+"\n")
    g.message("DONE: " + sample_log_tag)
