import gpt as g
import os
import numpy as np
import h5py
from proton_qPDF_class import proton_qPDF

"""
================================================================================
                        Collinear PDF Calculation
================================================================================
This script calculates collinear parton distribution functions (PDFs) for the proton.

The key concept is the z separation: PDFs are calculated using matrix elements
of the form <p'|O(z)|p> where O(z) is a non-local operator with a Wilson line
of length z in the z-direction.

For collinear PDFs, we compute:
- Wilson lines only in the z-direction (no transverse separation)
- Different z separations from 0 to z_max
- Matrix elements for both up and down quark flavors
- Different nucleon polarizations (PpSzp, PpSzm)
- Multiple time separations (tsep values)
- Multiple hadron momenta

The z separation corresponds to the spatial extent of the non-local operator
and is crucial for extracting the momentum dependence of PDFs.

Output:
- HDF5 files saved in ./output/ directory
- Structure: 2pt|3pt/momentum/[tsep/z_separation/flavor/polarization/gamma_structure]/real_imag
================================================================================
"""

# Import gamma structures from the PDF class
from proton_qPDF_class import my_gammas

def save_correlator_hdf5(results_dict, filename):
    """Save correlation data to HDF5 file with hierarchical structure
    
    Args:
        results_dict: Dictionary with structure:
            {
                "2pt": {momentum: corr_data},  # corr_data is a list[N_conf] of list[list[complex]]
                "3pt": {momentum: {tsep: {z_sep: {flavor: {pol: {gamma: {real/imag: array[N_conf, Lt]}}}}}}}
            }
        filename: Base filename for output
    """
    os.makedirs("output", exist_ok=True)
    filepath = f"output/{filename}.h5"
    
    with h5py.File(filepath, 'w') as f:
        # Create 2pt group and save data
        twopoint_group = f.create_group("2pt")
        for momentum_label, momentum_data in results_dict["2pt"].items():
            momentum_group = twopoint_group.create_group(momentum_label)
            # Process 2pt correlation function data
            # Reshape data to (N_conf, Lt)
            N_conf = len(momentum_data)
            Lt = len(momentum_data[0][0][0])  # [conf][mom][gamma][t]
            real_data = np.zeros((N_conf, Lt))
            imag_data = np.zeros((N_conf, Lt))
            
            for conf_idx, conf_data in enumerate(momentum_data):
                for t_idx, value in enumerate(conf_data[0][0]):  # Only one momentum and gamma structure
                    real_data[conf_idx, t_idx] = value.real
                    imag_data[conf_idx, t_idx] = value.imag
                    
            momentum_group.create_dataset("real", data=real_data)
            momentum_group.create_dataset("imag", data=imag_data)
        
        # Create 3pt group and save data
        threepoint_group = f.create_group("3pt")
        for momentum_label, momentum_data in results_dict["3pt"].items():
            momentum_group = threepoint_group.create_group(momentum_label)
            
            for tsep, tsep_data in momentum_data.items():
                tsep_group = momentum_group.create_group(f"tsep_{tsep}")
                
                for z_sep, z_data in tsep_data.items():
                    z_group = tsep_group.create_group(f"z_{z_sep}")
                    
                    for flavor, flavor_data in z_data.items():
                        flavor_group = z_group.create_group(flavor)
                        
                        for pol, pol_data in flavor_data.items():
                            pol_group = flavor_group.create_group(pol)
                            
                            for gamma_name, gamma_data in pol_data.items():
                                gamma_group = pol_group.create_group(gamma_name)
                                gamma_group.create_dataset("real", data=gamma_data["real"])  # Already in shape (N_conf, Lt)
                                gamma_group.create_dataset("imag", data=gamma_data["imag"])  # Already in shape (N_conf, Lt)
    
    g.message(f"Saved correlator data to {filepath}")

def process_correlator_data(corr_data):
    """Process correlator data into gamma-separated structure"""
    result = {}
    
    # corr_data has shape [momentum][gamma][time]
    for gamma_idx, gamma_name in enumerate(my_gammas):
        gamma_data = []
        
        for mom_idx, mom_data in enumerate(corr_data):
            for t_idx, value in enumerate(mom_data[gamma_idx]):
                gamma_data.append(value)
        
        result[f"gamma_{gamma_name}"] = {
            "real": np.array([val.real for val in gamma_data]),
            "imag": np.array([val.imag for val in gamma_data])
        }
    
    return result

# configuration tag
lat_tag = "S16T16_cg"

# Number of configurations
N_conf = 50  # Adjust this value as needed

# parameters for PDF calculation
parameters = {
    "hadron_momenta": [[0, 0, 0, 0]],  # list of hadron momentum states [px, py, pz, pt]
    "boost_in": [0, 0, 0],
    "boost_out": [0, 0, 0],
    "width": 2.0,  # width of Gaussian source
    "pol": ["PpSzp"],  # polarization of the proton
    "tsep_list": [8],  # time separation list for three point function
    "z_max": 2,  # maximum z separation for PDF calculation (in lattice units)
}

PDF_Measurement = proton_qPDF(parameters)
PDF_Measurement.print_calculation_info()

# prepare wilson line indices for PDF
W_index_list_PDF = PDF_Measurement.create_PDF_Wilsonline_index_list()
g.message("W_index_list_PDF:", np.shape(W_index_list_PDF))
g.message(W_index_list_PDF)

# Initialize results dictionary
results = {"2pt": {}, "3pt": {}}

conf_path = "../../conf/S16T16_cg/gauge"
vtrans_path = "../../conf/S16T16_cg/Vtrans"
precision = 1e-08

# Loop over configurations
for conf_n in range(N_conf):
    g.message(f"Processing configuration {conf_n}")
    
    # load gauge configuration
    g.message("Loading gauge configuration")
    Ls = 16
    Lt = 16
    grid = g.grid([Ls, Ls, Ls, Lt], g.double)
    rng = g.random(f"seed text")
    U_cg = g.convert(g.load(f"{conf_path}/wilson_b6.cg.{precision}.{conf_n}"), g.double)
    trafo = g.convert(g.load(f"{vtrans_path}/V_trans.{precision}.{conf_n}"), g.double)    
    g.message("Finished loading gauge config")

    # prepare inverter
    p = {
        "kappa": 0.12623,
        "csw_r": 1.02868,
        "csw_t": 1.02868,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1, 1, 1, -1],
    }
    w = g.qcd.fermion.wilson_clover(U_cg, p)
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-8, "maxiter": 10000})
    prop_exact = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))


    # create the source and the propagator
    pos = [0, 0, 0, 0]  # position of the source
    srcDp = PDF_Measurement.create_src_2pt(pos, trafo, U_cg[0].grid)
    prop_exact_f = g.eval(prop_exact * srcDp)

    # Loop over hadron momenta
    for hadron_momentum in PDF_Measurement.hadron_momenta:
        momentum_label = PDF_Measurement.get_momentum_label(hadron_momentum)
        g.message(f"Processing hadron momentum = {hadron_momentum} (label: {momentum_label})")
        
        # Initialize momentum results if not exists
        if momentum_label not in results["2pt"]:
            results["2pt"][momentum_label] = []
        if momentum_label not in results["3pt"]:
            results["3pt"][momentum_label] = {}
        
        #! 2pt calculation
        g.message("Contraction: Starting 2pt (includes sink smearing) for PDF")
        phases_2pt = PDF_Measurement.make_mom_phases_2pt(U_cg[0].grid, pos)
        corr_2pt = PDF_Measurement.contract_2pt_PDF(prop_exact_f, phases_2pt, trafo)
        results["2pt"][momentum_label].append(corr_2pt)
        g.message("Contraction: Done 2pt (includes sink smearing) for PDF")
        
        #! 3pt calculation
        # Initialize 3pt results for this momentum and configuration
        if conf_n == 0:
            for tsep in PDF_Measurement.tsep_list:
                results["3pt"][momentum_label][tsep] = {}
                
        # Loop over time separations
        for tsep in PDF_Measurement.tsep_list:
            g.message(f"Processing time separation tsep = {tsep}")
            
            # Create sequential propagators
            sequential_bw_prop_down = PDF_Measurement.create_bw_seq(
                prop_exact, prop_exact_f, trafo, 2, tsep, hadron_momentum, pos
            )
            sequential_bw_prop_up = PDF_Measurement.create_bw_seq(
                prop_exact, prop_exact_f, trafo, 1, tsep, hadron_momentum, pos
            )
            
            # Loop over Wilson line indices
            for WL_indices in W_index_list_PDF:
                current_z_sep = WL_indices[1]
                g.message(f"Processing Wilson line with z separation = {current_z_sep}, index: {WL_indices}")
                
                # Initialize z_sep results if not exists
                if current_z_sep not in results["3pt"][momentum_label][tsep]:
                    results["3pt"][momentum_label][tsep][current_z_sep] = {
                        "down": {},
                        "up": {}
                    }
                
                # Create Wilson line and forward prop
                W = PDF_Measurement.create_PDF_Wilsonline(U_cg, WL_indices)
                pdf_forward_prop = PDF_Measurement.create_fw_prop_PDF(
                    prop_exact_f, [W], [WL_indices]
                )

                # Zero momentum phases for now
                phases = [g.identity(g.complex(U_cg[0].grid))]

                # Calculate correlators
                corr_down = PDF_Measurement.contract_PDF(
                    pdf_forward_prop, sequential_bw_prop_down, phases
                )
                corr_up = PDF_Measurement.contract_PDF(
                    pdf_forward_prop, sequential_bw_prop_up, phases
                )

                # Process correlator data for each polarization
                for pol_index in range(len(PDF_Measurement.pol_list)):
                    pol_name = PDF_Measurement.pol_list[pol_index]
                    
                    # Initialize polarization results if not exists
                    if pol_name not in results["3pt"][momentum_label][tsep][current_z_sep]["down"]:
                        results["3pt"][momentum_label][tsep][current_z_sep]["down"][pol_name] = {}
                    if pol_name not in results["3pt"][momentum_label][tsep][current_z_sep]["up"]:
                        results["3pt"][momentum_label][tsep][current_z_sep]["up"][pol_name] = {}

                    # Process down quark results
                    corr_write_down = corr_down[pol_index]
                    down_data = process_correlator_data(corr_write_down)
                    for gamma_name, gamma_data in down_data.items():
                        if gamma_name not in results["3pt"][momentum_label][tsep][current_z_sep]["down"][pol_name]:
                            results["3pt"][momentum_label][tsep][current_z_sep]["down"][pol_name][gamma_name] = {
                                "real": np.zeros((N_conf, Lt)),
                                "imag": np.zeros((N_conf, Lt))
                            }
                        results["3pt"][momentum_label][tsep][current_z_sep]["down"][pol_name][gamma_name]["real"][conf_n] = gamma_data["real"]
                        results["3pt"][momentum_label][tsep][current_z_sep]["down"][pol_name][gamma_name]["imag"][conf_n] = gamma_data["imag"]

                    # Process up quark results
                    corr_write_up = corr_up[pol_index]
                    up_data = process_correlator_data(corr_write_up)
                    for gamma_name, gamma_data in up_data.items():
                        if gamma_name not in results["3pt"][momentum_label][tsep][current_z_sep]["up"][pol_name]:
                            results["3pt"][momentum_label][tsep][current_z_sep]["up"][pol_name][gamma_name] = {
                                "real": np.zeros((N_conf, Lt)),
                                "imag": np.zeros((N_conf, Lt))
                            }
                        results["3pt"][momentum_label][tsep][current_z_sep]["up"][pol_name][gamma_name]["real"][conf_n] = gamma_data["real"]
                        results["3pt"][momentum_label][tsep][current_z_sep]["up"][pol_name][gamma_name]["imag"][conf_n] = gamma_data["imag"]

                g.message(f"Processed correlators for tsep = {tsep}, momentum = {momentum_label}, z separation = {current_z_sep}")

# Save all results to HDF5 file
save_correlator_hdf5(results, f"{lat_tag}_proton_PDF")

g.message("PDF calculation completed!")
g.message(f"Output file saved as ./output/{lat_tag}_proton_PDF.h5")
g.message(f"Total configurations: {N_conf}")
g.message(f"Total z separations calculated: {PDF_Measurement.get_z_separations()}")
g.message(f"Time separations: {PDF_Measurement.tsep_list}")
g.message(f"Hadron momenta: {PDF_Measurement.hadron_momenta}")
g.message(f"Polarizations: {PDF_Measurement.pol_list}")
g.message("HDF5 structure: 2pt|3pt/momentum/[tsep/z_separation/flavor/polarization/gamma_structure]/real_imag")




