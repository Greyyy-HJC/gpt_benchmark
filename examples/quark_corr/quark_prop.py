"""
This is a script to measure quark propagator on the coulomb gauge configs.
"""
# %%
import gpt as g
import numpy as np
import gvar as gv

# Configuration
rng = g.random("T")
gamma_idx = "I"

conf_path = "../../conf/S16T16_cg"
# conf_path = "../../conf/S16T16/"
conf_n_ls = np.arange(0, 10)
Ls = 16
Lt = 16

src_positions = [
    (0, 0, 0, 0),
    (0, 0, 0, 4)
]

# Main loop
corr_conf_ls = []
for conf_n in conf_n_ls:
    U_fixed = g.convert(g.load(f"{conf_path}/gauge/wilson_b6.cg.1e-08.{conf_n}"), g.double)
    trafo = g.convert(g.load(f"{conf_path}/Vtrans/V_trans.1e-08.{conf_n}"), g.double)
    # U_fixed = g.convert(g.load(f"{conf_path}/wilson_b6.{conf_n}"), g.double)
    
    U_fixed = g.qcd.gauge.smear.hyp(U_fixed, alpha = np.array([0.75, 0.6, 0.3]))
    
    U_fixed, trafo = g.gauge_fix(U_fixed, maxiter=50000, prec=1e-8, use_fourier=False, orthog_dir=3) # Coulomb gauge

    # Quark and solver setup (same for all source positions)
    grid = U_fixed[0].grid
    L = np.array(grid.fdimensions)

    w = g.qcd.fermion.wilson_clover(
        U_fixed,
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

    # momentum
    # p = 2.0 * np.pi * np.array([1, 0, 0, 0]) / L
    # P = g.exp_ixp(p)

    # Source positions
    correlator = []
    for (x, y, z, t) in src_positions:
        src = g.mspincolor(grid)
        g.create.point(src, [x, y, z, t])
        
        #! boost smearing source
        # width = 0.0000001
        # boost_in = [0, 0, 0]
        # trafo = g.convert(trafo, grid.precision)
        # src = g.create.smear.boosted_smearing(trafo, src, w=width, boost=boost_in)
        
        dst = g.mspincolor(grid)
        dst @= propagator * src
        
        temp_ls = []
        for dz in range(Ls):
            wl = g.qcd.gauge.unit(grid)[0]
            for step in range(dz):
                wl = g.eval(wl * g.cshift(U_fixed[2], 2, step))
                
            dst_shift = g.cshift(dst, 2, dz)
            dst_dressed = g.eval(wl * dst_shift)
            
            #! boost smearing sink
            # width = 0.0000001
            # boost_out = [0, 0, 0]
            # dst_dressed = g.create.smear.boosted_smearing(trafo, dst_dressed, w=width, boost=boost_out)
            
            # temp = g(g.trace(dst_dressed * g.gamma[gamma_idx]))[x, y, z, t]
            
            temp = g(g.trace(dst * g.gamma[gamma_idx]))[x, y, (z+dz) % Ls, t]
            temp_ls.append(temp)
            
        correlator.append(np.roll(temp_ls, -z))
        
        
        # for dt in range(Lt):
        #     wl = g.qcd.gauge.unit(grid)[0]
        #     for step in range(dt):
        #         wl = g.eval(wl * g.cshift(U_fixed[3], 3, step))
                
        #     dst_shift = g.cshift(dst, 3, dt)
        #     dst_dressed = g.eval(wl * dst_shift)
            
        #     #! boost smearing sink
        #     # width = 0.0000001
        #     # boost_out = [0, 0, 0]
        #     # dst_dressed = g.create.smear.boosted_smearing(trafo, dst_dressed, w=width, boost=boost_out)
            
        #     # temp = g(g.trace(dst_dressed * g.gamma[gamma_idx]))[x, y, z, t]
            
        #     temp = g(g.trace(dst * g.gamma[gamma_idx]))[x, y, z, (t+dt) % Lt]
        #     temp_ls.append(temp)
            
        # correlator.append(np.roll(temp_ls, -t))
    
    correlator = np.mean(correlator, axis=0)
    corr_conf_ls.append(np.real(correlator))


gv.dump(corr_conf_ls, f"dump/quark_prop_conf_ls_gfix_hyp_gfix.dat")
# %%