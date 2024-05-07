"""
This is a script to measure gauge operators and glueball mass.
"""
# %%
import gpt as g
import gvar as gv
import numpy as np

conf_path = "../conf/S8T32"
conf_n_ls = np.arange(0, 5)
Nd = 4
Lt = 32
Ls = 8

def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr = 0.0
    vol = float(U[0].grid.fsites)
    Nd = len(U)
    ndim = U[0].otype.shape[0]
    for mu in range(Nd):
        for nu in range(mu):
            tr += g.sum(
                g.trace(
                    U[mu] * g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu])
                )
            )
    return 2.0 * tr.real / vol / Nd / (Nd - 1) / ndim

# Main loop
corr_conf_ls = []
plaq_t = []
plaq_0 = []
for conf_n in conf_n_ls:
    U_read = g.load(f"{conf_path}/wilson_b6.{conf_n}")
    vol = Ls ** 3
    ndim = U_read[0].otype.shape[0] # 3, means SU(3)
    
    val = np.zeros((Lt), dtype=complex)
    for mu in range(Nd):
        for nu in range(mu):
            temp = g.slice(
                g.trace(
                    U_read[mu] * g.cshift(U_read[nu], mu, 1) * g.adj(g.cshift(U_read[mu], nu, 1)) * g.adj(U_read[nu])
                ), 3
            )
            val += np.array(temp)

    plaq = 2.0 * val.real / vol / Nd / (Nd - 1) / ndim
    corr = plaq * plaq[0]

    plaq_t.append(plaq)
    plaq_0.append(plaq[0])
    corr_conf_ls.append(corr)

print( np.mean(corr_conf_ls, axis=0) - np.mean(plaq_0, axis=0) * np.mean(plaq_t, axis=0) )
# %%
