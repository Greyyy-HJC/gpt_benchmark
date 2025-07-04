import numpy as np
import cupy as cp
from opt_einsum import contract
import gpt

def pyquda_slice_trDA(src_list, rhs_list, mom_list, dim):
    """
    Pure‐Python replacement for g.slice_trDA(src_list, rhs_list, mom_list, dim)
    using opt_einsum.contract.

    Parameters
    ----------
    src_list : list of Lattice<vSpinColourMatrix>  (g‐lattice objects)
    rhs_list : same shape as src_list
    mom_list : list of LatticeComplex  (phase factors)
    dim      : int  which lattice dimension to slice on (e.g. 3 for time)

    Returns
    -------
    result : nested python list with shape
             [Nbasis][Nmom][Ngamma][Nslice]
    exactly matching g.slice_trDA(...)
    """

    # 1) eval g‐lattice objects → CuPy arrays
    src_arrs = [gpt.to_array(x) for x in gpt.eval(src_list)]
    rhs_arrs = [gpt.to_array(x) for x in gpt.eval(rhs_list)]
    mom_arrs = [gpt.to_array(x) for x in gpt.eval(mom_list)]

    # assume grid is 4D, shapes like (Lx,Ly,Lz,Lt, spin, colour, spin, colour) for src/rhs
    grid_shape = mom_arrs[0].shape           # e.g. (Lx,Ly,Lz,Lt)
    D = len(grid_shape)
    Lt = grid_shape[dim]
    Vs = int(np.prod(grid_shape) // Lt)      # # of sites per time‐slice

    # extract spin/colour dims from a src array
    sc_shape = src_arrs[0].shape[D:]         # e.g. (4,3,4,3)
    spin1, nc1, spin2, nc2 = sc_shape
    nSC = spin1 * nc1 * spin2 * nc2

    # 2) 把 time 轴搬到最前面，并把空间三维铺平成一个维度
    def reorg(arr, has_sc=True):
        # arr: CuPy array shape (*grid_shape, *sc_shape)  or (*grid_shape,) if no sc
        # 2a) 置换，把 dim 轴变成最前面
        axes = (dim,) + tuple(i for i in range(D) if i != dim) + tuple(range(D, arr.ndim))
        arr = arr.transpose(axes)
        # 2b) reshape
        if has_sc:
            arr = arr.reshape(Lt, Vs, nSC)
        else:
            arr = arr.reshape(Lt, Vs)
        return arr

    src_ts = [reorg(a, True) for a in src_arrs]     # → (Lt,Vs,nSC)
    rhs_ts = [reorg(a, True) for a in rhs_arrs]     # → (Lt,Vs,nSC)
    mom_ts = [reorg(a, False) for a in mom_arrs]    # → (Lt,Vs)

    # 3) 准备 Gamma 列表
    from pyquda_utils import gamma as gamma_pyq
    # 这里假设你只要用 16 个 Gmu16 矩阵中的一部分；按需替换索引
    Gmu16 = [gamma_pyq.gamma(mu) for mu in range(16)]
    # 颜色单位阵，用于扩展到 spin×colour 大矩阵
    I_col = np.eye(nc1, dtype=Gmu16[0].dtype)

    # 4) 真正“contract + trace”主循环
    result = []
    Nbasis = len(src_ts)
    Nmom   = len(mom_ts)
    Ngamma = len(Gmu16)
    for ib in range(Nbasis):
        out_basis = []
        A = rhs_ts[ib]    # (Lt,Vs,nSC)
        B = src_ts[ib]    # (Lt,Vs,nSC)
        for im in range(Nmom):
            phi = mom_ts[im]  # (Lt,Vs)
            # 4a) 首先做空间和 simd 的累加：
            #     raw[t, α, β] = sum_{n in slice}  A[t,n,α] * B[t,n,β] * φ[t,n]
            raw = contract('tna, tnb, tn -> tab',
                           A, B, phi, optimize='optimal')   # → (Lt, nSC, nSC)

            # 4b) 对于每个 G_mu 做插入并 trace
            out_mom = []
            for G in Gmu16:
                # 把 G 扩展到 spin×colour 大矩阵 G_big
                G_big = np.kron(G, I_col)          # → (spin1*nc1, spin2*nc2) == (nSC, nSC)

                # 对每个 t 做 trace
                corr = []
                for t in range(Lt):
                    M = raw[t]                     # CuPy or NumPy (nSC,nSC)
                    # 把 M 拉回 host，如果是 CuPy 的话
                    if isinstance(M, cp.ndarray):
                        M = M.get()
                    corr.append(np.trace(G_big.dot(M)))
                out_mom.append(corr)              # → list length Lt

            out_basis.append(out_mom)            # → [Nmom][Ngamma][Lt]

        result.append(out_basis)                 # → [Nbasis][Nmom][Ngamma][Lt]

    return result