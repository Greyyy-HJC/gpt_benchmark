#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import cgpt, gpt, numpy

def gauge_fix(U, maxiter=1000, prec=1e-8, use_fourier=False, orthog_dir=3): #! add use_fourier and orthog_dir
    field = {
            "U_grid": U[0].grid.obj,
            "U": [u.v_obj[0] for u in U],
        }
    r, f = cgpt.Gauge_fix(field, maxiter, prec, use_fourier, orthog_dir)

    # for gr in r:
    #     print("HALLO")
    #     print(gr)
    result=[]
    for gr in r:
        # grid = gpt.grid(
        #     gr[1], eval("gpt.precision." + gr[2]), eval("gpt." + gr[3]), gr[0]
        # )
        result_grid = []
        otype = gpt.ot_matrix_su_n_fundamental_group(3)
        for t_obj, s_ot, s_pr in gr[4]:
            assert s_pr == gr[2]

            # only allow loading su3 gauge fields from cgpt, rest done in python
            # in the long run, replace *any* IO from cgpt with gpt code
            assert s_ot == "ot_mcolor3"
            l = gpt.lattice(U[0].grid, otype, [t_obj])

            # l.metadata = metadata
            result_grid.append(l)
        result.append(result_grid)
    while len(result) == 1:
        result = result[0]
    fix = gpt.lattice(U[0], gpt.ot_matrix_su_n_fundamental_group(3),[f[0]])

    return result, fix

def projectStout(first):
    if(type(first)==gpt.lattice):
        l = gpt.eval(first)
    else:
        print("Type error in projectStout")
    for i in l.otype.v_idx:
        cgpt.ProjectStout(l.v_obj[i])

def projectSU3(first, second):
    if(type(first)==gpt.lattice and type(second)==gpt.lattice):
        l = gpt.eval(first)
        m = gpt.eval(second)
    else:
        print("Type error in projectSU3")
    for i in l.otype.v_idx:
        cgpt.ProjectSU3(l.v_obj[i], m.v_obj[i])

def cshift(first, second, third, fourth=None):
    if isinstance(first, gpt.expr):
        first = gpt.eval(first)
    if isinstance(first, gpt.expr):
        second = gpt.eval(second)
    return first.__class__.foundation.cshift(first, second, third, fourth)


def copy(first, second=None):
    return_list = isinstance(first, list)
    if second is not None:
        t = gpt.util.to_list(first)
        l = gpt.util.to_list(second)
    else:
        l = gpt.util.to_list(first)
        t = [x.new() for x in l]

    l[0].__class__.foundation.copy(t, l)
    if not return_list:
        return t[0]
    return t


def eval_list(a):
    return [gpt.eval(x) if isinstance(x, gpt.expr) else x for x in a]


def call_binary_aa_num(functional, a, b):
    return_list = (isinstance(a, list)) or (isinstance(b, list))
    a = gpt.util.to_list(a)
    b = gpt.util.to_list(b)
    res = functional(eval_list(a), eval_list(b))
    if return_list:
        return res
    return gpt.util.to_num(res[0, 0])


def call_unary_a_num(functional, a):
    return_list = isinstance(a, list)
    if not return_list:
        a = [a]
    a = eval_list(a)
    objects = {}
    indices = {}
    for n, x in enumerate(a):
        fnd = x.foundation
        if fnd not in objects:
            objects[fnd] = []
            indices[fnd] = []
        objects[fnd].append(x)
        indices[fnd].append(n)
    res = [None] * len(a)
    for fnd in objects:
        idx = indices[fnd]
        res_fnd = functional(objects[fnd])
        for i in range(len(idx)):
            res[idx[i]] = res_fnd[i]
    if return_list:
        return res
    return gpt.util.to_num(res[0])


def rank_inner_product(a, b, use_accelerator=True):
    return call_binary_aa_num(
        lambda la, lb: la[0].__class__.foundation.rank_inner_product(la, lb, use_accelerator), a, b
    )


def inner_product(a, b, use_accelerator=True):
    return call_binary_aa_num(
        lambda la, lb: la[0].__class__.foundation.inner_product(la, lb, use_accelerator), a, b
    )


def norm2(l):
    return call_unary_a_num(lambda la: la[0].__class__.foundation.norm2(la), l)


def object_rank_norm2(l):
    return call_unary_a_num(lambda la: la[0].__class__.foundation.object_rank_norm2(la), l)


def inner_product_norm2(a, b):
    if isinstance(a, gpt.tensor) and isinstance(b, gpt.tensor):
        return gpt.adj(a) * b, a.norm2()
    a = gpt.eval(a)
    b = gpt.eval(b)
    assert len(a.otype.v_idx) == len(b.otype.v_idx)
    r = [cgpt.lattice_inner_product_norm2(a.v_obj[i], b.v_obj[i]) for i in a.otype.v_idx]
    return (
        sum([x[0] for x in r]),
        sum([x[1] for x in r]),
    )  # todo, make local version of this too


def axpy(d, a, x, y):
    x = gpt.eval(x)
    y = gpt.eval(y)
    a = complex(a)
    assert len(y.otype.v_idx) == len(x.otype.v_idx)
    assert len(d.otype.v_idx) == len(x.otype.v_idx)
    for i in x.otype.v_idx:
        cgpt.lattice_axpy(d.v_obj[i], a, x.v_obj[i], y.v_obj[i])


def axpy_norm2(d, a, x, y):
    axpy(d, a, x, y)
    return norm2(d)


def fields_to_tensors(src, functor):
    return_list = isinstance(src, list)
    src = gpt.util.to_list(gpt.eval(src))

    # check for consistent otype
    assert all([src[0].otype.__name__ == obj.otype.__name__ for obj in src])

    result = functor(src)

    if return_list:
        return [[gpt.util.value_to_tensor(v, src[0].otype) for v in res] for res in result]
    return [gpt.util.value_to_tensor(v, src[0].otype) for v in result[0]]


def slice_tr(src, dim):
    return_list = isinstance(src, list)
    src = gpt.util.to_list(gpt.eval(src))

    # check for consistent otype
    assert all([src[0].otype == obj.otype for obj in src])

    result = cgpt.slice_trace(src, dim)

    if return_list:
        #return [[gpt.util.value_to_tensor(v, src[0].otype) for v in res] for res in result]
        return [[complex(v) for v in res] for res in result]
    return [complex(v) for v in result[0]]
    #return [gpt.util.value_to_tensor(v, src[0].otype) for v in result[0]]


def slice_trDA(src, rhs, mom, dim):
    return_list = isinstance(src, list)
    src = gpt.util.to_list(gpt.eval(src))
    rhs = gpt.util.to_list(gpt.eval(rhs))

    # check for consistent otype
    assert all([src[0].otype.__name__ == obj.otype.__name__ for obj in src])
    assert all([rhs[0].otype.__name__ == obj.otype.__name__ for obj in rhs])
#    assert(rhs[0].otype == src[0].otype)

    result = cgpt.slice_traceDA(src, rhs, mom, dim)

    return result

def slice_trQPDF(src, rhs, mom, dim):
    return_list = isinstance(src, list)
    src = gpt.util.to_list(gpt.eval(src))
    rhs = gpt.util.to_list(gpt.eval(rhs))

    # check for consistent otype
    assert all([src[0].otype.__name__ == obj.otype.__name__ for obj in src])
    assert all([rhs[0].otype.__name__ == obj.otype.__name__ for obj in rhs])
#    assert(rhs[0].otype == src[0].otype)

    result = cgpt.slice_traceQPDF(src, rhs, mom, dim)

    return result

def slice_proton(prop, mom, dim):
    return_list = isinstance(prop, list)
    prop = gpt.util.to_list(gpt.eval(prop))
    
    #check for consistent otype
    assert all([prop[0].otype == obj.otype for obj in prop])
    result = cgpt.slice_proton_2pt(prop, mom, dim)

    return result 


    # if return_list:
    #     #return [[gpt.util.value_to_tensor(v, src[0].otype) for v in res] for res in result]
    #     return [[complex(v) for v in res] for res in result]
    # return [complex(v) for v in result[0]]
    # #return [gpt.util.value_to_tensor(v, src[0].otype) for v in result[0]]
def slice(src, dim):
    return fields_to_tensors(src, lambda s: s[0].grid.globalsum(cgpt.lattice_rank_slice(s, dim)))


def indexed_sum(fields, index, length):
    index_obj = index.v_obj[0]
    return fields_to_tensors(
        fields, lambda s: s[0].grid.globalsum(cgpt.lattice_rank_indexed_sum(s, index_obj, length))
    )


def identity(src):
    return src.__class__.foundation.identity(src)


def infinitesimal_to_cartesian(src, dsrc):
    if gpt.util.is_num(src):
        return dsrc
    return dsrc.__class__.foundation.infinitesimal_to_cartesian(src, dsrc)


def project(src, method):
    otype = src.otype
    otype.project(src, method)
    src.otype = otype
    return src


def where(first, second, third, fourth=None):
    if fourth is None:
        question = first
        yes = second
        no = third
        answer = None
    else:
        question = second
        yes = third
        no = fourth
        answer = first

    question = gpt.eval(question)
    yes = gpt.eval(yes)
    no = gpt.eval(no)
    if answer is None:
        answer = gpt.lattice(yes)

    assert len(question.v_obj) == 1
    assert len(yes.v_obj) == len(no.v_obj)
    assert len(answer.v_obj) == len(yes.v_obj)

    params = {"operator": "?:"}

    for a, y, n in zip(answer.v_obj, yes.v_obj, no.v_obj):
        cgpt.ternary(a, question.v_obj[0], y, n, params)

    return answer


def scale_per_coordinate(d, s, a, dim):
    s = gpt.eval(s)
    assert len(d.otype.v_idx) == len(s.otype.v_idx)
    for i in d.otype.v_idx:
        cgpt.lattice_scale_per_coordinate(d.v_obj[i], s.v_obj[i], a, dim)
