import numpy as np
from helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
)
from scipy.linalg import expm
from rhs_t import compute_t_2_amplitudes
from rhs_l import compute_l_2_amplitudes
from density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)
from p_space_equations import compute_R_tilde_ai, compute_R_ia
from quantum_systems import construct_pyscf_system_rhf

molecule = "b 0.0 0.0 0.0; h 0.0 0.0 2.4"
basis = "cc-pvdz"

system = construct_pyscf_system_rhf(
    molecule=molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=True,
    anti_symmetrize=True,
)

no = system.n
nv = system.m
o, v = system.o, system.v

t_2 = np.zeros((nv, nv, no, no), dtype=system.h.dtype)
l_2 = np.zeros((no, no, nv, nv), dtype=system.h.dtype)

h = system.h.copy()
u = system.u.copy()
kappa = np.zeros(h.shape, dtype=h.dtype)

e_ref = (
    np.trace(h[o, o])
    + 0.5 * np.trace(np.trace(u[o, o, o, o], axis1=1, axis2=3))
    + system.nuclear_repulsion_energy
)
print(f"Eref: {e_ref.real}")
e_old = e_ref

tol = 1e-5
max_iters = 100
iters = 0
print()
for i in range(1, max_iters):
    C = expm(kappa)
    C_tilde = expm(-kappa)

    h = np.einsum("pa,ab,bq->pq", C_tilde, h, C, optimize=True)
    u = np.einsum(
        "pa,qb,abcd,cr,ds->pqrs", C_tilde, C_tilde, u, C, C, optimize=True
    )
    f = h + np.einsum("piqi->pq", u[:, o, :, o])

    d_t_1 = construct_d_t_1_matrix(f, o, v)
    d_t_2 = construct_d_t_2_matrix(f, o, v)

    rhs_t2 = compute_t_2_amplitudes(f, u, t_2, o, v, np)
    rhs_l2 = compute_l_2_amplitudes(f, u, t_2, l_2, o, v, np)
    t_2 += rhs_t2 / d_t_2
    l_2 += rhs_l2 / d_t_2.transpose(2, 3, 0, 1).copy()

    rho_qp = compute_one_body_density_matrix(t_2, l_2, o, v, np)
    rho_qspr = compute_two_body_density_matrix(t_2, l_2, o, v, np)

    residuals_t = [np.linalg.norm(rhs_t2)]
    ############################################################
    # This part of the code is common to most (if not all)
    # orbital-optimized methods.
    w_ai = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np)
    w_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np)
    residual_w_ai = np.linalg.norm(w_ai)
    residual_w_ia = np.linalg.norm(w_ia)

    kappa.fill(0)
    kappa[v, o] = -w_ai / d_t_1
    kappa[o, v] = -w_ia / d_t_1.T
    ############################################################
    energy = (
        np.einsum("pq,qp->", h, rho_qp, optimize=True)
        + 0.25 * np.einsum("pqrs,rspq->", u, rho_qspr, optimize=True)
        + system.nuclear_repulsion_energy
    )
    delta_E = energy - e_old
    e_old = energy

    print(f"** Iteration: {i}")
    print(f"EOACCD: {energy.real}, dE: {delta_E.real}")
    print(f"||w_ai||: {residual_w_ai}, ||w_ia||: {residual_w_ia}")
    print()
    if residual_w_ai < tol:
        break
