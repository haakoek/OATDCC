import numpy as np

def construct_d_t_1_matrix(f, o, v):
    f_diag = np.diag(f)
    d_t_1 = f_diag[o] - f_diag[v].reshape(-1, 1)

    return d_t_1


def construct_d_t_2_matrix(f, o, v):
    f_diag = np.diag(f)
    d_t_2 = (
        f_diag[o]
        + f_diag[o].reshape(-1, 1)
        - f_diag[v].reshape(-1, 1, 1)
        - f_diag[v].reshape(-1, 1, 1, 1)
    )

    return d_t_2