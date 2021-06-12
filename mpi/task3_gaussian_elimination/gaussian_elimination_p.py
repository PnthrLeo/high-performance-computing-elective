import warnings

from mpi4py import MPI
import numpy as np


def swap_rows(mat: np.matrix, row_1: int, row_2: int) -> np.matrix:
    mat[row_1, :], mat[row_2, :] = mat[row_2, :].copy(), mat[row_1, :].copy()
    return mat


def swap_cols(mat: np.matrix, col_1: int, col_2: int) -> np.matrix:
    if (mat.ndim == 1):
        mat = np.expand_dims(mat, axis=0)

    mat[:, col_1], mat[:, col_2] = mat[:, col_2].copy(), mat[:, col_1].copy()
    return mat


def gaussian_elimination(A: np.matrix, b: np.ndarray) -> np.ndarray:

    def forward_elimination(augmented_mat: np.matrix) -> np.matrix:
        comm = MPI.COMM_WORLD
        proc_quantity = comm.Get_size()

        rows, columns = np.shape(augmented_mat)
        nonlocal order

        for row in range(rows):
            residual_mat = abs(augmented_mat[row:, row:-1])
            max_num_ind = np.unravel_index(np.argmax(residual_mat, axis=None),
                                           residual_mat.shape)
            max_num_ind_row = row + max_num_ind[0]
            max_num_ind_col = row + max_num_ind[1]

            swap_rows(augmented_mat, row, max_num_ind_row)
            swap_cols(augmented_mat, row, max_num_ind_col)
            swap_cols(order, row, max_num_ind_col)

            if augmented_mat[row, row] == 0:
                if augmented_mat[row, columns - 1] != 0:
                    warnings.warn("System is inconsistent.")
                continue

            augmented_mat[row, :] /= augmented_mat[row, row]

            # parallel part of the forward_elimination func
            op = 0
            comm.bcast(op, root=0)
            comm.Bcast(augmented_mat, root=0)
            comm.bcast(row, root=0)

            gap = (rows - row - 1) // proc_quantity
            begin = row + 1
            end = begin + gap
            for diminished_row in range(begin, end):
                factor = augmented_mat[diminished_row, row]
                substraction_row = factor * augmented_mat[row, :]
                augmented_mat[diminished_row, :] -= substraction_row

            res_begin = row + 1 + gap * proc_quantity
            for diminished_row in range(res_begin, rows):
                factor = augmented_mat[diminished_row, row]
                substraction_row = factor * augmented_mat[row, :]
                augmented_mat[diminished_row, :] -= substraction_row

            recv_buf = np.empty((gap*proc_quantity, columns))
            send_buf = augmented_mat[begin:end, :]
            comm.Gather(send_buf, recv_buf, root=0)

            augmented_mat[begin:res_begin, :] = recv_buf

        return augmented_mat

    def back_substitution(augmented_mat: np.matrix) -> np.matrix:
        comm = MPI.COMM_WORLD
        proc_quantity = comm.Get_size()

        rows, columns = np.shape(augmented_mat)
        b_coeffs = columns - 1

        for row in reversed(range(rows)):
            # parallel part of the forward_elimination func
            op = 1
            comm.bcast(op, root=0)
            comm.Bcast(augmented_mat, root=0)
            comm.bcast(row, root=0)

            factor = augmented_mat[row, b_coeffs]
            ignored_rows = rows - row

            gap = (rows - ignored_rows) // proc_quantity
            end = -ignored_rows
            begin = end - gap
            substraction_col = factor * augmented_mat[begin:end, row]
            augmented_mat[begin:end, b_coeffs] -= substraction_col

            recv_buf = np.empty(gap*proc_quantity).flatten()
            send_buf = augmented_mat[begin:end,
                                     b_coeffs].flatten()[::-1].flatten()
            comm.Gather(send_buf, recv_buf, root=0)

            res_end = -ignored_rows - gap*proc_quantity
            if rows + res_end > 0:
                substraction_col = factor * augmented_mat[:res_end, row]
                augmented_mat[:res_end, b_coeffs] -= substraction_col
            augmented_mat[res_end:end, b_coeffs] = recv_buf[::-1]

        return augmented_mat

    # A must to be a square matrix so we need to check first
    equation_quantity, var_quantity = np.shape(A)
    if equation_quantity != var_quantity:
        warnings.warn("A needs to be a square matrix.")

    order = np.arange(var_quantity)

    augmented_mat = np.concatenate((A, b), axis=1)
    augmented_mat = augmented_mat.astype(np.float64)

    comm = MPI.COMM_WORLD
    comm.bcast(augmented_mat.shape, root=0)

    forward_elimination(augmented_mat)
    back_substitution(augmented_mat)

    x = np.zeros(var_quantity)
    for i in range(var_quantity):
        x[order[i]] = augmented_mat[:, var_quantity][i]
    return x


def run_worker():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    proc_quantity = comm.Get_size()

    augmented_mat_shape = (0, 0)
    augmented_mat_shape = comm.bcast(augmented_mat_shape, root=0)
    while True:
        op = -1
        op = comm.bcast(op, root=0)

        # help with forward elimination
        if op == 0:
            augmented_mat = np.empty(augmented_mat_shape)
            comm.Bcast(augmented_mat, root=0)
            row = 0
            row = comm.bcast(row, root=0)
            rows = augmented_mat_shape[0]

            gap = (rows - row - 1) // proc_quantity
            begin = row + 1 + gap * rank
            end = begin + gap
            for diminished_row in range(begin, end):
                factor = augmented_mat[diminished_row, row]
                substraction_row = factor * augmented_mat[row, :]
                augmented_mat[diminished_row, :] -= substraction_row

            recv_buf = None
            send_buf = augmented_mat[begin:end, :]
            comm.Gather(send_buf, recv_buf, root=0)

        # help with back_substitution
        elif op == 1:
            augmented_mat = np.empty(augmented_mat_shape)
            comm.Bcast(augmented_mat, root=0)
            row = 0
            row = comm.bcast(row, root=0)
            rows, columns = augmented_mat_shape
            b_coeffs = columns - 1

            factor = augmented_mat[row, b_coeffs]
            ignored_rows = rows - row

            gap = (rows - ignored_rows) // proc_quantity
            end = -ignored_rows - gap*rank
            begin = end - gap
            substraction_col = factor * augmented_mat[begin:end, row]
            augmented_mat[begin:end, b_coeffs] -= substraction_col

            recv_buf = None

            send_buf = augmented_mat[begin:end,
                                     b_coeffs].flatten()[::-1].flatten()
            comm.Gather(send_buf, recv_buf, root=0)

        # stop worker
        elif op == -1:
            return


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=False)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        n = 500
        a = np.random.rand(n, n)
        b = np.random.rand(n, 1)

        start = MPI.Wtime()
        x = gaussian_elimination(a, b)
        end = MPI.Wtime()

        is_equal = np.allclose(b.flatten(),
                               a.dot(x).flatten())
        print('Is answer true: ', is_equal)
        print('Time elapsed: ', end-start, 's')

        op = -1
        comm.bcast(op, root=0)
    else:
        run_worker()
