import warnings

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
            for diminished_row in range(row + 1, rows):
                factor = augmented_mat[diminished_row, row]
                substraction_row = factor * augmented_mat[row, :]
                augmented_mat[diminished_row, :] -= substraction_row

        return augmented_mat

    def back_substitution(augmented_mat: np.matrix) -> np.matrix:
        rows, columns = np.shape(augmented_mat)
        b_coeffs = columns - 1

        for row in reversed(range(rows)):
            factor = augmented_mat[row, b_coeffs]
            ignored_rows = rows - row
            substraction_col = factor * augmented_mat[:-ignored_rows, row]
            augmented_mat[:-ignored_rows, b_coeffs] -= substraction_col

        return augmented_mat

    # A must to be a square matrix so we need to check first
    equation_quantity, var_quantity = np.shape(A)
    if equation_quantity != var_quantity:
        warnings.warn("A needs to be a square matrix.")

    order = np.arange(var_quantity)

    augmented_mat = np.concatenate((A, b), axis=1)
    augmented_mat = augmented_mat.astype(np.float64)

    forward_elimination(augmented_mat)
    back_substitution(augmented_mat)
    x = np.zeros(var_quantity)
    for i in range(var_quantity):
        x[order[i]] = augmented_mat[:, var_quantity][i]
    return x


if __name__ == '__main__':
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 1)

    x = gaussian_elimination(a, b)
    is_equal = np.allclose(b.flatten(),
                           a.dot(x).flatten())
    print('Is answer true: ', is_equal)
