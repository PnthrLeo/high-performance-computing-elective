#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>


std::vector<std::vector<int>> generate_matrix(int n, int m, int range=1e+9, int seed=42) {
    std::mt19937 gen;
    gen.seed(seed);

    std::vector<std::vector<int>> mat(n, std::vector<int>(m));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat[i][j] = gen()*range;
        }
    }

    return mat;
}


void mult_block_arr(std::vector<std::vector<int>> &mat1, std::vector<std::vector<int>> &mat2, 
                    std::vector<std::vector<int>> &res, int thread_num) {
    int row_step, col_step;
    int left, right, top, bottom;

    if (mat2.size() > mat1.size()){
        row_step = mat2.size() / omp_get_max_threads() * 2;
        col_step = mat1.size() / 2;
        left = row_step*(thread_num / 2);
        right = left + row_step;
        top = col_step*(thread_num % 2);
        bottom = top + col_step;
    }
    else {
        row_step = mat2.size() / 2;
        col_step = mat1.size() / omp_get_max_threads() * 2;
        left = row_step*(thread_num % 2);
        right = left + row_step;
        top = col_step*(thread_num / 2);
        bottom = top + col_step;
    }

    int sum = 0;

    for (int i = top; i < bottom; i++) {
        for (int j = left; j < right; j++) {
            sum = 0;
            for (int k = 0; k < mat2.size(); k++) {
                sum += mat1[i][k] * mat2[k][j];
            }
            res[i][j] = sum;
        }
    }
}


std::vector<std::vector<int>> mult_mat_mat(std::vector<std::vector<int>> &mat1, std::vector<std::vector<int>> &mat2) {
    std::vector<std::vector<int>>res(mat1.size(), std::vector<int>(mat1.size(), 0));

    #pragma omp parallel
    {
        mult_block_arr(mat1, mat2, res, omp_get_thread_num());
    }

    return res;
}


void run_test(int n, int m, int iters) {
    clock_t cpu_time = 0;
    double real_time = 0;

    for (int i = 0 ; i < iters; i++) {
        std::vector<std::vector<int>> mat1 = generate_matrix(n, m, 1e+4, i);
        std::vector<std::vector<int>> mat2 = generate_matrix(m, n, 1e+4, i);

        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<int>> res = mult_mat_mat(mat1, mat2);

        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        cpu_time += (c_end - c_start);
        real_time += std::chrono::duration<double, std::milli>(t_end-t_start).count();
    }

    std::cout << "Result for matrix of size " << n << "x" << m << " with mean by " << iters << " iterations" << std::endl;
    std::cout << "---CPU time used: " << 1000.0 * cpu_time / iters / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "---Wall clock time passed: " << real_time / iters << " ms" << std::endl;
}


int main() {
    std::cout << std::endl;
    std::cout << "--- Number of threads: " << omp_get_max_threads() << " ---"<< std::endl;
    run_test(100, 100, 10);
    run_test(200, 200, 10);
    run_test(400, 400, 10);
    run_test(600, 600, 10);
    run_test(800, 800, 10);
    return 0;
}
