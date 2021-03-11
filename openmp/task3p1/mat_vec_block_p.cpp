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


std::vector<int> generate_array(int m, int range=1e+9, int seed=42) {
    std::mt19937 gen;
    gen.seed(seed);

    std::vector<int> arr(m);

    for (int i = 0; i < m; i++) {
        arr[i] = gen()*range;
    }

    return arr;
}


void mult_block_arr(std::vector<std::vector<int>> &mat, std::vector<int> &arr, std::vector<int> &res, int thread_num) {
    int row_step, col_step;
    int left, right, top, bottom;

    if (arr.size() > mat.size()){
        row_step = arr.size() / omp_get_max_threads() * 2;
        col_step = mat.size() / 2;
        left = row_step*(thread_num / 2);
        right = left + row_step;
        top = col_step*(thread_num % 2);
        bottom = top + col_step;
    }
    else {
        row_step = arr.size() / 2;
        col_step = mat.size() / omp_get_max_threads() * 2;
        left = row_step*(thread_num % 2);
        right = left + row_step;
        top = col_step*(thread_num / 2);
        bottom = top + col_step;
    }

    int temp = 0;

    for (int i = top; i < bottom; i++) {
        temp = 0;
        for (int j = left; j < right; j++) {
            temp += mat[i][j] * arr[j];
        }
        #pragma omp atomic
        res[i] += temp;
    }
}


std::vector<int> mult_mat_arr(std::vector<std::vector<int>> &mat, std::vector<int> &arr) {
    std::vector<int> res(mat.size(), 0);

    #pragma omp parallel
    {
        mult_block_arr(mat, arr, res, omp_get_thread_num());
    }

    return res;
}


int main(){
    int n, m;
    std::cin >> n >> m;
    std::vector<std::vector<int>> mat = generate_matrix(n, m, 1e+4);
    std::vector<int> arr = generate_array(m, 1e+4);

    std::clock_t c_start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<int> res = mult_mat_arr(mat, arr);

    std::clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "result: " << std::endl;
    for (auto e: res)
        std::cout << e << " ";
    std::cout << std::endl;
    std::cout << "CPU time used: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "Wall clock time passed: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    return 0;
}
