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


std::vector<std::vector<int>> mult_mat_mat(std::vector<std::vector<int>> &mat1, std::vector<std::vector<int>> &mat2) {
    std::vector<std::vector<int>>res(mat1.size(), std::vector<int>(mat1.size(), 0));

    #pragma omp parallel for
    for (int i = 0; i < mat1.size(); i++) {
        for (int j = 0; j < mat1.size(); j++) {
            int sum = 0;
            for (int k = 0; k < mat2.size(); k++) {
                sum += mat1[i][k] * mat2[k][j]; 
            }
            res[i][j] = sum;
        }
    }

    return res;
}


int main(){
    int n, m;
    std::cin >> n >> m;
    std::vector<std::vector<int>> mat1 = generate_matrix(n, m, 1e+4);
    std::vector<std::vector<int>> mat2 = generate_matrix(m, n, 1e+4);

    std::clock_t c_start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> res = mult_mat_mat(mat1, mat2);

    std::clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "result: " << std::endl;
    for (int i = 0; i < res.size(); i++) {
        for (int j = 0; j < res.size(); j++) {
            std::cout << res[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "CPU time used: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "Wall clock time passed: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    return 0;
}
