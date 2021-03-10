#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>


std::vector<std::vector<int>> generate_matrix(int n, int m, int range=1e+9, int seed=42) {
    std::mt19937 gen; 
    gen.seed(seed);

    std::vector<std::vector<int>> mat(n, std::vector<int>(m));

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            mat[i][j] = gen()*range;
        }
    }

    return mat;
}


std::vector<int> find_min_in_rows(std::vector<std::vector<int>> &mat) {
    int n = mat.size();
    std::vector<int> mins(n, 0);

    omp_set_num_threads(8);
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        mins[i] = *std::min_element(mat[i].begin(), mat[i].end());
    }

    return mins;
}


int find_max(std::vector<int> &arr) {
    int max = arr[0];

    #pragma omp parallel for reduction(max: max)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > max)
            max = arr[i];
    }
    
    return max;
}


int main(){
    int n, m;
    std::cin >> n >> m;
    std::vector<std::vector<int>> mat = generate_matrix(n, m);

    std::clock_t c_start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<int> arr = find_min_in_rows(mat);
    int sup = find_max(arr);

    std::clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "result: " <<  sup << std::endl;
    std::cout << "CPU time used: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "Wall clock time passed: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    return 0;
}
