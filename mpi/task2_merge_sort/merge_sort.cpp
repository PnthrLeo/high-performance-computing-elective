/**
 *  \author [Ayaan Khan](http://github.com/ayaankhan98)
 *
 *  \details
 *  Merge Sort is an efficient, general purpose, comparison
 *  based sorting algorithm.
 *  Merge Sort is a divide and conquer algorithm
 *
 */
#include <iostream>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>


void merge(int *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = new int[n1], *R = new int[n2];

    for (i = 0; i < n1; i++) L[i] = arr[l + i];
    for (j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 || j < n2) {
        if (j >= n2 || (i < n1 && L[i] <= R[j])) {
            arr[k] = L[i];
            i++;
        } 
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    delete[] L;
    delete[] R;
}


void merge_sort(int *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}


int* generate_array(int size, int range=1e+9, int seed=42) {
    std::mt19937 gen;
    gen.seed(seed);
    
    int* arr = new int[size];

    for (int i = 0; i < size; i++) {
        arr[i] = gen()*range;
    }

    return arr;
}


bool check_sorted_array(int* arr, int size) {
    for (int i = 0; i < size - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;
    
    return true;
}


void run_test(int n, int iters, int test_num = 1) {
    clock_t cpu_time = 0;
    double real_time = 0;

    for (int i = 0 ; i < iters; i++) {
        int* arr = generate_array(n, 1e+4, i*test_num);

        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        merge_sort(arr, 0, n - 1);

        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        if (!check_sorted_array(arr, n)) {
            std::cout << "Sort Error of array of size " << n << std::endl;
        }
        delete[] arr;

        cpu_time += (c_end - c_start);
        real_time += std::chrono::duration<double, std::milli>(t_end-t_start).count();
    }

    std::cout << "Result for array of size " << n << " with mean by " << iters << " iterations" << std::endl;
    std::cout << "---CPU time used: " << 1000.0 * cpu_time / iters / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "---Wall clock time passed: " << real_time / iters << " ms" << std::endl;
}


int main(int argc, char** argv) {
    run_test(10000, 100, 1);
    run_test(20000, 100, 2);
    run_test(40000, 100, 3);
    run_test(60000, 100, 4);
    run_test(80000, 100, 5);
    run_test(100000, 100, 6);
    run_test(1000000, 100, 6);

    return 0;
}
