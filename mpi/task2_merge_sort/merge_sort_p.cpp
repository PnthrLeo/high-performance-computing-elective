#include <iostream>
#include <mpi.h>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>


// Merge sorted arrays (arr[l..m], arr[m..r]) in one sorted array 
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


// Recursive merge sort with mpi modifications 
void merge_sort(int *arr, int l, int r, int thread_num, int max_threads, int thread_offset) {
    if (l >= r) {
        return;
    }

    int m = l + (r - l) / 2;
    int next_thread_num = thread_num + thread_offset;

    // if we have more free threads then...
    if (next_thread_num  < max_threads) {
        MPI_Request msg_request;
        MPI_Status msg_status;

        MPI_Isend(arr + m + 1, r - m, MPI_INT, next_thread_num , thread_offset*2, MPI_COMM_WORLD, &msg_request);
        merge_sort(arr, l, m, thread_num, max_threads, thread_offset*2);
        MPI_Recv(arr + m + 1, r - m, MPI_INT, next_thread_num , thread_offset*2, MPI_COMM_WORLD, &msg_status);
        merge(arr, l, m, r);

        // Just prevents a lot of warning in output, not necessary
        MPI_Wait(&msg_request, &msg_status);
    }
    else {
        merge_sort(arr, l, m, thread_num, max_threads, thread_offset*2);
        merge_sort(arr, m + 1, r, thread_num, max_threads, thread_offset*2);
        merge(arr, l, m, r);
    }
}


// run parallel code
void run_mpi(int argc, char** argv) {
    int max_threads, thread_num;
    MPI_Status msg_status;

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &max_threads);
	MPI_Comm_rank(MPI_COMM_WORLD, &thread_num);

    // main thread return to main
    // other threads wait for commands
    if (thread_num == 0) {
        return;
    }
    else {
        while (true) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msg_status);
            
            // op means operation
            // also (in this case) op means offset
            int op = msg_status.MPI_TAG;
            int source = msg_status.MPI_SOURCE;

            // if op equals 0 then thread closes
            if (op == 0) {
                int dummy;
                MPI_Recv(&dummy, 0, MPI_INT, source, op, MPI_COMM_WORLD, &msg_status);
                MPI_Finalize();
                exit(0);
            } 
            else {
                int* arr;
                int arr_size;

                MPI_Get_count(&msg_status, MPI_INT, &arr_size);
                arr = new int[arr_size];
                MPI_Recv(arr, arr_size, MPI_INT, source, op, MPI_COMM_WORLD, &msg_status);

                merge_sort(arr, 0, arr_size - 1, thread_num, max_threads, op);

                MPI_Send(arr, arr_size, MPI_INT, source, op, MPI_COMM_WORLD);
                delete[] arr;
            }
        }
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
    int max_threads, thread_num;
    MPI_Comm_size(MPI_COMM_WORLD, &max_threads);
	MPI_Comm_rank(MPI_COMM_WORLD, &thread_num);

    clock_t cpu_time = 0;
    double real_time = 0;

    for (int i = 0 ; i < iters; i++) {
        int* arr = generate_array(n, 1e+4, i*test_num);

        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        merge_sort(arr, 0, n - 1, thread_num, max_threads, 1);

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
    std::cout << "---CPU time used by main thread: " << 1000.0 * cpu_time / iters / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "---Wall clock time passed: " << real_time / iters << " ms" << std::endl;
}


int main(int argc, char** argv) {
    run_mpi(argc, argv);
    int max_threads;
    MPI_Comm_size(MPI_COMM_WORLD, &max_threads);

    run_test(10000, 100, 1);
    run_test(20000, 100, 2);
    run_test(40000, 100, 3);
    run_test(60000, 100, 4);
    run_test(80000, 100, 5);
    run_test(100000, 100, 6);
    run_test(1000000, 100, 6);

    // close all threads
    for (int i = 1; i < max_threads; i++) {
        MPI_Send(0, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
