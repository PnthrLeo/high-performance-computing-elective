#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>


class Array{
public:
    Array() {
        int* arr = nullptr;
        int size;
    }
    
    void generate_array(int in_size, int range=1e+9, int seed=42) {
        std::mt19937 gen;
        gen.seed(seed);
        
        this->size = in_size;
        this->arr = new int[this->size];

        for (int i = 0; i < this->size; i++) {
            this->arr[i] = gen()*range;
        }

        return;
    }

    void do_odd_even_sort() {
        bool is_sorted;
        int start_value = 0;

        while (!is_sorted || start_value != 0) {
            is_sorted = true;

            for (int i = start_value; i < this->size - 1; i += 2)
                if (this->arr[i] > this->arr[i + 1]) {
                    std::swap(this->arr[i], this->arr[i + 1]);
                    is_sorted = false;
                }

            start_value = 1 - start_value;
        }
    }

    int& operator[](int i) const {
        if (i < 0 || i >= this->size) throw std::out_of_range{ "Vector::operator[]" };
        return this->arr[i];
    }

    int get_size() {
        return this->size;
    }

    ~Array() {
        if (this->arr != nullptr)
            delete[] this->arr;
        this->arr = nullptr;
    }

private:
    int* arr;
    int size;
};


bool check_sorted_array(Array &arr) {
    for (int i = 0; i < arr.get_size() - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;
    
    return true;
}


void run_test(int n, int iters) {
    clock_t cpu_time = 0;
    double real_time = 0;

    for (int i = 0 ; i < iters; i++) {
        Array arr;
        arr.generate_array(n, 1e+4, i);

        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        arr.do_odd_even_sort();

        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        if (!check_sorted_array(arr)) {
            std::cout << "Sort Error of array of size " << n << std::endl;
        }
        cpu_time += (c_end - c_start);
        real_time += std::chrono::duration<double, std::milli>(t_end-t_start).count();
    }

    std::cout << "Result for array of size " << n << " with mean by " << iters << " iterations" << std::endl;
    std::cout << "---CPU time used: " << 1000.0 * cpu_time / iters / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "---Wall clock time passed: " << real_time / iters << " ms" << std::endl;
}


int main() {
    std::cout << std::endl;
    std::cout << "--- Number of threads: " << 1 << " ---"<< std::endl;
    run_test(2000, 10);
    run_test(4000, 10);
    run_test(8000, 10);
    run_test(12000, 10);
    run_test(16000, 10);
    return 0;
}
