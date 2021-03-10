#include <iostream>
#include <functional>
#include <iomanip>
#include <ctime>
#include <chrono>


double simpsonIntegral(double a, double b, int n, const std::function<double (double)> &f) {
    const double width = (b-a)/n;

    double simpson_integral = 0;
    for (int step = 0; step < n; step++) {
        const double x1 = a + step*width;
        const double x2 = a + (step+1)*width;

        simpson_integral += (x2-x1)/6.0*(f(x1) + 4.0*f(0.5*(x1+x2)) + f(x2));
    }

    return simpson_integral;
}


double f(double x){
    return 4.0/(1 + x*x);
}


int main(){
    clock_t c_start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();

    std::function<double (double)> func = f;
    double res = simpsonIntegral(0, 1, 1000000, func);

    clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "result: " << res << std::endl;
    std::cout << "CPU time used: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "Wall clock time passed: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    return 0;
}
