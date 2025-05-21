#include <cmath> // std::asinh
#include <complex>
#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
    // Host calculation
    std::complex<double> a(-67108050.0, 1.0);
    std::complex<double> result_host = std::asinh(a);
    std::cout << "Host asinh(" << a << ") = " << result_host << std::endl;

    // Create SYCL queue using default device
    sycl::queue q{sycl::default_selector_v};

    // Allocate USM shared memory
    auto *b = sycl::malloc_shared<std::complex<double>>(1, q);
    auto *result_dev = sycl::malloc_shared<std::complex<double>>(1, q);

    b[0] = a;

    // Run kernel
    q.single_task([=]() { result_dev[0] = std::asinh(b[0]); }).wait();

    std::cout << "Device asinh(" << b[0] << ") = " << result_dev[0]
              << std::endl;

    // Clean up
    sycl::free(b, q);
    sycl::free(result_dev, q);

    return 0;
}
