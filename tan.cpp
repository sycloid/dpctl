#include <cmath> // std::tan
#include <complex>
#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
    // Host calculation
    std::complex<float> a(11.0, 0.0);
    std::complex<float> result_host = std::tan(a);
    std::cout << "Host tan(" << a << ") = " << result_host << std::endl;

    // Create SYCL queue using default device
    sycl::queue q{sycl::default_selector_v};

    // Allocate USM shared memory
    auto *b = sycl::malloc_shared<std::complex<float>>(1, q);
    auto *result_dev = sycl::malloc_shared<std::complex<float>>(1, q);

    b[0] = a;

    // Run kernel
    q.single_task([=]() { result_dev[0] = std::tan(b[0]); }).wait();

    std::cout << "Device tan(" << b[0] << ") = " << result_dev[0] << std::endl;

    // Clean up
    sycl::free(b, q);
    sycl::free(result_dev, q);

    return 0;
}
