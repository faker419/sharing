#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <limits>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

// Function to find the nearest nail index
int find_nearest_nail(double angle, const std::vector<double>& theta_values) {
    double angle_wrapped = fmod(angle, 2 * M_PI);
    if (angle_wrapped < 0) angle_wrapped += 2 * M_PI;

    double min_difference = std::numeric_limits<double>::max();
    int nearest_index = -1;

    // Parallelize the loop to find the nearest nail
    #pragma omp parallel for reduction(min:min_difference)
    for (size_t i = 0; i < theta_values.size(); ++i) {
        double difference = fabs(theta_values[i] - angle_wrapped);
        if (difference < min_difference) {
            nearest_index = i;
            min_difference = difference;
        }
    }
    return nearest_index + 1;
}



py::array_t<double> p_regionfun(
    double alpha0, double s0,
    py::array_t<double> ALPHA, py::array_t<double> S,
    int R, double t) {

    auto alpha_buf = ALPHA.request();
    auto s_buf = S.request();

    const double* alpha_ptr = static_cast<double*>(alpha_buf.ptr);
    const double* s_ptr = static_cast<double*>(s_buf.ptr);

    py::array_t<double> result(ALPHA.size());
    auto result_ptr = static_cast<double*>(result.request().ptr);

    size_t total_size = alpha_buf.size;

    // OpenMP Parallel for Loop with Efficient Memory Access
    #pragma omp parallel for
    for (size_t i = 0; i + 3 < total_size; i += 4) {
        // Fetch values only once for four elements
        double alpha1 = alpha_ptr[i];
        double alpha2 = alpha_ptr[i + 1];
        double alpha3 = alpha_ptr[i + 2];
        double alpha4 = alpha_ptr[i + 3];

        double s1 = s_ptr[i];
        double s2 = s_ptr[i + 1];
        double s3 = s_ptr[i + 2];
        double s4 = s_ptr[i + 3];

        // Compute for four elements at once
        double cos_term1 = std::cos(alpha1 - alpha0);
        double sin_term1 = std::sin(alpha1 - alpha0);
        result_ptr[i] = (std::pow(s1, 2) + std::pow(s0, 2) - 
                         2 * s1 * s0 * cos_term1) / 
                         (std::pow(sin_term1, 2) + t) - std::pow(R, 2);

        double cos_term2 = std::cos(alpha2 - alpha0);
        double sin_term2 = std::sin(alpha2 - alpha0);
        result_ptr[i + 1] = (std::pow(s2, 2) + std::pow(s0, 2) - 
                             2 * s2 * s0 * cos_term2) / 
                             (std::pow(sin_term2, 2) + t) - std::pow(R, 2);

        double cos_term3 = std::cos(alpha3 - alpha0);
        double sin_term3 = std::sin(alpha3 - alpha0);
        result_ptr[i + 2] = (std::pow(s3, 2) + std::pow(s0, 2) - 
                             2 * s3 * s0 * cos_term3) / 
                             (std::pow(sin_term3, 2) + t) - std::pow(R, 2);

        double cos_term4 = std::cos(alpha4 - alpha0);
        double sin_term4 = std::sin(alpha4 - alpha0);
        result_ptr[i + 3] = (std::pow(s4, 2) + std::pow(s0, 2) - 
                             2 * s4 * s0 * cos_term4) / 
                             (std::pow(sin_term4, 2) + t) - std::pow(R, 2);
    }

    // Handle remaining elements without parallelism
    for (size_t i = total_size - total_size % 4; i < total_size; i++) {
        double alpha_val = alpha_ptr[i];
        double s_val = s_ptr[i];

        double cos_term = std::cos(alpha_val - alpha0);
        double sin_term = std::sin(alpha_val - alpha0);
        result_ptr[i] = (std::pow(s_val, 2) + std::pow(s0, 2) - 
                         2 * s_val * s0 * cos_term) / 
                         (std::pow(sin_term, 2) + t) - std::pow(R, 2);
    }

    return result.reshape(alpha_buf.shape);
}




py::array_t<double> maskfun(
    double alpha0, double s0,
    py::array_t<double> ALPHA, py::array_t<double> S,
    int R, double tstart, double tend) {

    // Create the mask array and get a raw pointer
    py::array_t<double> mask(ALPHA.request().shape);
    auto mask_ptr = static_cast<double*>(mask.request().ptr);
    std::fill(mask_ptr, mask_ptr + ALPHA.size(), 0.0);

    // Evaluate p_regionfun at tstart
    auto mask_condition = p_regionfun(alpha0, s0, ALPHA, S, R, tstart);
    auto mask_condition_ptr = static_cast<double*>(mask_condition.request().ptr);

    // Update mask for the initial condition directly
    #pragma omp parallel for
    for (size_t i = 0; i < ALPHA.size(); i++) {
        if (mask_condition_ptr[i] < 0) {
            mask_ptr[i] = 1.0;
        }
    }

    // Time steps
    size_t n = 4;
    std::vector<double> t(n);
    for (size_t i = 0; i < n; i++) {
        t[i] = tstart + i * (tend - tstart) / (n - 1);
    }

    // Optimize by reducing memory operations and copying
    for (size_t i = 0; i < n - 1; i++) {
        auto region_start = p_regionfun(alpha0, s0, ALPHA, S, R, t[i]);
        auto region_end = p_regionfun(alpha0, s0, ALPHA, S, R, t[i + 1]);

        auto start_ptr = static_cast<double*>(region_start.request().ptr);
        auto end_ptr = static_cast<double*>(region_end.request().ptr);

        #pragma omp parallel for
        for (size_t j = 0; j < ALPHA.size(); j++) {
            if (start_ptr[j] > 0 && end_ptr[j] < 0) {
                mask_ptr[j] = (tend - t[i]) / (tend - tstart);
            }
        }
    }

    return mask;
}


// Function: p_line_fun
py::array_t<double> p_line_fun(
    double alpha0, double s0,
    py::array_t<double> ALPHA, py::array_t<double> S,
    int R, py::array_t<double> L,
    double tstart, double tend, double d, double p_min) {

    // Generate the mask (no change here, assuming maskfun is already optimized)
    auto mask = maskfun(alpha0, s0, ALPHA, S, R, tstart, tend);

    // Access the raw memory buffers of ALPHA, L, and mask
    auto alpha_buf = ALPHA.request();
    auto l_buf = L.request();
    auto mask_buf = mask.request();

    const double* alpha_ptr = static_cast<double*>(alpha_buf.ptr);
    const double* l_ptr = static_cast<double*>(l_buf.ptr);
    const double* mask_ptr = static_cast<double*>(mask_buf.ptr);

    // Prepare the result array
    py::array_t<double> p_line(ALPHA.size());
    auto p_line_ptr = static_cast<double*>(p_line.request().ptr);

    // Compute the result directly in memory
    #pragma omp parallel for
    for (size_t i = 0; i < ALPHA.size(); i++) {
        double sin_term = std::abs(std::sin(alpha_ptr[i] - alpha0));
        double factor = d * p_min / ((d * l_ptr[i] - p_min) * sin_term + p_min);
        p_line_ptr[i] = factor * mask_ptr[i];
    }

    return p_line.reshape(alpha_buf.shape);
}



// Pybind11 module definition
PYBIND11_MODULE(CreateFunctionsFinal9, m) {
    m.def("find_nearest_nail", &find_nearest_nail, "Find the nearest nail index");
    m.def("p_regionfun", &p_regionfun, "Calculate p_region");
    m.def("maskfun", &maskfun, "Generate mask");
    m.def("p_line_fun", &p_line_fun, "Calculate p_line");
}
