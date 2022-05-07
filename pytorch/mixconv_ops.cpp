#include <torch/extension.h>
#include "mixconv.h"
#include <iostream>

void torch_launch_mixconv(const torch::Tensor &bottom_data,
                        const torch::Tensor &weight_data,
                        torch::Tensor &top_data,
                        const torch::Tensor &weight_ptr_list,
                        const torch::Tensor &kernel_h_list,
                        const torch::Tensor &kernel_w_list,
                        const torch::Tensor &pad_h_list,
                        const torch::Tensor &pad_w_list,
                        const int64_t stride_h, const int64_t stride_w) {
    
    // std::cout << weight_data << std::endl;

    launch_mixconv(top_data.size(0), top_data.size(1),
                top_data.size(2), top_data.size(3),
                (const int64_t *) weight_ptr_list.data_ptr(),
                (const int64_t *) kernel_h_list.data_ptr(),
                (const int64_t *) kernel_w_list.data_ptr(),
                (const int64_t *) pad_h_list.data_ptr(),
                (const int64_t *) pad_w_list.data_ptr(),
                stride_h, stride_w,
                bottom_data.size(2), bottom_data.size(3),
                (const float *) bottom_data.data_ptr(),
                (const float *) weight_data.data_ptr(),
                (float *) top_data.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_mixconv",
          &torch_launch_mixconv,
          "mixconv kernel warpper");
}

TORCH_LIBRARY(mixconv, m) {
    m.def("torch_launch_mixconv", torch_launch_mixconv);
}