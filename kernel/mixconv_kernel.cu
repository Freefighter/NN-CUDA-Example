#include <stdio.h>

__global__ void MixConv2DForward(const int64_t channels,
                                const int64_t height, const int64_t width,
                                const int64_t *weight_ptr_list,
                                const int64_t *kernel_h_list,
                                const int64_t *kernel_w_list,
                                const int64_t *pad_h_list,
                                const int64_t *pad_w_list,
                                const int64_t stride_h, const int64_t stride_w,
                                const int64_t bottom_height, const int64_t bottom_width,
                                const float *bottom_data,
                                const float *weight_data,
                                float *top_data) {
    
    // Compute thread id
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // compute the position w.r.t. (N,C,H,W) based on the cuda thread index
    const int n = index / channels / height / width;
    const int c = (index / height / width) % channels;
    const int h = (index / width) % height;
    const int w = index % width;

    // move the weight pointer to the c-th convolutional kernel weight matrix
    const float *weight = weight_data + weight_ptr_list[c];
    
    // read the kernel size and the padding size
    const int kernel_h = kernel_h_list[c];
    const int kernel_w = kernel_w_list[c];
    const int pad_h = pad_h_list[c];
    const int pad_w = pad_w_list[c];

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = -pad_h + h * stride_h + kh;
            const int w_in = -pad_w + w * stride_w + kw;

            if ((h_in >= 0) && (h_in < bottom_height)
            && (w_in >=0) && (w_in < bottom_width)) {
                // compute the index of the corresponding input element
                const int offset = ((n * channels + c) * bottom_height + h_in)
                    * bottom_width + w_in;
                // update the value by gradually doing Hadamard product
                top_data[index] += (*weight) * bottom_data[offset];
            }
            ++weight;
        }
    }
}

void launch_mixconv(const int64_t output_size, const int64_t channels,
            const int64_t height, const int64_t width,
            const int64_t *weight_ptr_list,
            const int64_t *kernel_h_list,
            const int64_t *kernel_w_list,
            const int64_t *pad_h_list,
            const int64_t *pad_w_list,
            const int64_t stride_h, const int64_t stride_w,
            const int64_t bottom_height, const int64_t bottom_width,
            const float *bottom_data,
            const float *weight_data,
            float *top_data) {

    dim3 grid(output_size*channels);
    dim3 block(height*width);

    MixConv2DForward<<<grid, block>>>(channels, height, width, 
                                weight_ptr_list, 
                                kernel_h_list, kernel_w_list,
                                pad_h_list, pad_w_list,
                                stride_h, stride_w,
                                bottom_height, bottom_width,
                                bottom_data, weight_data, top_data);
}