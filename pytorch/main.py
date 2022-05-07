import time
import numpy as np
import torch
import torch.nn as nn

def _split_channels(num_chan, num_groups): # group the channels
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)

    # return the number of channels in each group
    return split

class MixConv(nn.Module):
    def __init__(self,
                    in_channels, 
                    kernel_size=[3,5,7], 
                    padding=None, 
                    stride=1):
        super(MixConv, self).__init__()
        if padding is None: # padding to make the output have the same shape
            padding = [(k-1)//2 for k in kernel_size]

        self.num_groups = len(kernel_size)
        self.in_splits = _split_channels(in_channels, self.num_groups)
        self.layer = nn.ModuleList([]) # initialize the kernels
        for c, k, p in zip(self.in_splits, kernel_size, padding):
            self.layer.append(
                nn.Conv2d(c, c, k, stride, p, groups=c, bias=False)
                # dw-conv can be realized when groups == c_in and 
                # c_out == K * c_in, where K is a positive integer, 
                # this operation is also known as a "depthwise convolution".
            )
        
        self.kernel_size = torch.tensor(kernel_size)
        self.padding = torch.tensor(padding)
        self.weight_data = torch.cat([i.weight.view(-1) 
                                    for i in self.layer])
        for i in self.layer:
            print(i.weight.shape)
            
        self.weight_ptr_list = torch.repeat_interleave(self.kernel_size**2,
                                    torch.tensor(self.in_splits))
        self.weight_ptr_list = torch.cumsum(self.weight_ptr_list, 0).roll(1, 0)
        self.weight_ptr_list[0] = 0
        self.stride = stride
    
    def prepare_before_cuda(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.kernel_size = self.kernel_size.cuda()
        self.padding = self.padding.cuda()
        self.weight_data = self.weight_data.cuda()
        self.weight_ptr_list =  self.weight_ptr_list.cuda()

        self.kernel_size = self.kernel_size.detach()
        self.padding = self.padding.detach()
        self.weight_data = self.weight_data.detach()
        self.weight_ptr_list =  self.weight_ptr_list.detach()

        print(self.kernel_size.device, self.padding.device, 
                self.weight_data.device, self.weight_ptr_list.device,
                self.weight_ptr_list.shape, self.in_splits)
        torch.ops.load_library("libmixconv.so")
        # from torch.utils.cpp_extension import load
        # self.cuda_module = load(name="mixconv",
        #                         extra_include_paths=["../include"],
        #                         sources=["../pytorch/mixconv_ops.cpp", 
        #                                 "../kernel/mixconv_kernel.cu"],
        #                         verbose=True)

    def cuda_forward(self, x):
        res = torch.empty_like(x)
        # tmp = torch.rand_like(x, device="cuda:0")
        # print(self.weight_data.shape, self.weight_data.is_contiguous(), 
        #         self.weight_data.dtype, x.dtype, x.shape, 
        #         self.layer[0].weight.device, self.weight_data.device, tmp.device, tmp.dtype)
        """(const torch::Tensor &bottom_data,
                        const torch::Tensor &weight_data,
                        torch::Tensor &top_data,
                        const torch::Tensor &weight_ptr_list,
                        const torch::Tensor &kernel_h_list,
                        const torch::Tensor &kernel_w_list,
                        const torch::Tensor &pad_h_list,
                        const torch::Tensor &pad_w_list,
                        const int64_t stride_h, const int64_t stride_w)"""
        # self.cuda_module.torch_launch_mixconv(
        torch.ops.mixconv.torch_launch_mixconv(
            x, self.weight_data, res, 
            self.weight_ptr_list, self.kernel_size, self.kernel_size,
            self.padding, self.padding, self.stride, self.stride)
        
        # print(res.sum())

        return res
    
    def forward(self, x):
        out = []
        x_split = torch.split(x, self.in_splits, dim=1)
        for m, _x in zip(self.layer, x_split): # compute the output w/ for-loop
            out.append(m(_x))

        return torch.cat(out, dim=1) # concatenate to have the final output


def show_time(func, ntest, X):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func(X)
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        res = func(X)
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

if __name__ == "__main__":

    for n in range(3, 23, 2):
        # N, C, H, W = 2, 3, 64, 64
        N, C, H, W = 16, n, 32, 32
        # N, C, H, W = [16, 512, 32, 32]
        X = torch.rand((N, C, H, W))
        kernel_size = [2*i+3 for i in range(n)]

        ntest = 30

        # ask PyTorch to not track the gradient to avoid unnecessary computation
        torch_conv = MixConv(C, kernel_size=kernel_size)
        for param in torch_conv.layer.parameters():
            param.requires_grad = False
        
        print("Running torch...")
        cpu_time, cpu_res = show_time(torch_conv.forward, ntest, X)
        print("CPU time:  {:.3f}us".format(np.mean(cpu_time)))
        
        X = X.cuda()
        torch_conv.to("cuda:0")
        torch_conv.prepare_before_cuda()

        print("Running cuda...")
        cuda_time, cuda_res = show_time(torch_conv.cuda_forward, ntest, X)
        print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

        # print(cuda_res.sum())

        print("Running torch...")
        torch_time, torch_res = show_time(torch_conv.forward, ntest, X)
        print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

        # print(torch_res.sum())

        print("%f\t%f\t%f\t%f\t%f\t%f\n"%(np.mean(cuda_time), 
                        np.std(cuda_time, ddof=1) / np.sqrt(ntest),
                        np.mean(torch_time), 
                        np.std(torch_time, ddof=1) / np.sqrt(ntest),
                        np.mean(cpu_time), 
                        np.std(cpu_time, ddof=1) / np.sqrt(ntest)))