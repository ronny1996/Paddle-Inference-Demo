// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>

#include "paddle/include/experimental/ext_all.h"

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cpu_forward(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), x.size());
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cpu_backward(const paddle::Tensor& x,
                                              const paddle::Tensor& out,
                                              const paddle::Tensor& grad_out) {
  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(x.place()),
                                   out.size());
                             }));

  return {grad_x};
}

std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  return relu_cpu_forward(x);
}

std::vector<paddle::Tensor> ReluBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out) {
  return relu_cpu_backward(x, out, grad_out);
}

PD_BUILD_OP(custom_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluForward));

PD_BUILD_GRAD_OP(custom_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluBackward));
