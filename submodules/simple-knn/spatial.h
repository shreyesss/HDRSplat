#include <torch/extension.h>

torch::Tensor distCUDA2(const torch::Tensor& points);
// torch::Tensor distCUDA_with_points(const torch::Tensor& points);