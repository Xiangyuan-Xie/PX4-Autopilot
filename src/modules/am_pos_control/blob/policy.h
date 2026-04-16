#pragma once
#include <rl_tools/containers/matrix/matrix.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn/parameters/parameters.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/nn/layers/gru/layer.h>
#include <rl_tools/nn_models/sequential/model.h>

namespace rl_tools::checkpoint::actor {
    namespace layer_0 {
        namespace weights {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    158, 119, 103, 191, 160, 239, 15, 63, 11, 81, 53, 63, 234, 7, 50, 189,
                    205, 217, 12, 192, 158, 53, 189, 190, 6, 215, 38, 64, 97, 235, 194, 189,
                    238, 38, 28, 64, 84, 154, 62, 62, 238, 72, 54, 192, 110, 16, 245, 61,
                    56, 163, 19, 63, 188, 227, 112, 192, 245, 179, 155, 61, 95, 235, 192, 60,
                    95, 209, 57, 63, 176, 101, 254, 61, 82, 169, 37, 191, 82, 120, 152, 61,
                    144, 206, 65, 64, 127, 71, 160, 188, 49, 135, 10, 62, 69, 197, 133, 61,
                    169, 216, 177, 189, 148, 85, 46, 191, 57, 172, 203, 61, 122, 193, 173, 62,
                    238, 191, 101, 63, 122, 173, 194, 60, 135, 254, 100, 191, 89, 41, 151, 191,
                    185, 142, 130, 60, 97, 140, 187, 189, 73, 140, 228, 62, 229, 102, 211, 62,
                    75, 58, 19, 191, 166, 46, 57, 63, 22, 143, 214, 61, 137, 216, 105, 64,
                    198, 182, 119, 191, 207, 173, 120, 192, 233, 70, 241, 189, 57, 126, 31, 191,
                    57, 205, 54, 63, 116, 45, 173, 63, 14, 33, 67, 189, 82, 77, 16, 192,
                    150, 1, 215, 191, 247, 178, 53, 190, 74, 37, 175, 62, 74, 190, 244, 61,
                    62, 196, 129, 62, 227, 111, 227, 61, 180, 32, 223, 61, 97, 174, 72, 192,
                    9, 162, 100, 189, 240, 98, 224, 61, 72, 49, 236, 61, 80, 144, 222, 189,
                    184, 113, 37, 191, 36, 193, 103, 189, 242, 53, 197, 190, 78, 161, 12, 63,
                    122, 118, 18, 191, 101, 43, 170, 59, 118, 75, 124, 61, 84, 174, 129, 190,
                    229, 55, 68, 62, 190, 26, 143, 191, 192, 13, 50, 64, 173, 70, 96, 192,
                    157, 220, 145, 192, 50, 231, 153, 189, 76, 213, 22, 191, 7, 82, 72, 189,
                    193, 222, 15, 60, 49, 205, 7, 61, 118, 225, 23, 191, 179, 66, 163, 190,
                    210, 119, 156, 63, 77, 229, 153, 61, 151, 56, 246, 191, 73, 219, 160, 64,
                    131, 201, 153, 61, 22, 253, 18, 64, 197, 173, 91, 192, 121, 11, 118, 192,
                    2, 99, 220, 62, 80, 43, 205, 62, 14, 39, 5, 62, 175, 45, 4, 188,
                    234, 59, 236, 61, 35, 244, 102, 61, 81, 75, 113, 60, 174, 206, 50, 189,
                    192, 247, 85, 190, 77, 168, 151, 62, 220, 28, 153, 62, 101, 92, 5, 190,
                    82, 208, 1, 191, 31, 185, 156, 61, 97, 150, 116, 60, 65, 223, 0, 190,
                    227, 93, 132, 189, 227, 184, 82, 191, 62, 85, 233, 189, 252, 176, 203, 63,
                    219, 213, 27, 189, 66, 187, 131, 63, 149, 145, 166, 191, 133, 128, 124, 191,
                    88, 232, 135, 61, 109, 253, 75, 191, 148, 61, 206, 63, 93, 254, 148, 63,
                    3, 137, 38, 62, 167, 170, 52, 64, 22, 66, 90, 64, 6, 105, 174, 61,
                    182, 147, 11, 191, 101, 56, 56, 191, 238, 195, 142, 63, 83, 148, 247, 63,
                    16, 211, 114, 191, 52, 141, 12, 192, 176, 152, 15, 190, 255, 148, 69, 189,
                    140, 74, 18, 61, 157, 10, 13, 189, 110, 60, 24, 190, 81, 139, 222, 190,
                    232, 92, 182, 189, 68, 30, 47, 62, 211, 117, 72, 61, 117, 66, 23, 63,
                    233, 47, 147, 190, 161, 34, 128, 62, 171, 251, 7, 61, 209, 156, 189, 190,
                    129, 119, 20, 64, 181, 140, 81, 64, 20, 195, 94, 192, 8, 114, 164, 189,
                    40, 220, 83, 191, 101, 107, 143, 190, 122, 14, 132, 62, 216, 100, 61, 62,
                    22, 158, 156, 191, 146, 159, 134, 61, 95, 181, 151, 63, 239, 69, 28, 59,
                    137, 9, 44, 192, 108, 102, 209, 192, 182, 100, 169, 189, 213, 44, 176, 63,
                    133, 108, 63, 64, 116, 239, 62, 192, 55, 32, 109, 191, 189, 59, 0, 63,
                    168, 85, 140, 63, 66, 85, 4, 190, 245, 229, 160, 189, 71, 178, 180, 187,
                    159, 238, 205, 188, 73, 29, 205, 188, 214, 213, 171, 62, 52, 170, 26, 62,
                    101, 38, 44, 189, 229, 70, 217, 61, 80, 0, 181, 63, 185, 4, 180, 189,
                    102, 37, 169, 59, 28, 103, 198, 61, 231, 98, 183, 188, 244, 154, 120, 63,
                    38, 17, 58, 191, 203, 246, 238, 62, 55, 94, 50, 189, 13, 1, 103, 62,
                    10, 181, 127, 63, 127, 4, 18, 191, 123, 248, 8, 190, 90, 243, 168, 191,
                    228, 219, 152, 191, 150, 84, 170, 63, 154, 13, 239, 61, 160, 70, 114, 192,
                    133, 224, 128, 63, 167, 33, 107, 190, 90, 155, 64, 63, 84, 104, 44, 190,
                    233, 75, 146, 62, 28, 184, 52, 61, 28, 193, 175, 63, 238, 94, 158, 187,
                    164, 59, 160, 191, 33, 130, 88, 191, 100, 2, 128, 189, 16, 220, 250, 60,
                    227, 16, 238, 62, 243, 216, 77, 63, 167, 165, 5, 63, 174, 88, 35, 62,
                    134, 233, 203, 62, 84, 106, 118, 63, 239, 34, 215, 189, 148, 30, 91, 62,
                    38, 94, 8, 62, 233, 142, 158, 61, 81, 224, 177, 189, 80, 154, 106, 191,
                    34, 13, 162, 190, 19, 173, 126, 189, 195, 242, 127, 191, 121, 70, 14, 63,
                    234, 115, 235, 62, 240, 1, 61, 62, 37, 104, 113, 190, 160, 123, 119, 191,
                    115, 104, 174, 190, 198, 20, 107, 62, 173, 160, 44, 190, 74, 193, 196, 191,
                    110, 199, 179, 61, 135, 40, 6, 63, 18, 128, 181, 191, 226, 161, 73, 63,
                    40, 134, 237, 191, 158, 37, 172, 191, 109, 202, 61, 64, 250, 93, 168, 190,
                    35, 115, 181, 190, 134, 2, 219, 189, 34, 171, 150, 61, 90, 180, 11, 63,
                    23, 119, 225, 61, 158, 72, 38, 61, 251, 125, 102, 189, 101, 155, 236, 62,
                    151, 103, 127, 63, 245, 37, 17, 61, 46, 178, 86, 61, 191, 202, 177, 189,
                    111, 191, 10, 190, 108, 115, 147, 63, 100, 146, 135, 192, 65, 252, 167, 190,
                    255, 138, 239, 61, 171, 251, 83, 62, 108, 38, 101, 63, 179, 140, 178, 60,
                    137, 6, 196, 189, 66, 72, 12, 192, 65, 204, 138, 191, 188, 43, 62, 64,
                    154, 253, 67, 187, 65, 63, 61, 192, 171, 60, 188, 64, 241, 212, 148, 190,
                    93, 237, 15, 63, 77, 32, 209, 191, 45, 36, 102, 62, 166, 156, 65, 64,
                    204, 68, 143, 63, 50, 23, 216, 188, 31, 196, 190, 61, 112, 163, 81, 189,
                    24, 163, 111, 189, 101, 58, 88, 60, 194, 222, 189, 61, 39, 54, 79, 63,
                    181, 31, 58, 63, 110, 39, 238, 61, 245, 155, 104, 190, 177, 106, 3, 64,
                    137, 58, 82, 190, 35, 24, 167, 60, 20, 169, 37, 60, 201, 166, 126, 188,
                    74, 166, 11, 62, 48, 174, 202, 62, 43, 149, 42, 61, 178, 254, 49, 62,
                    153, 148, 93, 62, 32, 155, 81, 191, 199, 220, 254, 190, 133, 216, 27, 62,
                    41, 38, 237, 191, 86, 153, 96, 62, 29, 224, 165, 63, 254, 76, 132, 61,
                    222, 156, 66, 63, 100, 80, 84, 64, 91, 191, 245, 189, 140, 111, 135, 62,
                    20, 89, 1, 63, 215, 222, 104, 62, 72, 164, 2, 64, 33, 80, 57, 191,
                    106, 242, 214, 63, 160, 236, 32, 62, 117, 240, 204, 59, 89, 11, 104, 61,
                    68, 81, 83, 189, 205, 38, 91, 63, 210, 178, 67, 62, 52, 119, 70, 191,
                    16, 215, 13, 191, 172, 120, 68, 62, 138, 163, 56, 62, 85, 204, 55, 62,
                    47, 185, 75, 190, 12, 159, 88, 191, 105, 197, 153, 190, 215, 194, 232, 63,
                    229, 99, 128, 190, 222, 72, 48, 61, 53, 200, 9, 190, 180, 91, 49, 191,
                    93, 69, 108, 63, 237, 58, 143, 63, 57, 152, 92, 190, 247, 114, 103, 62,
                    254, 187, 200, 191, 12, 157, 140, 62, 50, 114, 20, 190, 28, 39, 84, 192,
                    235, 194, 45, 63, 51, 238, 129, 189, 205, 139, 69, 63, 120, 127, 215, 190,
                    198, 214, 105, 191, 78, 1, 196, 63, 214, 61, 99, 63, 23, 38, 231, 63,
                    15, 147, 130, 62, 132, 214, 31, 62, 69, 170, 195, 61, 20, 94, 162, 189,
                    53, 101, 100, 191, 196, 68, 163, 188, 105, 72, 45, 62, 236, 78, 198, 189,
                    167, 20, 169, 62, 40, 152, 1, 62, 60, 12, 130, 190, 39, 40, 109, 62,
                    136, 255, 179, 62, 44, 254, 0, 190, 22, 149, 197, 63, 34, 161, 12, 192,
                    181, 88, 89, 189, 46, 206, 30, 61, 122, 123, 254, 191, 243, 144, 28, 63,
                    60, 212, 244, 63, 190, 93, 24, 62, 241, 208, 70, 63, 148, 49, 156, 191,
                    128, 132, 130, 191, 82, 196, 214, 61, 61, 248, 86, 192, 132, 63, 78, 64,
                    248, 34, 134, 62, 52, 15, 132, 63, 201, 223, 58, 191, 116, 82, 237, 62,
                    216, 4, 72, 63, 161, 65, 159, 62, 66, 9, 74, 64, 70, 53, 154, 61,
                    157, 147, 11, 62, 99, 242, 233, 60, 8, 11, 3, 190, 14, 17, 56, 190,
                    188, 254, 137, 61, 16, 5, 59, 62, 248, 23, 182, 190, 209, 224, 112, 186,
                    109, 172, 92, 63, 183, 241, 197, 189, 53, 196, 131, 191, 246, 11, 190, 62,
                    122, 249, 3, 61, 146, 205, 169, 63, 224, 205, 18, 191, 227, 250, 246, 62,
                    203, 177, 2, 190, 191, 74, 22, 60, 89, 180, 31, 64, 154, 52, 130, 62,
                    44, 120, 178, 189, 220, 25, 48, 63, 116, 84, 42, 192, 93, 171, 19, 191,
                    15, 236, 22, 190, 154, 37, 143, 192, 72, 80, 196, 62, 16, 73, 16, 190,
                    208, 168, 254, 62, 237, 135, 94, 61, 120, 174, 102, 63, 14, 22, 214, 189,
                    93, 240, 30, 64, 58, 35, 130, 63, 105, 103, 25, 62, 186, 205, 173, 61,
                    183, 125, 222, 187, 245, 83, 144, 61, 114, 217, 152, 62, 186, 160, 60, 63,
                    125, 40, 138, 189, 172, 22, 52, 190, 49, 82, 235, 59, 211, 0, 24, 191,
                    53, 176, 212, 188, 43, 117, 37, 62, 220, 181, 241, 190, 107, 197, 170, 61,
                    92, 148, 155, 191, 32, 164, 171, 63, 6, 110, 174, 189, 5, 13, 77, 62,
                    244, 18, 146, 61, 8, 77, 239, 62, 63, 45, 62, 189, 195, 242, 85, 189,
                    194, 247, 87, 190, 98, 15, 100, 188, 176, 104, 185, 190, 52, 232, 212, 60,
                    196, 29, 112, 64, 80, 146, 40, 191, 199, 191, 32, 62, 202, 41, 180, 190,
                    165, 239, 79, 190, 234, 113, 235, 61, 35, 228, 134, 189, 175, 4, 3, 192,
                    32, 49, 169, 190, 237, 19, 231, 186, 40, 117, 199, 190, 206, 66, 187, 189,
                    136, 84, 185, 188, 104, 22, 8, 63, 15, 33, 154, 191, 54, 228, 115, 191,
                    33, 231, 36, 61, 188, 119, 184, 62, 204, 225, 203, 190, 81, 90, 148, 62,
                    170, 173, 136, 190, 7, 190, 32, 63, 70, 35, 60, 190, 216, 151, 41, 62,
                    57, 9, 177, 62, 236, 147, 57, 63, 138, 217, 118, 61, 82, 217, 164, 192,
                    63, 1, 158, 190, 93, 77, 137, 64, 45, 6, 142, 189, 128, 143, 22, 63,
                    165, 145, 63, 63, 98, 195, 69, 191, 232, 111, 208, 60, 123, 97, 223, 191,
                    225, 24, 137, 192, 200, 95, 164, 60, 32, 198, 41, 188, 75, 128, 134, 63,
                    111, 92, 99, 62, 40, 179, 135, 191, 90, 131, 247, 62, 65, 199, 15, 64,
                    152, 134, 40, 62, 107, 98, 88, 189, 68, 41, 128, 61, 208, 222, 162, 61,
                    29, 81, 31, 62, 50, 90, 99, 190, 186, 225, 78, 63, 160, 40, 140, 190,
                    110, 178, 20, 191, 131, 252, 121, 190, 123, 130, 3, 190, 148, 20, 213, 61,
                    65, 126, 187, 62, 163, 119, 99, 191, 107, 254, 78, 192, 168, 149, 228, 190,
                    100, 143, 169, 192, 93, 112, 220, 61, 147, 192, 26, 190, 241, 199, 159, 63,
                    101, 230, 1, 191, 244, 17, 99, 61, 176, 243, 206, 190, 197, 68, 128, 191,
                    94, 27, 110, 63, 160, 199, 7, 62, 132, 149, 103, 64, 250, 238, 53, 191,
                    97, 91, 158, 62, 253, 157, 22, 192, 164, 70, 40, 63, 179, 235, 150, 192,
                    35, 88, 148, 61, 122, 52, 115, 191, 80, 52, 228, 61, 184, 115, 252, 60,
                    208, 95, 185, 60, 200, 152, 196, 188, 0, 22, 7, 61, 73, 160, 28, 191,
                    82, 174, 176, 190, 73, 127, 172, 189, 237, 95, 15, 190, 217, 217, 210, 190,
                    143, 10, 18, 191, 207, 150, 21, 190, 233, 54, 147, 189, 186, 153, 196, 189,
                    93, 14, 128, 188, 209, 248, 159, 63, 176, 31, 24, 64, 137, 225, 80, 191,
                    124, 105, 31, 190, 198, 147, 176, 63, 83, 96, 191, 63, 229, 240, 72, 191,
                    167, 6, 174, 60, 201, 206, 77, 64, 147, 167, 215, 191, 207, 94, 24, 192,
                    206, 97, 57, 60, 129, 236, 131, 192, 214, 252, 109, 192, 7, 51, 36, 190,
                    13, 39, 189, 63, 31, 141, 211, 63, 150, 0, 41, 189, 142, 26, 195, 191,
                    151, 48, 195, 63, 150, 91, 45, 192, 62, 132, 136, 62, 16, 168, 175, 61,
                    24, 48, 120, 188, 89, 126, 94, 59, 79, 49, 154, 190, 51, 48, 103, 63,
                    108, 220, 250, 62, 129, 238, 141, 189, 230, 244, 135, 190, 2, 231, 108, 191,
                    47, 99, 29, 190, 143, 161, 89, 59, 215, 87, 90, 188, 5, 232, 111, 61,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16, 35>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace biases {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    221, 80, 119, 61, 1, 21, 176, 61, 66, 194, 61, 188, 61, 17, 165, 189,
                    199, 174, 81, 189, 139, 11, 28, 61, 222, 236, 59, 62, 43, 192, 121, 188,
                    3, 26, 73, 189, 58, 200, 64, 188, 141, 22, 13, 189, 236, 147, 220, 188,
                    74, 204, 114, 189, 12, 120, 72, 61, 186, 135, 119, 62, 159, 69, 130, 190,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 16, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::ELU, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal>;
        using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 35>;
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
        const TYPE module = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory_function(){return T_TYPE{weights::parameters, biases::parameters};}
    }
    namespace layer_1 {
        namespace weights {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    109, 51, 219, 62, 131, 51, 50, 191, 65, 208, 50, 190, 108, 46, 103, 190,
                    51, 165, 161, 190, 100, 129, 221, 190, 195, 219, 198, 62, 1, 73, 184, 190,
                    183, 107, 102, 62, 214, 99, 207, 61, 78, 184, 8, 190, 173, 227, 102, 61,
                    205, 66, 204, 61, 211, 46, 129, 61, 65, 252, 119, 62, 189, 120, 202, 190,
                    13, 231, 103, 190, 192, 23, 33, 62, 129, 163, 79, 190, 122, 161, 158, 62,
                    159, 34, 236, 190, 217, 86, 58, 191, 23, 32, 145, 62, 163, 239, 155, 190,
                    52, 204, 188, 62, 204, 184, 1, 62, 208, 15, 13, 63, 118, 254, 250, 189,
                    49, 91, 243, 62, 236, 107, 41, 62, 90, 23, 160, 62, 204, 151, 194, 189,
                    126, 21, 195, 62, 167, 74, 5, 62, 6, 185, 83, 190, 41, 53, 231, 62,
                    59, 86, 25, 191, 141, 220, 113, 188, 81, 100, 9, 62, 69, 78, 109, 62,
                    242, 38, 6, 62, 157, 117, 147, 190, 153, 132, 108, 62, 170, 44, 150, 62,
                    201, 119, 159, 190, 165, 97, 187, 61, 47, 184, 24, 191, 68, 52, 202, 62,
                    59, 183, 42, 191, 150, 17, 97, 62, 188, 149, 187, 62, 49, 232, 98, 61,
                    202, 82, 184, 62, 174, 130, 27, 62, 152, 173, 225, 60, 61, 181, 151, 62,
                    229, 135, 7, 60, 62, 239, 254, 61, 136, 47, 10, 190, 188, 200, 87, 189,
                    59, 49, 66, 190, 144, 211, 196, 62, 138, 215, 62, 63, 166, 77, 75, 62,
                    80, 21, 213, 189, 57, 77, 117, 189, 219, 180, 70, 190, 15, 255, 15, 190,
                    209, 46, 172, 61, 229, 124, 156, 190, 243, 203, 1, 63, 159, 250, 95, 191,
                    13, 142, 179, 188, 244, 245, 251, 190, 218, 123, 156, 190, 220, 94, 204, 59,
                    59, 8, 223, 62, 58, 129, 157, 62, 235, 203, 158, 61, 23, 74, 23, 63,
                    185, 82, 128, 190, 3, 46, 124, 190, 3, 134, 221, 60, 248, 143, 27, 189,
                    65, 154, 162, 191, 159, 104, 84, 62, 35, 110, 167, 190, 148, 45, 221, 62,
                    196, 46, 225, 190, 239, 233, 239, 61, 38, 1, 105, 62, 230, 138, 205, 190,
                    60, 153, 165, 189, 155, 181, 31, 61, 187, 0, 102, 190, 89, 148, 38, 191,
                    107, 251, 47, 190, 232, 108, 237, 62, 229, 241, 198, 190, 0, 75, 50, 191,
                    28, 170, 67, 62, 124, 103, 194, 61, 223, 59, 179, 61, 41, 175, 120, 189,
                    38, 172, 42, 60, 63, 50, 192, 61, 174, 76, 197, 62, 113, 162, 63, 190,
                    3, 33, 198, 189, 108, 76, 172, 61, 102, 85, 47, 191, 102, 213, 9, 63,
                    103, 239, 203, 62, 2, 253, 222, 61, 2, 243, 75, 191, 104, 143, 10, 190,
                    59, 244, 185, 61, 165, 219, 108, 190, 40, 11, 240, 190, 205, 61, 39, 191,
                    4, 193, 109, 189, 222, 133, 164, 62, 187, 177, 243, 190, 163, 99, 174, 190,
                    182, 223, 60, 62, 247, 207, 135, 190, 197, 171, 123, 62, 79, 219, 159, 62,
                    217, 92, 35, 190, 67, 27, 241, 189, 8, 202, 182, 62, 95, 17, 214, 62,
                    166, 168, 46, 190, 198, 87, 159, 190, 12, 113, 91, 61, 225, 74, 215, 189,
                    207, 88, 252, 62, 168, 68, 35, 191, 126, 191, 9, 191, 77, 108, 19, 62,
                    50, 192, 54, 190, 13, 252, 58, 191, 4, 242, 130, 63, 83, 22, 131, 62,
                    187, 184, 192, 190, 246, 124, 103, 190, 228, 45, 109, 62, 98, 27, 12, 190,
                    189, 79, 99, 189, 73, 64, 184, 62, 145, 245, 252, 190, 25, 196, 130, 189,
                    37, 136, 71, 62, 198, 63, 195, 62, 13, 233, 168, 62, 187, 167, 44, 62,
                    87, 208, 41, 189, 50, 167, 146, 189, 91, 70, 132, 191, 15, 45, 23, 62,
                    21, 52, 189, 62, 255, 83, 104, 190, 211, 223, 152, 190, 197, 23, 91, 190,
                    254, 79, 156, 62, 33, 49, 47, 59, 236, 55, 27, 62, 145, 194, 13, 63,
                    65, 121, 93, 62, 209, 120, 68, 62, 179, 30, 220, 189, 207, 226, 126, 61,
                    61, 161, 20, 63, 236, 129, 15, 62, 116, 215, 227, 62, 53, 212, 92, 62,
                    156, 168, 12, 190, 81, 73, 187, 190, 233, 128, 109, 62, 85, 247, 23, 191,
                    218, 151, 67, 63, 219, 115, 37, 190, 252, 128, 159, 190, 65, 60, 132, 190,
                    124, 60, 113, 189, 252, 216, 44, 189, 27, 26, 29, 62, 220, 186, 100, 190,
                    247, 115, 90, 61, 105, 250, 159, 190, 24, 152, 131, 63, 248, 163, 218, 61,
                    93, 166, 1, 62, 129, 111, 18, 61, 71, 252, 57, 62, 191, 140, 35, 190,
                    206, 59, 95, 62, 185, 113, 82, 63, 51, 11, 138, 62, 8, 66, 121, 190,
                    141, 139, 184, 190, 46, 187, 10, 191, 144, 92, 121, 188, 65, 192, 204, 62,
                    215, 238, 174, 190, 196, 107, 194, 61, 182, 253, 52, 191, 18, 134, 67, 59,
                    172, 14, 91, 59, 20, 99, 138, 190, 153, 112, 146, 191, 28, 153, 180, 189,
                    140, 196, 134, 62, 61, 36, 168, 190, 221, 62, 44, 191, 27, 89, 58, 191,
                    239, 210, 60, 190, 150, 150, 112, 190, 81, 221, 29, 190, 37, 37, 96, 190,
                    211, 202, 96, 189, 230, 98, 121, 62, 183, 181, 51, 62, 233, 143, 174, 190,
                    185, 51, 165, 62, 207, 180, 15, 62, 140, 21, 132, 61, 166, 140, 160, 189,
                    137, 76, 2, 63, 121, 177, 167, 190, 71, 147, 187, 190, 1, 212, 185, 189,
                    11, 178, 143, 190, 50, 113, 183, 62, 121, 142, 198, 189, 83, 105, 164, 190,
                    115, 236, 166, 190, 235, 174, 122, 62, 46, 52, 216, 62, 93, 128, 5, 191,
                    147, 222, 176, 189, 16, 57, 212, 190, 216, 244, 143, 62, 169, 252, 202, 189,
                    239, 236, 188, 61, 138, 226, 75, 61, 251, 125, 149, 62, 111, 243, 145, 60,
                    137, 55, 244, 62, 237, 169, 255, 190, 84, 113, 168, 62, 234, 59, 162, 62,
                    214, 35, 149, 190, 246, 107, 121, 62, 2, 55, 200, 62, 145, 76, 130, 191,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace biases {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    92, 32, 253, 189, 152, 81, 153, 190, 203, 66, 169, 190, 111, 92, 178, 190,
                    51, 107, 96, 190, 205, 205, 56, 190, 203, 90, 36, 190, 127, 199, 170, 188,
                    195, 114, 8, 191, 220, 85, 56, 190, 226, 250, 198, 189, 30, 50, 9, 190,
                    187, 181, 163, 190, 107, 147, 184, 61, 135, 227, 149, 190, 93, 106, 43, 190,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 16, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::ELU, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal>;
        using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 16>;
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
        const TYPE module = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory_function(){return T_TYPE{weights::parameters, biases::parameters};}
    }
    namespace layer_2 {
        namespace weights {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    231, 42, 232, 188, 96, 67, 137, 190, 3, 124, 129, 62, 31, 57, 154, 62,
                    134, 74, 75, 59, 42, 21, 158, 190, 157, 209, 248, 62, 138, 223, 173, 62,
                    107, 240, 176, 190, 208, 218, 19, 62, 146, 183, 178, 190, 24, 59, 138, 190,
                    160, 185, 193, 61, 54, 68, 76, 62, 93, 105, 143, 189, 83, 14, 207, 60,
                    182, 73, 131, 60, 28, 179, 168, 190, 123, 36, 21, 191, 2, 182, 151, 61,
                    175, 2, 148, 60, 36, 134, 129, 190, 131, 101, 49, 62, 212, 32, 59, 188,
                    64, 71, 218, 61, 129, 108, 163, 190, 86, 54, 86, 190, 223, 255, 89, 190,
                    175, 234, 236, 190, 50, 219, 140, 190, 230, 76, 1, 191, 22, 164, 184, 189,
                    101, 197, 236, 61, 28, 247, 211, 62, 18, 21, 104, 59, 118, 253, 25, 190,
                    88, 115, 65, 61, 209, 97, 153, 62, 241, 83, 131, 190, 213, 123, 194, 62,
                    32, 176, 176, 62, 196, 69, 158, 190, 230, 75, 9, 190, 254, 147, 193, 190,
                    54, 196, 206, 190, 189, 91, 113, 62, 144, 133, 72, 190, 12, 2, 27, 190,
                    149, 198, 94, 190, 151, 118, 38, 190, 156, 175, 39, 62, 45, 188, 54, 61,
                    254, 252, 56, 62, 169, 244, 156, 190, 129, 226, 249, 190, 247, 230, 116, 190,
                    47, 253, 238, 190, 154, 35, 11, 190, 171, 141, 145, 62, 19, 108, 187, 61,
                    175, 208, 174, 189, 244, 201, 86, 62, 24, 36, 9, 189, 66, 109, 156, 189,
                    233, 146, 219, 187, 112, 25, 2, 63, 180, 99, 183, 60, 138, 173, 201, 61,
                    202, 46, 176, 190, 149, 55, 82, 60, 228, 189, 112, 62, 136, 5, 51, 189,
                    102, 94, 201, 190, 214, 108, 165, 190, 205, 115, 4, 61, 202, 26, 153, 62,
                    20, 189, 88, 190, 219, 224, 157, 62, 31, 146, 43, 62, 70, 59, 168, 62,
                    26, 173, 203, 188, 237, 131, 47, 188, 46, 33, 115, 190, 140, 27, 23, 62,
                    80, 26, 212, 61, 17, 33, 169, 61, 141, 104, 165, 61, 44, 180, 179, 62,
                    71, 250, 225, 61, 131, 101, 70, 59, 128, 164, 120, 189, 231, 155, 211, 189,
                    72, 84, 141, 62, 52, 21, 139, 62, 149, 190, 129, 61, 71, 140, 215, 190,
                    50, 117, 74, 190, 97, 133, 213, 60, 0, 216, 82, 189, 176, 53, 119, 62,
                    230, 241, 88, 189, 65, 145, 133, 62, 5, 214, 0, 190, 17, 213, 141, 189,
                    80, 170, 123, 190, 30, 229, 83, 59, 21, 247, 50, 187, 84, 210, 43, 190,
                    11, 70, 112, 62, 248, 14, 83, 190, 18, 24, 86, 190, 128, 177, 17, 190,
                    243, 29, 23, 191, 49, 35, 177, 62, 68, 125, 81, 190, 30, 123, 144, 61,
                    193, 251, 219, 59, 79, 84, 217, 188, 29, 207, 147, 62, 206, 35, 10, 191,
                    126, 194, 241, 62, 252, 172, 34, 62, 158, 248, 150, 189, 93, 63, 131, 61,
                    169, 161, 48, 187, 4, 194, 141, 61, 136, 209, 156, 190, 150, 19, 52, 190,
                    82, 110, 147, 62, 96, 167, 243, 189, 240, 179, 163, 189, 31, 51, 68, 62,
                    6, 154, 60, 62, 212, 126, 43, 62, 163, 116, 23, 190, 116, 77, 186, 62,
                    179, 47, 164, 62, 192, 53, 207, 189, 75, 152, 250, 62, 234, 34, 136, 190,
                    145, 66, 103, 189, 161, 29, 87, 190, 151, 109, 189, 189, 190, 170, 79, 189,
                    72, 242, 152, 190, 90, 11, 154, 189, 58, 42, 152, 62, 83, 201, 61, 62,
                    225, 36, 0, 191, 175, 109, 238, 190, 23, 21, 78, 190, 159, 237, 4, 190,
                    236, 233, 34, 190, 74, 34, 0, 62, 120, 99, 130, 62, 72, 140, 171, 190,
                    69, 160, 149, 189, 146, 154, 21, 189, 129, 124, 23, 189, 85, 63, 35, 190,
                    154, 111, 60, 190, 157, 83, 0, 62, 155, 167, 170, 61, 86, 115, 9, 190,
                    163, 215, 170, 190, 240, 71, 37, 190, 253, 239, 98, 190, 190, 49, 190, 61,
                    247, 137, 109, 62, 25, 236, 4, 62, 92, 52, 193, 60, 252, 75, 14, 190,
                    86, 191, 31, 185, 93, 63, 30, 60, 130, 157, 196, 61, 25, 214, 196, 190,
                    136, 248, 77, 61, 20, 189, 42, 190, 150, 94, 94, 190, 210, 196, 161, 189,
                    76, 43, 4, 191, 166, 131, 238, 62, 185, 162, 229, 189, 191, 152, 154, 190,
                    69, 224, 128, 62, 119, 230, 194, 62, 57, 95, 82, 61, 201, 75, 252, 189,
                    171, 187, 56, 190, 248, 29, 184, 190, 145, 121, 23, 187, 252, 193, 69, 61,
                    133, 107, 153, 62, 125, 163, 214, 61, 152, 44, 130, 190, 125, 210, 169, 189,
                    10, 194, 241, 61, 241, 42, 194, 190, 251, 28, 4, 190, 146, 1, 18, 62,
                    36, 79, 53, 190, 148, 59, 18, 61, 217, 6, 131, 62, 107, 255, 129, 61,
                    212, 46, 100, 62, 97, 208, 181, 187, 198, 131, 106, 190, 138, 81, 116, 58,
                    27, 155, 20, 62, 16, 6, 33, 61, 134, 156, 58, 190, 80, 208, 168, 190,
                    126, 119, 71, 189, 30, 198, 252, 61, 151, 44, 162, 62, 89, 181, 22, 190,
                    171, 95, 183, 189, 226, 145, 200, 61, 118, 220, 187, 61, 73, 154, 50, 191,
                    146, 177, 92, 189, 194, 4, 133, 62, 137, 65, 101, 190, 101, 104, 217, 61,
                    188, 124, 2, 190, 86, 150, 84, 190, 24, 68, 223, 189, 162, 141, 140, 190,
                    57, 50, 144, 190, 209, 66, 145, 190, 244, 200, 47, 62, 179, 78, 69, 189,
                    10, 56, 104, 62, 133, 148, 29, 61, 110, 187, 40, 62, 191, 53, 31, 189,
                    209, 189, 11, 190, 196, 58, 152, 62, 70, 15, 151, 190, 89, 217, 175, 62,
                    245, 20, 113, 62, 162, 234, 238, 189, 217, 35, 140, 62, 125, 33, 212, 190,
                    198, 103, 17, 190, 17, 228, 39, 62, 149, 68, 103, 62, 125, 128, 182, 190,
                    8, 228, 137, 62, 64, 93, 205, 62, 10, 34, 222, 189, 155, 184, 22, 63,
                    241, 204, 187, 61, 190, 13, 71, 62, 193, 83, 129, 190, 19, 28, 122, 190,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace biases {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    69, 5, 176, 61, 31, 31, 129, 190, 42, 196, 170, 61, 148, 135, 15, 189,
                    46, 89, 190, 61, 28, 44, 137, 189, 170, 183, 51, 189, 81, 251, 154, 188,
                    208, 8, 58, 61, 97, 40, 157, 61, 161, 21, 189, 188, 26, 236, 107, 61,
                    172, 237, 36, 61, 58, 191, 107, 62, 52, 242, 45, 60, 140, 21, 224, 189,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 16, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::IDENTITY, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal>;
        using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 16>;
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
        const TYPE module = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory_function(){return T_TYPE{weights::parameters, biases::parameters};}
    }
    namespace layer_3 {
        namespace weights_input {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    213, 49, 45, 63, 217, 210, 176, 190, 0, 201, 241, 189, 144, 195, 75, 190,
                    242, 16, 172, 190, 90, 148, 135, 63, 187, 83, 214, 189, 215, 120, 55, 189,
                    182, 243, 245, 190, 227, 44, 86, 63, 38, 131, 183, 63, 121, 103, 7, 63,
                    115, 214, 94, 190, 192, 130, 225, 190, 33, 104, 247, 62, 71, 154, 92, 62,
                    14, 211, 247, 190, 109, 28, 194, 191, 17, 62, 183, 62, 14, 122, 95, 63,
                    108, 106, 7, 62, 147, 90, 54, 189, 116, 252, 9, 63, 213, 206, 43, 190,
                    37, 14, 52, 63, 165, 153, 66, 63, 6, 15, 181, 62, 241, 246, 30, 191,
                    30, 109, 32, 191, 170, 25, 36, 62, 188, 95, 101, 60, 162, 96, 247, 189,
                    143, 213, 43, 190, 86, 233, 158, 63, 107, 6, 220, 190, 83, 1, 158, 63,
                    117, 181, 159, 189, 175, 247, 151, 63, 3, 204, 207, 190, 208, 94, 177, 191,
                    228, 119, 228, 190, 243, 181, 121, 191, 38, 125, 75, 62, 244, 214, 140, 63,
                    66, 126, 64, 63, 250, 166, 54, 63, 109, 73, 99, 61, 60, 177, 24, 191,
                    219, 208, 25, 191, 46, 105, 249, 191, 71, 110, 208, 61, 132, 3, 130, 190,
                    126, 171, 87, 61, 47, 9, 152, 189, 77, 213, 76, 63, 23, 3, 239, 190,
                    102, 158, 183, 190, 141, 114, 140, 191, 66, 113, 148, 191, 121, 4, 29, 191,
                    202, 129, 143, 63, 210, 196, 101, 191, 153, 90, 145, 62, 111, 196, 206, 190,
                    64, 117, 179, 63, 164, 14, 39, 63, 169, 241, 40, 191, 95, 139, 222, 189,
                    72, 250, 193, 62, 99, 156, 162, 62, 154, 161, 141, 190, 156, 183, 170, 191,
                    76, 214, 166, 189, 19, 127, 23, 63, 137, 48, 8, 61, 54, 117, 202, 190,
                    32, 33, 179, 62, 71, 114, 210, 190, 184, 253, 30, 63, 101, 60, 240, 191,
                    20, 176, 93, 190, 214, 58, 46, 64, 17, 29, 76, 63, 253, 23, 254, 191,
                    182, 249, 164, 190, 23, 54, 250, 189, 91, 94, 85, 190, 56, 7, 151, 63,
                    245, 224, 85, 190, 189, 111, 66, 191, 14, 13, 174, 190, 122, 135, 131, 189,
                    212, 204, 134, 62, 21, 219, 29, 191, 222, 186, 157, 63, 160, 176, 183, 61,
                    56, 245, 2, 191, 40, 241, 222, 190, 175, 36, 85, 191, 203, 216, 25, 190,
                    158, 36, 195, 63, 104, 168, 170, 190, 70, 39, 51, 189, 8, 59, 131, 62,
                    14, 39, 233, 62, 247, 51, 64, 189, 114, 219, 37, 191, 55, 167, 41, 191,
                    246, 43, 120, 63, 86, 18, 123, 191, 106, 111, 83, 63, 0, 161, 98, 62,
                    148, 79, 102, 191, 60, 18, 190, 189, 208, 215, 218, 61, 34, 250, 135, 62,
                    89, 217, 4, 63, 120, 145, 81, 62, 24, 19, 22, 62, 77, 107, 166, 191,
                    113, 249, 217, 190, 118, 126, 130, 190, 80, 66, 19, 63, 72, 205, 30, 63,
                    206, 188, 174, 191, 181, 252, 36, 191, 241, 147, 3, 63, 59, 134, 68, 191,
                    235, 194, 160, 190, 169, 168, 214, 190, 242, 82, 17, 63, 220, 146, 58, 191,
                    195, 200, 25, 64, 206, 121, 236, 191, 149, 88, 29, 192, 84, 200, 33, 63,
                    89, 197, 129, 190, 244, 223, 79, 191, 250, 203, 144, 191, 189, 176, 48, 191,
                    137, 226, 75, 63, 245, 205, 134, 62, 116, 9, 179, 63, 145, 161, 130, 62,
                    214, 104, 1, 191, 93, 204, 29, 191, 138, 229, 149, 189, 219, 228, 7, 63,
                    32, 189, 167, 191, 181, 210, 164, 63, 252, 127, 36, 63, 137, 144, 128, 191,
                    115, 236, 38, 63, 93, 199, 200, 191, 109, 230, 248, 60, 54, 97, 171, 190,
                    249, 185, 64, 63, 10, 30, 67, 63, 3, 203, 203, 191, 101, 56, 91, 191,
                    124, 157, 171, 62, 219, 66, 175, 63, 214, 145, 70, 63, 90, 39, 62, 191,
                    65, 151, 31, 191, 32, 70, 128, 190, 193, 33, 154, 62, 130, 242, 72, 63,
                    27, 192, 62, 63, 116, 145, 198, 62, 13, 190, 40, 190, 192, 216, 51, 191,
                    167, 42, 101, 63, 230, 159, 187, 63, 247, 138, 215, 60, 117, 192, 71, 191,
                    10, 71, 20, 191, 224, 90, 57, 191, 244, 61, 128, 189, 109, 238, 33, 63,
                    11, 112, 137, 60, 75, 143, 96, 63, 235, 165, 62, 190, 144, 181, 119, 189,
                    24, 210, 238, 190, 3, 40, 174, 191, 159, 124, 17, 191, 106, 186, 241, 61,
                    39, 224, 134, 62, 203, 121, 143, 62, 212, 186, 110, 191, 40, 165, 107, 62,
                    56, 62, 135, 191, 138, 8, 122, 63, 125, 23, 15, 191, 149, 94, 206, 190,
                    222, 132, 235, 190, 17, 37, 38, 60, 57, 107, 155, 190, 144, 161, 144, 63,
                    85, 134, 167, 62, 192, 193, 56, 191, 254, 254, 251, 190, 99, 242, 88, 63,
                    80, 188, 83, 190, 53, 39, 124, 191, 108, 207, 150, 63, 234, 118, 173, 60,
                    68, 68, 5, 63, 178, 79, 12, 62, 81, 101, 26, 191, 240, 193, 115, 62,
                    143, 75, 253, 189, 152, 127, 36, 61, 100, 203, 212, 60, 237, 110, 60, 63,
                    100, 94, 216, 190, 60, 181, 243, 61, 117, 155, 49, 61, 194, 134, 120, 62,
                    32, 47, 57, 190, 251, 95, 121, 190, 61, 16, 53, 62, 220, 218, 10, 62,
                    205, 169, 74, 63, 224, 53, 68, 190, 82, 165, 74, 190, 174, 54, 231, 60,
                    42, 194, 6, 63, 60, 163, 91, 191, 28, 161, 254, 190, 57, 199, 70, 190,
                    132, 182, 144, 63, 232, 20, 118, 63, 155, 117, 253, 62, 78, 123, 31, 191,
                    136, 122, 214, 190, 80, 173, 216, 189, 58, 64, 141, 189, 139, 251, 67, 63,
                    173, 152, 140, 63, 35, 160, 121, 191, 92, 220, 149, 191, 82, 58, 222, 62,
                    109, 163, 5, 63, 30, 110, 143, 63, 34, 10, 146, 62, 97, 48, 217, 189,
                    187, 42, 84, 191, 148, 120, 1, 191, 255, 137, 157, 190, 5, 100, 10, 191,
                    181, 98, 128, 62, 104, 76, 116, 191, 135, 183, 237, 191, 175, 74, 137, 62,
                    201, 150, 219, 189, 68, 45, 155, 190, 172, 183, 241, 189, 30, 199, 19, 63,
                    22, 1, 161, 61, 91, 176, 221, 61, 139, 133, 39, 63, 229, 24, 165, 189,
                    223, 24, 205, 190, 24, 13, 16, 63, 192, 14, 12, 63, 47, 31, 228, 61,
                    224, 209, 12, 63, 8, 171, 233, 62, 230, 86, 15, 63, 224, 172, 197, 60,
                    248, 41, 222, 190, 98, 146, 178, 190, 227, 80, 68, 63, 200, 112, 26, 63,
                    199, 193, 78, 62, 184, 69, 196, 62, 216, 212, 75, 61, 0, 90, 29, 62,
                    2, 19, 23, 63, 90, 21, 128, 190, 234, 53, 132, 61, 55, 70, 245, 190,
                    72, 4, 54, 62, 122, 112, 35, 191, 208, 133, 150, 190, 198, 170, 4, 60,
                    229, 203, 74, 190, 46, 2, 194, 62, 131, 8, 30, 62, 20, 17, 161, 189,
                    53, 106, 16, 63, 66, 108, 30, 63, 121, 171, 96, 190, 24, 134, 218, 190,
                    42, 68, 84, 191, 57, 213, 133, 190, 42, 221, 118, 60, 55, 92, 172, 62,
                    102, 129, 92, 189, 107, 50, 10, 63, 165, 190, 41, 63, 228, 126, 151, 190,
                    189, 217, 142, 62, 11, 50, 28, 190, 184, 127, 131, 189, 114, 11, 139, 190,
                    128, 235, 31, 62, 247, 125, 99, 190, 107, 48, 57, 190, 168, 11, 172, 190,
                    189, 143, 253, 61, 149, 71, 21, 190, 244, 166, 39, 189, 217, 56, 127, 61,
                    11, 32, 252, 190, 165, 64, 202, 190, 90, 53, 73, 190, 173, 71, 121, 62,
                    23, 37, 134, 63, 82, 42, 105, 191, 249, 46, 118, 191, 100, 123, 146, 63,
                    225, 169, 194, 62, 167, 18, 72, 62, 33, 176, 162, 62, 110, 238, 123, 191,
                    153, 216, 94, 189, 183, 23, 101, 63, 212, 116, 233, 61, 213, 14, 185, 190,
                    0, 32, 50, 189, 245, 4, 0, 191, 207, 8, 161, 189, 31, 133, 65, 63,
                    15, 37, 252, 62, 12, 97, 157, 63, 48, 206, 205, 62, 11, 144, 126, 63,
                    218, 100, 254, 190, 41, 210, 103, 62, 103, 42, 147, 60, 71, 26, 65, 62,
                    2, 52, 243, 61, 236, 2, 155, 62, 100, 199, 234, 62, 108, 155, 118, 191,
                    58, 188, 106, 191, 158, 34, 191, 62, 108, 183, 5, 191, 90, 70, 76, 191,
                    141, 45, 202, 62, 91, 30, 165, 62, 110, 72, 36, 188, 234, 62, 2, 63,
                    119, 8, 208, 62, 104, 105, 224, 190, 216, 44, 156, 189, 174, 187, 65, 191,
                    82, 211, 57, 191, 118, 81, 161, 189, 250, 40, 68, 62, 202, 119, 198, 190,
                    4, 23, 20, 191, 45, 142, 62, 190, 48, 90, 121, 62, 67, 214, 189, 190,
                    251, 137, 52, 191, 142, 102, 84, 63, 204, 83, 7, 63, 129, 109, 183, 63,
                    103, 106, 43, 191, 105, 101, 198, 62, 253, 131, 67, 63, 110, 252, 178, 61,
                    85, 7, 26, 191, 224, 169, 124, 62, 204, 247, 254, 62, 114, 67, 25, 191,
                    232, 79, 27, 63, 74, 95, 75, 63, 220, 107, 88, 63, 117, 252, 100, 191,
                    230, 252, 31, 63, 105, 69, 39, 63, 189, 231, 220, 62, 102, 251, 150, 190,
                    36, 209, 25, 62, 21, 157, 44, 63, 247, 170, 218, 190, 193, 2, 63, 191,
                    206, 101, 217, 62, 79, 165, 43, 63, 57, 24, 1, 64, 54, 26, 166, 190,
                    222, 240, 90, 190, 120, 146, 24, 62, 179, 34, 49, 189, 123, 165, 79, 191,
                    26, 34, 210, 62, 165, 36, 81, 190, 122, 58, 27, 190, 209, 69, 200, 61,
                    98, 241, 83, 63, 110, 185, 16, 62, 3, 19, 99, 191, 104, 84, 43, 62,
                    5, 36, 14, 63, 193, 27, 131, 190, 161, 219, 179, 189, 235, 240, 120, 62,
                    242, 67, 82, 60, 118, 235, 72, 190, 57, 52, 23, 62, 49, 105, 185, 189,
                    111, 217, 71, 191, 212, 193, 152, 63, 64, 213, 17, 63, 219, 117, 158, 63,
                    35, 7, 31, 191, 70, 220, 38, 191, 26, 90, 32, 189, 68, 14, 204, 189,
                    105, 163, 142, 190, 112, 254, 3, 191, 207, 248, 22, 190, 85, 109, 107, 189,
                    142, 98, 120, 190, 111, 223, 48, 61, 218, 204, 41, 189, 232, 38, 43, 62,
                    236, 22, 54, 61, 31, 171, 132, 62, 211, 213, 42, 63, 195, 38, 13, 63,
                    33, 161, 212, 190, 24, 222, 31, 190, 174, 74, 66, 190, 124, 47, 126, 61,
                    163, 108, 68, 191, 183, 182, 18, 62, 81, 220, 1, 63, 115, 194, 147, 62,
                    187, 5, 61, 62, 167, 162, 84, 62, 126, 71, 49, 63, 83, 7, 181, 189,
                    97, 115, 197, 60, 68, 103, 207, 63, 234, 215, 224, 190, 170, 178, 225, 62,
                    178, 128, 142, 62, 253, 250, 106, 62, 35, 134, 225, 62, 22, 92, 211, 61,
                    25, 195, 134, 191, 21, 152, 63, 191, 235, 176, 72, 191, 107, 119, 133, 62,
                    248, 45, 73, 62, 43, 57, 36, 63, 144, 175, 79, 191, 49, 25, 219, 62,
                    251, 63, 58, 191, 125, 172, 211, 62, 82, 133, 173, 190, 125, 2, 164, 62,
                    19, 235, 51, 191, 130, 19, 226, 189, 21, 31, 235, 62, 101, 168, 48, 62,
                    140, 96, 34, 63, 77, 237, 154, 189, 78, 42, 206, 190, 148, 147, 186, 62,
                    132, 53, 3, 63, 168, 232, 167, 189, 143, 247, 153, 62, 140, 48, 8, 190,
                    108, 1, 185, 62, 220, 139, 65, 63, 58, 27, 208, 62, 122, 227, 149, 189,
                    68, 109, 214, 190, 33, 254, 78, 190, 106, 220, 195, 62, 146, 88, 178, 189,
                    116, 50, 26, 63, 49, 167, 50, 63, 112, 172, 149, 190, 25, 55, 50, 190,
                    228, 16, 73, 63, 137, 51, 210, 62, 181, 161, 211, 190, 98, 17, 107, 191,
                    125, 165, 191, 61, 76, 15, 11, 63, 9, 109, 30, 191, 113, 148, 132, 189,
                    115, 144, 246, 62, 187, 134, 32, 191, 7, 1, 36, 61, 229, 80, 28, 64,
                    227, 216, 203, 191, 161, 153, 67, 62, 46, 251, 26, 191, 123, 40, 20, 63,
                    231, 138, 65, 191, 235, 126, 87, 63, 11, 182, 184, 61, 53, 46, 124, 190,
                    58, 15, 46, 190, 157, 86, 124, 62, 4, 152, 191, 190, 146, 209, 195, 188,
                    238, 149, 186, 189, 94, 116, 219, 189, 210, 91, 127, 190, 194, 0, 52, 63,
                    37, 202, 191, 189, 74, 242, 114, 62, 77, 180, 141, 62, 255, 190, 238, 62,
                    203, 60, 93, 189, 147, 135, 16, 62, 195, 121, 187, 62, 150, 136, 202, 62,
                    136, 213, 149, 62, 226, 49, 0, 190, 154, 214, 89, 189, 12, 197, 249, 62,
                    123, 213, 137, 62, 98, 104, 9, 61, 24, 232, 3, 189, 243, 152, 25, 191,
                    250, 199, 146, 189, 42, 108, 35, 63, 51, 77, 85, 62, 35, 64, 52, 190,
                    168, 62, 200, 61, 3, 119, 168, 62, 18, 188, 244, 62, 14, 195, 20, 190,
                    105, 203, 173, 60, 101, 179, 72, 63, 251, 253, 172, 190, 74, 213, 47, 188,
                    125, 197, 166, 61, 140, 178, 38, 62, 52, 239, 196, 190, 119, 116, 153, 189,
                    78, 184, 38, 191, 177, 71, 1, 191, 171, 34, 132, 190, 191, 59, 236, 62,
                    234, 29, 168, 62, 137, 55, 192, 189, 62, 221, 11, 63, 211, 217, 27, 63,
                    121, 122, 183, 61, 92, 49, 95, 62, 252, 81, 200, 62, 66, 84, 1, 191,
                    87, 206, 153, 63, 227, 42, 36, 62, 155, 70, 18, 63, 210, 224, 118, 63,
                    69, 251, 176, 189, 207, 149, 142, 190, 65, 23, 224, 190, 38, 198, 181, 190,
                    41, 5, 161, 190, 87, 100, 105, 62, 2, 133, 132, 190, 139, 211, 20, 191,
                    88, 217, 154, 62, 71, 117, 229, 190, 116, 72, 128, 190, 44, 9, 179, 57,
                    175, 177, 167, 189, 35, 74, 24, 189, 249, 29, 3, 62, 1, 132, 232, 61,
                    245, 165, 188, 190, 99, 79, 200, 62, 217, 111, 118, 62, 140, 164, 22, 191,
                    160, 215, 247, 189, 231, 130, 62, 190, 102, 177, 32, 190, 142, 186, 210, 62,
                    144, 157, 19, 190, 58, 97, 226, 61, 254, 18, 214, 62, 195, 212, 38, 62,
                    105, 224, 83, 59, 146, 117, 42, 59, 12, 120, 36, 60, 249, 26, 145, 61,
                    175, 121, 160, 61, 92, 31, 173, 62, 156, 8, 229, 62, 166, 44, 58, 62,
                    199, 227, 143, 190, 11, 193, 103, 190, 47, 200, 103, 190, 140, 99, 158, 190,
                    162, 131, 47, 60, 107, 82, 85, 190, 46, 197, 147, 188, 41, 101, 122, 190,
                    30, 132, 163, 62, 122, 78, 140, 190, 103, 23, 0, 63, 27, 161, 132, 62,
                    203, 219, 245, 190, 252, 22, 177, 62, 103, 207, 249, 60, 75, 34, 215, 62,
                    190, 60, 162, 190, 202, 140, 31, 62, 233, 168, 135, 61, 191, 162, 30, 62,
                    131, 4, 214, 189, 119, 244, 214, 190, 76, 54, 230, 62, 212, 193, 55, 190,
                    212, 251, 151, 61, 156, 110, 250, 60, 246, 234, 113, 190, 204, 214, 60, 190,
                    87, 201, 127, 189, 116, 185, 197, 189, 151, 216, 87, 62, 49, 144, 48, 190,
                    114, 125, 155, 189, 185, 8, 139, 62, 237, 88, 58, 60, 109, 160, 97, 62,
                    104, 58, 162, 62, 6, 229, 43, 63, 67, 67, 10, 189, 42, 66, 151, 189,
                    235, 6, 49, 189, 9, 188, 218, 62, 68, 21, 200, 60, 242, 247, 245, 61,
                    180, 70, 86, 62, 218, 234, 155, 61, 233, 2, 90, 188, 98, 120, 231, 62,
                    8, 178, 146, 186, 157, 241, 240, 62, 46, 255, 100, 62, 119, 32, 128, 190,
                    230, 65, 7, 190, 157, 128, 218, 62, 85, 135, 217, 61, 40, 17, 160, 61,
                    59, 227, 144, 62, 252, 2, 28, 189, 13, 58, 119, 63, 171, 141, 56, 191,
                    231, 27, 221, 61, 144, 58, 140, 62, 158, 93, 43, 189, 4, 75, 83, 63,
                    6, 21, 8, 191, 241, 114, 138, 62, 44, 227, 4, 189, 118, 225, 89, 191,
                    244, 188, 186, 189, 191, 148, 239, 189, 146, 208, 143, 62, 166, 204, 10, 62,
                    223, 84, 222, 62, 180, 246, 88, 190, 152, 198, 234, 189, 179, 178, 4, 190,
                    145, 189, 70, 190, 151, 53, 98, 61, 161, 101, 227, 61, 245, 215, 254, 61,
                    93, 145, 174, 189, 246, 94, 185, 62, 44, 43, 128, 189, 102, 205, 216, 59,
                    76, 225, 137, 62, 10, 51, 136, 62, 185, 212, 63, 61, 96, 19, 71, 189,
                    143, 32, 49, 62, 171, 125, 247, 62, 21, 34, 131, 187, 62, 55, 142, 62,
                    99, 137, 223, 189, 244, 222, 185, 190, 103, 132, 171, 190, 246, 73, 215, 190,
                    198, 139, 189, 189, 156, 166, 34, 60, 151, 219, 131, 61, 93, 225, 61, 190,
                    41, 15, 220, 189, 213, 23, 205, 61, 140, 229, 47, 191, 4, 154, 154, 62,
                    14, 193, 145, 62, 133, 151, 4, 190, 109, 80, 105, 62, 184, 203, 139, 61,
                    127, 139, 15, 61, 56, 213, 21, 62, 105, 155, 167, 190, 202, 5, 58, 62,
                    102, 205, 196, 189, 102, 163, 166, 190, 211, 245, 110, 188, 31, 222, 79, 190,
                    255, 93, 19, 63, 140, 213, 199, 190, 155, 191, 247, 190, 210, 29, 187, 189,
                    55, 109, 40, 189, 194, 116, 99, 190, 15, 94, 196, 62, 93, 174, 176, 189,
                    17, 45, 36, 188, 27, 167, 178, 62, 72, 57, 171, 189, 254, 206, 122, 190,
                    143, 9, 35, 189, 117, 207, 31, 62, 180, 145, 125, 190, 218, 154, 245, 189,
                    115, 23, 134, 61, 217, 5, 133, 190, 147, 95, 213, 189, 212, 18, 173, 189,
                    30, 164, 1, 191, 45, 57, 44, 61, 14, 98, 184, 62, 200, 16, 139, 60,
                    105, 44, 253, 189, 103, 176, 170, 61, 133, 83, 90, 61, 246, 188, 158, 62,
                    80, 69, 16, 190, 126, 119, 82, 62, 147, 64, 34, 61, 149, 6, 14, 63,
                    36, 49, 163, 62, 57, 126, 242, 62, 238, 134, 178, 62, 158, 76, 121, 62,
                    164, 93, 231, 61, 122, 105, 100, 62, 20, 210, 241, 61, 114, 173, 185, 62,
                    170, 238, 78, 190, 90, 174, 30, 62, 124, 105, 72, 62, 29, 176, 152, 190,
                    157, 84, 174, 190, 255, 51, 25, 189, 196, 237, 80, 62, 219, 45, 67, 190,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 48, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace weights_hidden {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    87, 167, 124, 191, 102, 167, 133, 62, 18, 42, 221, 190, 206, 60, 9, 62,
                    222, 55, 29, 191, 60, 30, 218, 190, 140, 199, 13, 63, 246, 202, 12, 63,
                    52, 120, 140, 63, 155, 133, 168, 191, 115, 231, 200, 62, 138, 218, 105, 63,
                    78, 36, 180, 190, 98, 208, 86, 62, 140, 231, 227, 191, 101, 214, 5, 191,
                    239, 0, 183, 191, 27, 208, 89, 63, 63, 26, 53, 191, 122, 196, 130, 63,
                    223, 66, 212, 62, 47, 84, 93, 190, 14, 128, 121, 63, 74, 207, 77, 63,
                    196, 87, 5, 191, 245, 141, 212, 63, 173, 62, 27, 191, 171, 230, 168, 62,
                    174, 129, 1, 63, 203, 202, 187, 61, 194, 50, 30, 190, 117, 122, 1, 191,
                    238, 43, 105, 191, 177, 13, 89, 63, 217, 63, 251, 62, 21, 93, 105, 191,
                    37, 17, 112, 191, 28, 215, 121, 62, 242, 114, 230, 190, 235, 137, 72, 190,
                    201, 33, 71, 62, 101, 5, 242, 63, 77, 237, 225, 63, 92, 18, 18, 63,
                    198, 153, 186, 63, 207, 223, 205, 62, 58, 174, 43, 62, 22, 232, 70, 62,
                    152, 27, 111, 189, 100, 80, 174, 63, 5, 196, 221, 61, 225, 109, 182, 63,
                    78, 1, 15, 191, 215, 255, 114, 191, 145, 178, 85, 190, 247, 208, 28, 191,
                    136, 7, 98, 62, 5, 103, 166, 63, 246, 185, 200, 62, 29, 30, 149, 62,
                    228, 147, 153, 61, 110, 5, 20, 192, 102, 61, 14, 63, 78, 31, 68, 191,
                    58, 14, 120, 63, 110, 195, 148, 191, 130, 182, 84, 63, 155, 41, 162, 191,
                    84, 10, 146, 191, 139, 27, 131, 63, 63, 170, 176, 62, 191, 142, 62, 63,
                    165, 54, 195, 63, 230, 18, 122, 191, 123, 161, 191, 191, 160, 171, 236, 191,
                    234, 80, 63, 62, 231, 116, 96, 63, 66, 5, 195, 62, 74, 128, 203, 191,
                    70, 51, 136, 191, 116, 23, 175, 62, 249, 37, 94, 191, 185, 165, 218, 63,
                    133, 84, 179, 62, 193, 206, 160, 190, 136, 72, 245, 190, 183, 119, 54, 190,
                    126, 25, 182, 191, 18, 121, 2, 191, 246, 85, 179, 190, 58, 34, 155, 190,
                    59, 176, 152, 63, 204, 124, 6, 62, 13, 186, 169, 191, 123, 230, 162, 189,
                    225, 130, 9, 191, 184, 242, 216, 61, 209, 97, 172, 62, 198, 198, 204, 63,
                    237, 133, 16, 190, 149, 72, 15, 63, 66, 204, 112, 190, 159, 158, 200, 62,
                    206, 133, 142, 62, 135, 143, 233, 191, 235, 217, 184, 63, 56, 55, 113, 191,
                    53, 33, 45, 63, 205, 82, 180, 62, 91, 125, 28, 63, 180, 33, 8, 191,
                    138, 62, 11, 63, 21, 173, 88, 191, 225, 43, 169, 189, 41, 0, 197, 191,
                    133, 131, 248, 189, 240, 2, 177, 191, 96, 237, 242, 190, 173, 131, 159, 62,
                    211, 136, 106, 62, 182, 78, 165, 190, 119, 39, 87, 190, 133, 197, 228, 62,
                    28, 78, 83, 63, 30, 38, 133, 190, 116, 28, 96, 191, 202, 237, 111, 191,
                    157, 15, 240, 190, 111, 227, 116, 63, 63, 60, 116, 63, 165, 167, 148, 61,
                    81, 105, 84, 63, 122, 196, 148, 190, 60, 238, 159, 63, 122, 54, 133, 62,
                    153, 186, 172, 63, 19, 34, 47, 191, 122, 53, 0, 190, 172, 49, 68, 63,
                    236, 181, 200, 61, 105, 168, 182, 62, 73, 48, 41, 191, 188, 243, 50, 62,
                    85, 110, 40, 191, 78, 91, 68, 191, 132, 225, 155, 190, 147, 245, 38, 192,
                    71, 246, 9, 63, 5, 177, 80, 190, 119, 57, 132, 63, 118, 32, 62, 62,
                    89, 102, 104, 190, 7, 83, 154, 62, 175, 64, 4, 190, 146, 145, 160, 63,
                    253, 139, 244, 190, 216, 166, 137, 63, 71, 161, 160, 62, 193, 67, 23, 192,
                    63, 99, 139, 191, 203, 40, 195, 191, 201, 70, 36, 63, 254, 245, 44, 63,
                    59, 64, 50, 190, 99, 222, 73, 191, 37, 20, 82, 63, 120, 183, 130, 63,
                    178, 219, 169, 191, 52, 0, 37, 63, 62, 118, 58, 191, 220, 218, 68, 188,
                    255, 50, 171, 63, 149, 67, 204, 191, 147, 147, 212, 191, 21, 36, 9, 189,
                    153, 173, 90, 63, 122, 99, 56, 191, 196, 42, 222, 62, 101, 18, 87, 190,
                    124, 131, 193, 63, 128, 2, 166, 62, 152, 34, 218, 191, 174, 36, 29, 63,
                    171, 228, 80, 62, 49, 170, 141, 191, 251, 135, 183, 190, 156, 109, 230, 191,
                    75, 118, 225, 62, 71, 237, 35, 188, 245, 154, 34, 64, 113, 44, 60, 191,
                    120, 11, 254, 188, 221, 10, 63, 63, 228, 173, 137, 63, 28, 186, 158, 191,
                    120, 69, 216, 61, 227, 18, 145, 191, 233, 200, 229, 190, 64, 94, 115, 191,
                    65, 30, 205, 191, 21, 115, 17, 63, 132, 18, 184, 62, 110, 21, 4, 62,
                    124, 225, 72, 61, 129, 232, 195, 190, 7, 156, 214, 191, 9, 35, 25, 63,
                    168, 21, 198, 190, 247, 207, 102, 191, 23, 50, 15, 63, 214, 17, 165, 191,
                    190, 0, 123, 190, 115, 142, 77, 63, 67, 178, 37, 62, 195, 226, 175, 190,
                    117, 202, 222, 190, 221, 193, 136, 191, 152, 16, 117, 62, 199, 140, 167, 62,
                    22, 84, 203, 62, 194, 4, 70, 191, 186, 253, 103, 191, 189, 5, 64, 63,
                    220, 204, 124, 63, 56, 56, 240, 191, 62, 52, 154, 63, 193, 6, 28, 63,
                    116, 176, 204, 62, 251, 104, 46, 192, 211, 216, 205, 63, 55, 142, 155, 63,
                    229, 114, 29, 191, 219, 68, 187, 63, 108, 7, 51, 62, 1, 216, 64, 63,
                    149, 151, 95, 63, 95, 253, 226, 191, 12, 101, 37, 63, 98, 129, 35, 192,
                    125, 156, 221, 191, 209, 145, 38, 63, 6, 28, 9, 63, 150, 253, 61, 191,
                    193, 30, 1, 61, 161, 188, 105, 62, 93, 9, 190, 187, 145, 164, 235, 62,
                    48, 210, 33, 191, 173, 132, 28, 61, 123, 91, 15, 64, 225, 87, 133, 191,
                    116, 237, 222, 62, 66, 3, 174, 191, 68, 43, 175, 190, 147, 122, 202, 190,
                    238, 213, 198, 60, 90, 252, 33, 190, 209, 193, 124, 63, 159, 33, 27, 190,
                    221, 28, 20, 191, 81, 216, 218, 62, 52, 39, 42, 61, 13, 179, 100, 62,
                    185, 178, 88, 190, 15, 106, 96, 63, 227, 132, 174, 188, 153, 176, 192, 186,
                    124, 92, 31, 191, 43, 93, 141, 190, 126, 123, 1, 191, 249, 38, 152, 62,
                    3, 240, 28, 191, 244, 254, 26, 62, 196, 177, 147, 189, 53, 252, 163, 62,
                    217, 87, 184, 63, 7, 214, 166, 189, 105, 214, 99, 63, 67, 35, 243, 62,
                    187, 226, 244, 190, 63, 254, 58, 63, 155, 116, 75, 190, 141, 130, 30, 191,
                    175, 1, 252, 190, 121, 99, 136, 188, 123, 243, 56, 63, 250, 56, 98, 63,
                    26, 99, 22, 191, 119, 166, 219, 190, 188, 171, 21, 190, 62, 149, 124, 63,
                    189, 224, 210, 61, 48, 180, 241, 190, 179, 65, 94, 62, 19, 26, 97, 190,
                    59, 68, 72, 63, 106, 60, 170, 191, 135, 88, 60, 191, 34, 135, 255, 189,
                    165, 158, 58, 190, 65, 214, 198, 62, 42, 190, 22, 62, 242, 139, 164, 190,
                    196, 86, 143, 60, 118, 226, 135, 189, 249, 70, 200, 63, 58, 29, 234, 60,
                    48, 171, 132, 59, 231, 162, 49, 63, 62, 176, 26, 62, 6, 115, 76, 61,
                    13, 82, 244, 61, 187, 235, 137, 189, 67, 22, 220, 190, 150, 2, 31, 63,
                    143, 106, 159, 190, 154, 67, 80, 190, 165, 7, 58, 63, 10, 74, 127, 61,
                    154, 185, 142, 190, 171, 138, 49, 63, 150, 213, 122, 63, 75, 167, 221, 191,
                    142, 6, 31, 62, 198, 20, 217, 63, 33, 232, 24, 191, 195, 89, 187, 190,
                    115, 44, 220, 191, 23, 29, 206, 63, 151, 185, 141, 191, 73, 28, 219, 189,
                    103, 201, 170, 191, 182, 120, 80, 191, 64, 195, 30, 192, 228, 198, 109, 63,
                    205, 206, 192, 191, 84, 131, 27, 64, 171, 78, 58, 191, 227, 222, 18, 192,
                    4, 253, 213, 190, 144, 190, 104, 190, 213, 86, 200, 63, 160, 125, 226, 190,
                    149, 177, 117, 190, 177, 222, 7, 63, 20, 156, 245, 62, 36, 103, 27, 192,
                    117, 239, 109, 190, 7, 139, 243, 190, 55, 64, 126, 189, 55, 218, 248, 62,
                    53, 39, 64, 62, 234, 61, 136, 191, 165, 252, 160, 63, 233, 215, 50, 191,
                    95, 175, 27, 191, 2, 59, 180, 191, 146, 56, 47, 62, 226, 101, 156, 62,
                    50, 70, 138, 191, 55, 30, 182, 62, 169, 238, 85, 63, 137, 37, 60, 63,
                    69, 166, 170, 63, 106, 109, 46, 63, 184, 42, 183, 62, 51, 228, 76, 191,
                    131, 153, 163, 190, 200, 157, 160, 191, 132, 21, 9, 189, 225, 10, 185, 191,
                    34, 30, 160, 63, 8, 192, 80, 192, 163, 6, 39, 63, 85, 6, 232, 63,
                    50, 99, 190, 190, 100, 168, 37, 63, 207, 107, 151, 63, 134, 212, 69, 62,
                    155, 34, 95, 63, 37, 126, 202, 190, 130, 159, 173, 191, 112, 40, 55, 192,
                    67, 115, 8, 62, 126, 172, 173, 62, 9, 149, 30, 63, 65, 181, 77, 192,
                    231, 103, 239, 62, 173, 37, 232, 189, 4, 186, 130, 63, 66, 221, 227, 190,
                    43, 45, 154, 63, 227, 161, 160, 191, 52, 27, 160, 63, 84, 34, 92, 191,
                    148, 36, 102, 62, 119, 198, 155, 190, 233, 85, 237, 63, 73, 202, 32, 63,
                    158, 22, 24, 62, 125, 84, 55, 63, 40, 120, 138, 63, 60, 113, 213, 62,
                    123, 232, 151, 190, 163, 83, 62, 63, 172, 80, 191, 61, 134, 5, 25, 63,
                    40, 235, 101, 63, 28, 131, 108, 190, 131, 48, 235, 190, 229, 46, 8, 191,
                    170, 187, 27, 191, 84, 181, 204, 61, 208, 99, 222, 62, 210, 74, 203, 190,
                    166, 150, 131, 62, 255, 125, 149, 63, 224, 184, 178, 63, 207, 188, 132, 189,
                    222, 104, 93, 61, 57, 35, 248, 189, 236, 166, 166, 190, 129, 144, 80, 191,
                    107, 140, 139, 63, 192, 187, 227, 63, 248, 192, 189, 63, 131, 102, 194, 191,
                    101, 206, 7, 191, 243, 20, 50, 63, 190, 213, 196, 191, 112, 123, 180, 63,
                    177, 30, 90, 191, 140, 144, 114, 63, 186, 14, 244, 63, 79, 109, 127, 62,
                    64, 71, 3, 64, 77, 193, 167, 191, 62, 155, 129, 63, 128, 247, 84, 191,
                    172, 92, 93, 188, 203, 214, 89, 64, 232, 69, 88, 63, 146, 125, 40, 191,
                    249, 60, 163, 190, 152, 237, 67, 63, 255, 101, 96, 191, 202, 160, 21, 192,
                    198, 40, 164, 191, 193, 175, 185, 63, 28, 31, 95, 63, 218, 0, 42, 64,
                    72, 28, 133, 61, 109, 122, 92, 190, 12, 247, 19, 191, 175, 245, 174, 63,
                    137, 204, 49, 191, 46, 39, 21, 63, 230, 110, 155, 63, 196, 220, 159, 63,
                    21, 79, 77, 62, 225, 27, 189, 191, 159, 120, 143, 61, 110, 140, 202, 62,
                    122, 114, 145, 62, 171, 156, 225, 62, 200, 16, 131, 63, 227, 14, 7, 191,
                    25, 236, 144, 62, 185, 211, 255, 61, 118, 215, 10, 63, 190, 31, 54, 62,
                    225, 184, 126, 62, 195, 125, 70, 63, 251, 176, 31, 190, 72, 86, 179, 62,
                    160, 193, 112, 190, 227, 132, 231, 61, 101, 214, 212, 188, 110, 185, 1, 63,
                    187, 87, 57, 191, 177, 14, 176, 191, 106, 21, 166, 191, 220, 165, 58, 192,
                    146, 92, 188, 63, 158, 179, 128, 191, 77, 68, 210, 190, 201, 101, 4, 191,
                    247, 239, 60, 191, 84, 152, 207, 188, 108, 198, 238, 62, 39, 166, 20, 192,
                    195, 173, 81, 191, 82, 30, 130, 189, 233, 28, 110, 63, 190, 99, 173, 189,
                    51, 112, 70, 191, 117, 14, 171, 63, 171, 60, 114, 63, 93, 115, 185, 63,
                    93, 50, 11, 63, 25, 22, 101, 190, 18, 245, 31, 192, 246, 103, 242, 63,
                    212, 6, 139, 191, 118, 155, 49, 190, 165, 42, 181, 190, 60, 58, 204, 63,
                    108, 62, 65, 191, 31, 249, 12, 191, 63, 233, 222, 190, 3, 153, 44, 63,
                    207, 30, 128, 190, 76, 139, 113, 190, 43, 251, 103, 191, 80, 240, 17, 191,
                    205, 216, 14, 63, 99, 174, 141, 191, 162, 210, 146, 62, 58, 88, 112, 189,
                    51, 171, 46, 189, 109, 239, 139, 63, 76, 29, 164, 62, 139, 204, 161, 63,
                    85, 186, 196, 190, 101, 156, 201, 190, 166, 254, 248, 190, 37, 14, 18, 190,
                    183, 173, 21, 62, 4, 148, 155, 191, 157, 113, 140, 63, 175, 68, 51, 63,
                    196, 202, 190, 190, 22, 53, 87, 63, 17, 134, 242, 63, 26, 183, 233, 62,
                    209, 193, 237, 62, 107, 183, 72, 189, 127, 106, 11, 190, 201, 230, 97, 62,
                    73, 166, 104, 190, 142, 34, 152, 190, 79, 114, 115, 190, 76, 167, 174, 190,
                    102, 7, 163, 63, 102, 208, 90, 191, 80, 75, 186, 191, 42, 112, 76, 191,
                    11, 212, 100, 191, 3, 119, 204, 190, 248, 108, 138, 190, 221, 178, 72, 191,
                    212, 82, 161, 61, 215, 76, 42, 190, 153, 127, 198, 61, 123, 127, 147, 62,
                    129, 159, 250, 60, 57, 231, 192, 60, 108, 130, 173, 62, 241, 244, 1, 61,
                    6, 200, 36, 63, 137, 43, 230, 190, 93, 185, 220, 62, 192, 10, 252, 191,
                    24, 92, 189, 188, 190, 45, 38, 191, 50, 70, 63, 63, 117, 222, 208, 190,
                    185, 117, 65, 62, 127, 75, 156, 190, 78, 18, 17, 63, 234, 147, 2, 60,
                    209, 184, 20, 63, 57, 6, 171, 63, 119, 128, 133, 188, 130, 185, 15, 62,
                    31, 107, 79, 191, 79, 81, 220, 62, 124, 216, 137, 62, 120, 17, 199, 191,
                    45, 131, 2, 191, 205, 86, 99, 62, 242, 85, 255, 189, 179, 99, 194, 190,
                    164, 123, 180, 189, 126, 144, 72, 63, 196, 90, 10, 63, 169, 18, 212, 61,
                    236, 43, 51, 61, 117, 52, 84, 63, 76, 219, 162, 190, 95, 91, 62, 190,
                    210, 131, 165, 190, 54, 153, 24, 189, 146, 203, 91, 63, 5, 218, 219, 189,
                    246, 89, 203, 189, 37, 22, 201, 190, 235, 65, 196, 61, 60, 89, 21, 62,
                    61, 158, 245, 187, 0, 78, 167, 190, 110, 96, 133, 62, 157, 164, 10, 191,
                    225, 230, 63, 61, 152, 19, 91, 191, 56, 72, 195, 61, 143, 33, 216, 62,
                    80, 0, 95, 62, 144, 209, 214, 59, 112, 88, 207, 62, 248, 10, 62, 191,
                    94, 170, 170, 190, 34, 208, 71, 191, 13, 173, 246, 190, 44, 210, 87, 61,
                    193, 208, 166, 190, 62, 145, 235, 61, 121, 218, 201, 190, 69, 148, 53, 63,
                    33, 53, 27, 61, 222, 226, 49, 190, 76, 36, 165, 191, 142, 206, 67, 62,
                    114, 244, 169, 190, 175, 189, 42, 190, 60, 75, 97, 190, 133, 163, 225, 189,
                    77, 175, 85, 63, 196, 108, 252, 62, 241, 253, 179, 189, 212, 83, 55, 190,
                    82, 122, 210, 61, 7, 208, 162, 190, 98, 249, 219, 191, 228, 98, 218, 190,
                    26, 106, 50, 63, 78, 28, 201, 61, 49, 41, 12, 191, 6, 81, 15, 191,
                    207, 99, 111, 189, 215, 172, 166, 62, 213, 89, 130, 63, 100, 66, 39, 191,
                    105, 202, 130, 62, 224, 159, 28, 61, 34, 71, 224, 62, 43, 26, 198, 189,
                    110, 188, 244, 190, 33, 106, 153, 190, 250, 179, 141, 61, 26, 151, 121, 189,
                    71, 23, 41, 63, 57, 191, 163, 190, 50, 133, 130, 191, 75, 217, 160, 190,
                    167, 117, 129, 60, 155, 157, 108, 62, 152, 119, 3, 191, 247, 120, 147, 190,
                    157, 76, 81, 191, 135, 166, 184, 189, 218, 21, 93, 191, 172, 74, 33, 191,
                    99, 252, 136, 62, 187, 246, 192, 191, 129, 198, 33, 191, 45, 44, 114, 63,
                    43, 97, 66, 190, 135, 179, 132, 63, 168, 207, 129, 190, 77, 83, 179, 63,
                    211, 3, 93, 191, 92, 61, 172, 191, 98, 51, 100, 190, 25, 250, 170, 61,
                    8, 21, 65, 190, 149, 224, 209, 62, 87, 122, 18, 62, 176, 186, 192, 63,
                    140, 187, 82, 189, 80, 70, 89, 191, 88, 224, 30, 191, 225, 211, 199, 62,
                    201, 164, 203, 62, 235, 48, 144, 60, 23, 25, 38, 190, 153, 47, 20, 190,
                    135, 179, 56, 191, 178, 85, 17, 63, 188, 149, 119, 191, 3, 204, 25, 190,
                    16, 247, 26, 62, 73, 117, 225, 62, 41, 83, 172, 60, 18, 96, 227, 189,
                    89, 185, 10, 188, 102, 107, 48, 189, 252, 178, 96, 62, 136, 19, 43, 62,
                    194, 152, 93, 190, 246, 239, 42, 61, 63, 128, 131, 189, 140, 167, 29, 190,
                    41, 149, 232, 63, 182, 199, 197, 63, 93, 70, 160, 190, 112, 52, 182, 62,
                    170, 231, 187, 190, 30, 36, 42, 191, 254, 159, 112, 63, 143, 127, 43, 62,
                    165, 218, 170, 62, 225, 143, 195, 61, 206, 228, 176, 63, 138, 108, 109, 191,
                    128, 21, 156, 191, 36, 133, 188, 189, 104, 73, 213, 62, 154, 220, 51, 63,
                    6, 6, 91, 189, 181, 18, 32, 61, 123, 3, 207, 62, 211, 131, 132, 190,
                    251, 242, 224, 190, 209, 233, 18, 63, 94, 169, 167, 62, 10, 204, 20, 191,
                    193, 82, 100, 189, 141, 193, 7, 191, 208, 209, 110, 63, 72, 50, 110, 189,
                    138, 52, 181, 190, 65, 202, 60, 191, 223, 159, 252, 190, 0, 97, 126, 62,
                    208, 219, 56, 63, 252, 185, 251, 60, 232, 10, 193, 62, 10, 84, 14, 63,
                    100, 187, 127, 62, 71, 79, 201, 189, 191, 55, 154, 61, 59, 127, 251, 62,
                    37, 70, 230, 62, 107, 173, 93, 63, 95, 48, 178, 190, 93, 227, 1, 191,
                    202, 15, 218, 189, 86, 160, 155, 189, 200, 6, 216, 190, 31, 177, 88, 62,
                    65, 46, 160, 190, 66, 80, 224, 62, 208, 174, 8, 189, 31, 148, 180, 191,
                    8, 105, 16, 62, 190, 7, 43, 63, 173, 36, 88, 63, 232, 225, 255, 60,
                    166, 75, 147, 190, 185, 41, 253, 190, 216, 39, 102, 63, 61, 83, 232, 190,
                    163, 145, 248, 61, 96, 81, 246, 62, 171, 168, 170, 62, 236, 180, 27, 191,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 48, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace biases_input {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    178, 0, 167, 63, 206, 125, 16, 190, 205, 14, 98, 58, 169, 110, 68, 62,
                    34, 188, 26, 63, 233, 109, 169, 191, 179, 67, 154, 63, 80, 87, 19, 63,
                    108, 48, 57, 189, 144, 95, 146, 63, 105, 47, 153, 191, 32, 28, 236, 63,
                    108, 233, 198, 191, 154, 209, 14, 63, 147, 49, 191, 61, 10, 20, 246, 190,
                    235, 185, 17, 190, 127, 54, 108, 190, 163, 220, 152, 63, 247, 193, 15, 64,
                    58, 247, 4, 192, 108, 216, 212, 191, 52, 57, 148, 191, 31, 86, 217, 191,
                    72, 42, 152, 191, 158, 159, 22, 64, 224, 92, 199, 191, 205, 152, 154, 191,
                    25, 254, 155, 191, 141, 184, 94, 190, 93, 126, 22, 191, 64, 19, 194, 62,
                    227, 245, 200, 61, 145, 6, 234, 188, 112, 194, 77, 190, 17, 135, 50, 187,
                    114, 23, 87, 191, 43, 26, 20, 61, 224, 37, 83, 62, 243, 245, 99, 62,
                    66, 68, 164, 63, 195, 192, 24, 191, 158, 238, 24, 61, 233, 216, 10, 62,
                    213, 53, 163, 190, 25, 204, 53, 190, 191, 1, 48, 62, 8, 113, 170, 60,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 48>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace biases_hidden {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    71, 157, 209, 63, 160, 233, 51, 60, 200, 242, 22, 62, 54, 180, 241, 62,
                    57, 236, 127, 63, 58, 1, 205, 191, 253, 249, 161, 63, 84, 194, 23, 62,
                    207, 231, 132, 189, 35, 54, 97, 63, 90, 79, 104, 191, 74, 230, 223, 63,
                    113, 68, 217, 191, 30, 127, 45, 63, 17, 165, 57, 61, 4, 180, 49, 191,
                    82, 39, 254, 60, 116, 157, 83, 61, 139, 22, 141, 63, 127, 205, 18, 64,
                    121, 67, 12, 192, 221, 75, 207, 191, 234, 175, 119, 191, 166, 197, 228, 191,
                    35, 170, 137, 191, 92, 179, 26, 64, 36, 74, 228, 191, 17, 67, 147, 191,
                    88, 23, 185, 191, 11, 133, 23, 191, 73, 84, 89, 191, 255, 95, 71, 62,
                    188, 199, 160, 62, 226, 206, 106, 189, 232, 136, 124, 190, 129, 98, 233, 190,
                    169, 143, 0, 191, 130, 220, 225, 62, 118, 22, 137, 190, 254, 53, 222, 62,
                    205, 244, 107, 63, 142, 152, 235, 190, 64, 61, 23, 63, 103, 234, 80, 62,
                    43, 13, 17, 191, 152, 188, 176, 62, 26, 150, 233, 61, 74, 101, 112, 63,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 48>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace initial_hidden_state {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::gru::Configuration<TYPE_POLICY, unsigned long, 16, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, true>;
        using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::gru::BindConfiguration<CONFIG>;
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 16>;
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::gru::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
        const TYPE module = {weights_input::parameters, biases_input::parameters, weights_hidden::parameters, biases_hidden::parameters, initial_hidden_state::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory = {weights_input::parameters, biases_input::parameters, weights_hidden::parameters, biases_hidden::parameters, initial_hidden_state::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory_function(){return {weights_input::parameters, biases_input::parameters, weights_hidden::parameters, biases_hidden::parameters, initial_hidden_state::parameters};}
    }
    namespace layer_4 {
        namespace weights {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    228, 246, 154, 189, 19, 167, 168, 189, 33, 59, 22, 190, 100, 228, 52, 63,
                    74, 63, 12, 190, 15, 0, 238, 189, 246, 221, 184, 190, 159, 97, 33, 62,
                    176, 69, 203, 62, 72, 74, 212, 190, 92, 205, 109, 189, 130, 197, 152, 62,
                    249, 165, 234, 190, 155, 14, 133, 61, 14, 36, 171, 61, 222, 216, 185, 62,
                    24, 97, 69, 62, 62, 73, 232, 189, 118, 172, 208, 190, 107, 118, 1, 191,
                    2, 226, 201, 188, 12, 62, 28, 62, 204, 120, 138, 62, 255, 57, 255, 189,
                    129, 192, 235, 62, 124, 30, 5, 62, 0, 192, 25, 190, 137, 174, 144, 189,
                    233, 62, 85, 190, 84, 45, 186, 62, 42, 46, 15, 63, 43, 44, 106, 61,
                    31, 79, 47, 190, 138, 85, 231, 61, 216, 30, 117, 190, 147, 231, 161, 62,
                    67, 108, 202, 190, 136, 131, 137, 189, 41, 35, 85, 62, 87, 62, 101, 62,
                    143, 63, 148, 188, 188, 158, 17, 63, 12, 158, 53, 63, 5, 130, 48, 189,
                    10, 64, 141, 189, 125, 55, 118, 189, 218, 178, 97, 62, 168, 222, 106, 190,
                    2, 81, 4, 190, 226, 134, 48, 62, 30, 173, 81, 61, 207, 114, 23, 191,
                    48, 198, 224, 189, 135, 129, 16, 191, 183, 255, 80, 190, 181, 192, 41, 62,
                    64, 221, 159, 62, 11, 61, 127, 190, 129, 191, 56, 62, 166, 193, 240, 61,
                    48, 131, 21, 61, 65, 195, 40, 63, 149, 210, 27, 62, 67, 229, 41, 189,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 4, 16>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        namespace biases {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    234, 72, 120, 62, 113, 73, 168, 62, 149, 127, 225, 62, 190, 191, 190, 62,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 4>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 4, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::IDENTITY, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal>;
        using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 16>;
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
        const TYPE module = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory = {weights::parameters, biases::parameters};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory_function(){return T_TYPE{weights::parameters, biases::parameters};}
    }
namespace model_definition {
    using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
    template <typename T_CONTENT, typename T_NEXT_MODULE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::OutputModule>
    using Module = typename RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
    using MODULE_CHAIN = Module<layer_0::TEMPLATE, Module<layer_1::TEMPLATE, Module<layer_2::TEMPLATE, Module<layer_3::TEMPLATE, Module<layer_4::TEMPLATE>>>>>;
    using MODEL = typename RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, layer_0::INPUT_SHAPE>;
}
using TYPE = model_definition::MODEL;
const TYPE module = {layer_0::factory<TYPE::CONTENT>, {layer_1::factory<TYPE::NEXT_MODULE::CONTENT>, {layer_2::factory<TYPE::NEXT_MODULE::NEXT_MODULE::CONTENT>, {layer_3::factory<TYPE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::CONTENT>, {layer_4::factory<TYPE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::CONTENT>, {}}}}}};
template <typename T_TYPE = TYPE>
const T_TYPE factory = {layer_0::factory<typename T_TYPE::CONTENT>, {layer_1::factory<typename T_TYPE::NEXT_MODULE::CONTENT>, {layer_2::factory<typename T_TYPE::NEXT_MODULE::NEXT_MODULE::CONTENT>, {layer_3::factory<typename T_TYPE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::CONTENT>, {layer_4::factory<typename T_TYPE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::CONTENT>, {}}}}}};
template <typename T_TYPE = TYPE>
const T_TYPE factory_function(){return T_TYPE{layer_0::factory_function<typename T_TYPE::CONTENT>(), {layer_1::factory_function<typename T_TYPE::NEXT_MODULE::CONTENT>(), {layer_2::factory_function<typename T_TYPE::NEXT_MODULE::NEXT_MODULE::CONTENT>(), {layer_3::factory_function<typename T_TYPE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::CONTENT>(), {layer_4::factory_function<typename T_TYPE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::NEXT_MODULE::CONTENT>(), {}}}}}};}
}
namespace rl_tools::checkpoint::example::input {
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 35>;
    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
    const CONTAINER_TYPE container = {(const float*)memory};
}
namespace rl_tools::checkpoint::example::output {
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 4>;
    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
    const CONTAINER_TYPE container = {(const float*)memory};
}
namespace rl_tools::checkpoint::meta{
   inline char name[] = "acesim_export";
   inline char commit_hash[] = "unknown";
}
