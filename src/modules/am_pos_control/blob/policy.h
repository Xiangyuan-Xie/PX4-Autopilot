#pragma once
#include <cmath>
#include <rl_tools/containers/matrix/matrix.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn/parameters/parameters.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/nn/layers/gru/layer.h>
#include <rl_tools/nn_models/sequential/model.h>

namespace rl_tools::checkpoint::beta_mean {
    struct State {};
    struct Buffer {};

    template <typename T_TYPE_POLICY, typename T_TI, T_TI T_OUTPUT_DIM>
    struct Configuration {
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI OUTPUT_DIM = T_OUTPUT_DIM;
    };

    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification: T_CAPABILITY, T_CONFIG {
        using CONFIG = T_CONFIG;
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        static constexpr TI INPUT_DIM = get_last(INPUT_SHAPE{});
        static constexpr TI OUTPUT_DIM = CONFIG::OUTPUT_DIM;
        static_assert(INPUT_DIM == 2 * OUTPUT_DIM);

        template <typename NEW_INPUT_SHAPE>
        struct OUTPUT_SHAPE_FACTORY {
            static constexpr TI NEW_INPUT_DIM = get_last(NEW_INPUT_SHAPE{});
            static_assert(NEW_INPUT_DIM == INPUT_DIM);
            using SHAPE = tensor::Replace<NEW_INPUT_SHAPE, OUTPUT_DIM, length(NEW_INPUT_SHAPE{}) - 1>;
        };

        using OUTPUT_SHAPE = typename OUTPUT_SHAPE_FACTORY<INPUT_SHAPE>::SHAPE;
        static constexpr TI INTERNAL_BATCH_SIZE = get<0>(tensor::CumulativeProduct<tensor::PopBack<INPUT_SHAPE>>{});
        static constexpr TI NUM_WEIGHTS = 0;
    };

    template <typename T_SPEC>
    struct LayerForward {
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        static constexpr TI INTERNAL_BATCH_SIZE = SPEC::INTERNAL_BATCH_SIZE;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        template <typename NEW_INPUT_SHAPE>
        using OUTPUT_SHAPE_FACTORY = typename SPEC::template OUTPUT_SHAPE_FACTORY<NEW_INPUT_SHAPE>::SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;
        template <bool DYNAMIC_ALLOCATION = true>
        using Buffer = beta_mean::Buffer;
        template <bool DYNAMIC_ALLOCATION = true>
        using State = beta_mean::State;
    };

    template <typename CONFIG>
    struct BindConfiguration {
        template <typename CAPABILITY, typename INPUT_SHAPE>
        using Layer = beta_mean::LayerForward<beta_mean::Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>;
    };
}

namespace rl_tools {
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, checkpoint::beta_mean::State&) {}

    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, checkpoint::beta_mean::State&) {}

    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, checkpoint::beta_mean::Buffer&) {}

    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, checkpoint::beta_mean::Buffer&) {}

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, checkpoint::beta_mean::LayerForward<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, checkpoint::beta_mean::LayerForward<SPEC>&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE&, checkpoint::beta_mean::LayerForward<SPEC>&, RNG&) {}

    template <typename DEVICE, typename SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(
        DEVICE&,
        const checkpoint::beta_mean::LayerForward<SPEC>&,
        checkpoint::beta_mean::State&,
        RNG&,
        const Mode<MODE>& = Mode<mode::Default<>>{}
    ) {}

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T checkpoint_beta_mean_softplus(T x) {
        return x > (T)0 ? x + std::log((T)1 + std::exp(-x)) : std::log((T)1 + std::exp(x));
    }

    template <
        typename DEVICE,
        typename LAYER_SPEC,
        typename INPUT_SPEC,
        typename OUTPUT_SPEC,
        typename RNG,
        typename MODE = mode::Default<>
    >
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(
        DEVICE& device,
        const checkpoint::beta_mean::LayerForward<LAYER_SPEC>&,
        const Tensor<INPUT_SPEC>& input,
        Tensor<OUTPUT_SPEC>& output,
        checkpoint::beta_mean::Buffer&,
        RNG&,
        const Mode<MODE>& = Mode<mode::Default<>>{}
    ) {
        using T = typename INPUT_SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(tensor::dense_row_major_layout<INPUT_SPEC, true>());
        static_assert(tensor::dense_row_major_layout<OUTPUT_SPEC, true>());
        static_assert(get_last(typename INPUT_SPEC::SHAPE{}) == LAYER_SPEC::INPUT_DIM);
        static_assert(get_last(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::OUTPUT_DIM);
        static_assert(LAYER_SPEC::INPUT_DIM == 2 * LAYER_SPEC::OUTPUT_DIM);

        constexpr auto input_dim = LAYER_SPEC::INPUT_DIM;
        constexpr auto output_dim = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto internal_batch_size = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        for (TI batch_i = 0; batch_i < (TI)internal_batch_size; ++batch_i) {
            for (TI action_i = 0; action_i < (TI)output_dim; ++action_i) {
                const T alpha_raw = get_flat(device, input, batch_i * input_dim + action_i);
                const T beta_raw = get_flat(device, input, batch_i * input_dim + output_dim + action_i);
                const T alpha = checkpoint_beta_mean_softplus(alpha_raw) + (T)1;
                const T beta = checkpoint_beta_mean_softplus(beta_raw) + (T)1;
                *(data(output) + batch_i * output_dim + action_i) = alpha / (alpha + beta);
            }
        }
    }

    template <
        typename DEVICE,
        typename LAYER_SPEC,
        typename INPUT_SPEC,
        typename OUTPUT_SPEC,
        typename RNG,
        typename MODE = mode::Default<>
    >
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(
        DEVICE& device,
        const checkpoint::beta_mean::LayerForward<LAYER_SPEC>& layer,
        const Tensor<INPUT_SPEC>& input,
        checkpoint::beta_mean::State&,
        Tensor<OUTPUT_SPEC>& output,
        checkpoint::beta_mean::Buffer& buffer,
        RNG& rng,
        const Mode<MODE>& mode = Mode<mode::Default<>>{}
    ) {
        evaluate(device, layer, input, output, buffer, rng, mode);
    }

    template <typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(
        DEVICE&,
        checkpoint::beta_mean::State&,
        const Mode<MODE>& = Mode<mode::Default<>>{}
    ) {
        return false;
    }

    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(
        DEVICE&,
        checkpoint::beta_mean::LayerForward<SPEC>&,
        const Mode<MODE>& = Mode<mode::Default<>>{}
    ) {
        return false;
    }
}


namespace rl_tools::checkpoint::actor {
    namespace layer_0 {
        namespace weights {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    64, 167, 77, 192, 26, 69, 135, 64, 132, 109, 82, 64, 247, 38, 185, 60,
                    146, 35, 169, 62, 88, 80, 27, 192, 19, 200, 176, 191, 54, 218, 191, 189,
                    70, 8, 108, 64, 245, 33, 3, 64, 92, 128, 49, 192, 1, 199, 42, 61,
                    222, 117, 44, 64, 175, 73, 111, 192, 164, 186, 152, 190, 181, 182, 53, 192,
                    252, 81, 72, 64, 33, 171, 8, 64, 22, 27, 0, 191, 73, 191, 100, 191,
                    98, 152, 3, 63, 104, 47, 78, 189, 252, 56, 148, 188, 156, 108, 163, 60,
                    133, 219, 157, 60, 237, 121, 215, 189, 105, 27, 193, 190, 105, 233, 166, 62,
                    29, 42, 7, 62, 223, 78, 85, 189, 120, 22, 209, 63, 12, 104, 72, 62,
                    73, 18, 193, 63, 94, 173, 6, 62, 246, 223, 182, 62, 125, 127, 133, 64,
                    159, 248, 203, 190, 0, 76, 224, 189, 8, 212, 93, 63, 204, 194, 135, 192,
                    180, 27, 76, 191, 202, 252, 56, 189, 197, 86, 138, 192, 47, 251, 146, 191,
                    236, 211, 118, 190, 87, 107, 19, 64, 92, 139, 58, 63, 51, 27, 142, 63,
                    74, 90, 192, 190, 141, 12, 214, 63, 51, 102, 67, 191, 190, 215, 165, 191,
                    171, 94, 42, 191, 210, 96, 6, 62, 24, 30, 77, 189, 234, 103, 181, 189,
                    29, 34, 27, 62, 4, 41, 6, 190, 42, 194, 222, 61, 197, 183, 200, 190,
                    16, 228, 231, 63, 160, 197, 22, 63, 50, 51, 119, 192, 158, 166, 3, 62,
                    160, 150, 107, 63, 73, 199, 49, 63, 187, 49, 149, 63, 66, 166, 155, 61,
                    83, 178, 196, 62, 88, 170, 164, 191, 185, 39, 138, 191, 218, 255, 5, 190,
                    235, 237, 50, 191, 99, 56, 246, 190, 69, 214, 115, 61, 197, 59, 2, 64,
                    148, 68, 41, 189, 27, 205, 83, 192, 4, 29, 128, 191, 116, 243, 165, 191,
                    174, 134, 1, 62, 224, 44, 204, 190, 131, 156, 119, 190, 28, 164, 126, 61,
                    209, 136, 74, 61, 22, 149, 64, 62, 189, 215, 91, 62, 254, 128, 86, 190,
                    109, 253, 98, 62, 168, 212, 206, 188, 47, 86, 94, 192, 222, 14, 136, 191,
                    176, 123, 27, 62, 19, 98, 38, 61, 3, 222, 217, 62, 152, 203, 25, 192,
                    134, 184, 151, 62, 196, 23, 4, 188, 235, 220, 100, 192, 212, 21, 31, 64,
                    194, 154, 60, 64, 135, 20, 133, 61, 65, 213, 48, 64, 249, 33, 84, 64,
                    19, 163, 54, 189, 112, 164, 36, 192, 220, 17, 173, 191, 88, 97, 15, 63,
                    25, 188, 2, 64, 100, 160, 234, 191, 204, 45, 140, 189, 25, 173, 35, 61,
                    191, 26, 71, 190, 110, 109, 220, 59, 108, 62, 203, 188, 44, 80, 247, 188,
                    85, 110, 8, 62, 194, 17, 239, 61, 68, 246, 190, 190, 226, 174, 152, 62,
                    225, 223, 51, 62, 121, 233, 79, 189, 88, 206, 93, 191, 204, 64, 0, 190,
                    240, 93, 1, 192, 6, 184, 15, 64, 180, 107, 104, 190, 10, 48, 184, 188,
                    46, 254, 14, 192, 67, 127, 50, 192, 213, 131, 42, 64, 215, 133, 73, 62,
                    176, 64, 34, 192, 174, 124, 233, 63, 172, 2, 166, 189, 175, 76, 238, 63,
                    187, 120, 106, 190, 89, 189, 248, 190, 135, 4, 241, 63, 5, 226, 16, 64,
                    246, 176, 170, 63, 103, 24, 71, 191, 55, 175, 20, 191, 41, 50, 41, 190,
                    117, 229, 3, 189, 237, 174, 189, 190, 32, 112, 51, 190, 147, 125, 214, 189,
                    152, 201, 48, 189, 121, 47, 201, 62, 75, 145, 52, 64, 174, 77, 171, 64,
                    241, 5, 3, 191, 1, 168, 126, 62, 206, 224, 250, 62, 229, 115, 78, 63,
                    148, 249, 232, 190, 0, 100, 161, 189, 96, 127, 138, 64, 250, 126, 223, 191,
                    19, 66, 157, 192, 182, 92, 69, 190, 230, 148, 165, 191, 36, 133, 147, 192,
                    145, 81, 178, 189, 179, 191, 174, 63, 236, 53, 152, 64, 106, 182, 16, 191,
                    15, 167, 209, 191, 178, 11, 88, 63, 93, 133, 190, 190, 253, 101, 237, 61,
                    55, 190, 34, 61, 48, 138, 64, 189, 14, 248, 135, 186, 108, 49, 219, 61,
                    142, 50, 9, 190, 164, 230, 12, 62, 40, 15, 191, 62, 240, 45, 193, 190,
                    223, 51, 5, 192, 76, 77, 7, 63, 193, 176, 248, 192, 125, 60, 149, 61,
                    205, 6, 150, 191, 32, 197, 206, 62, 68, 29, 155, 63, 178, 9, 81, 189,
                    150, 9, 23, 62, 114, 228, 141, 190, 152, 53, 150, 191, 62, 4, 44, 61,
                    158, 14, 48, 191, 253, 102, 81, 190, 16, 109, 71, 190, 125, 229, 176, 191,
                    111, 91, 12, 63, 53, 15, 189, 192, 134, 51, 133, 190, 163, 222, 163, 62,
                    132, 163, 23, 64, 93, 27, 143, 189, 42, 175, 252, 61, 43, 225, 136, 60,
                    213, 172, 150, 189, 175, 29, 155, 58, 193, 228, 41, 61, 50, 14, 132, 190,
                    80, 204, 125, 189, 236, 218, 219, 190, 77, 37, 31, 191, 78, 221, 26, 191,
                    174, 219, 71, 191, 179, 110, 147, 189, 73, 7, 146, 63, 23, 210, 202, 62,
                    45, 71, 48, 191, 54, 115, 7, 190, 104, 155, 131, 64, 19, 97, 220, 189,
                    81, 40, 177, 192, 153, 177, 187, 188, 186, 9, 249, 190, 191, 44, 137, 192,
                    6, 77, 17, 62, 115, 235, 115, 189, 29, 161, 120, 63, 88, 124, 185, 62,
                    33, 199, 69, 192, 38, 29, 113, 190, 157, 84, 216, 63, 62, 242, 98, 190,
                    237, 22, 87, 189, 222, 5, 30, 59, 142, 130, 244, 188, 249, 238, 90, 62,
                    101, 50, 92, 190, 177, 61, 129, 189, 217, 254, 23, 62, 72, 246, 23, 62,
                    134, 142, 53, 190, 81, 231, 151, 63, 38, 157, 223, 188, 137, 149, 14, 61,
                    53, 172, 192, 191, 46, 15, 227, 191, 69, 108, 11, 64, 219, 241, 23, 61,
                    72, 98, 27, 63, 57, 92, 24, 64, 10, 166, 179, 191, 39, 197, 116, 62,
                    196, 89, 4, 64, 75, 91, 14, 191, 187, 126, 140, 190, 109, 53, 35, 191,
                    111, 122, 26, 62, 208, 91, 121, 63, 141, 12, 208, 62, 187, 111, 222, 191,
                    98, 171, 73, 190, 168, 221, 87, 189, 104, 112, 244, 188, 74, 157, 3, 62,
                    197, 10, 225, 188, 167, 242, 167, 62, 205, 239, 206, 190, 211, 187, 170, 62,
                    118, 220, 1, 60, 36, 95, 26, 191, 144, 49, 204, 190, 107, 170, 2, 191,
                    138, 240, 244, 63, 68, 53, 235, 60, 60, 55, 53, 64, 64, 10, 65, 63,
                    17, 21, 79, 192, 250, 45, 84, 62, 148, 98, 246, 191, 24, 37, 140, 191,
                    193, 230, 28, 64, 50, 51, 35, 62, 191, 13, 82, 191, 168, 220, 1, 64,
                    69, 190, 75, 190, 136, 44, 9, 63, 125, 194, 30, 191, 201, 214, 161, 63,
                    210, 54, 123, 63, 186, 251, 37, 63, 179, 120, 244, 191, 57, 250, 148, 61,
                    47, 96, 157, 61, 28, 45, 2, 61, 208, 132, 24, 61, 60, 167, 94, 60,
                    16, 226, 225, 61, 97, 49, 40, 191, 187, 150, 92, 191, 137, 35, 35, 62,
                    204, 52, 11, 192, 178, 121, 182, 190, 192, 255, 224, 191, 254, 118, 3, 62,
                    144, 234, 105, 64, 85, 218, 58, 192, 85, 22, 69, 192, 164, 75, 21, 61,
                    129, 56, 173, 62, 162, 42, 251, 63, 198, 121, 224, 190, 206, 75, 199, 189,
                    40, 161, 55, 64, 191, 20, 171, 189, 83, 54, 224, 61, 2, 139, 213, 191,
                    144, 14, 45, 191, 5, 146, 228, 191, 66, 65, 210, 189, 66, 98, 11, 191,
                    33, 212, 155, 192, 184, 193, 177, 61, 126, 70, 87, 190, 182, 107, 88, 60,
                    142, 43, 19, 189, 251, 95, 95, 61, 133, 201, 121, 188, 35, 246, 54, 63,
                    65, 223, 17, 191, 83, 7, 179, 62, 112, 107, 181, 191, 253, 82, 230, 63,
                    222, 218, 131, 192, 253, 2, 77, 62, 103, 28, 62, 190, 97, 135, 14, 192,
                    199, 36, 156, 189, 201, 88, 63, 60, 215, 200, 195, 63, 159, 221, 44, 64,
                    97, 15, 16, 192, 2, 54, 142, 62, 35, 50, 27, 64, 217, 166, 181, 191,
                    210, 51, 99, 190, 189, 73, 84, 192, 196, 236, 241, 63, 103, 152, 178, 192,
                    30, 133, 47, 62, 232, 71, 248, 187, 162, 110, 228, 191, 148, 243, 82, 61,
                    71, 56, 76, 188, 208, 79, 116, 188, 239, 195, 129, 189, 167, 11, 37, 190,
                    211, 173, 19, 191, 235, 239, 91, 62, 48, 41, 72, 190, 140, 242, 130, 190,
                    133, 245, 134, 63, 217, 122, 139, 61, 11, 242, 127, 191, 2, 73, 131, 62,
                    21, 145, 39, 188, 57, 23, 196, 191, 206, 114, 133, 62, 120, 236, 110, 62,
                    35, 55, 75, 63, 232, 196, 0, 64, 51, 77, 131, 191, 111, 92, 163, 62,
                    45, 13, 192, 63, 136, 146, 118, 191, 114, 21, 153, 190, 45, 1, 129, 190,
                    131, 40, 223, 190, 185, 140, 9, 187, 132, 28, 88, 191, 44, 81, 103, 62,
                    102, 88, 19, 192, 105, 85, 72, 190, 97, 151, 177, 190, 42, 235, 243, 190,
                    100, 9, 180, 189, 53, 170, 2, 61, 69, 164, 78, 60, 7, 122, 150, 191,
                    248, 208, 80, 189, 95, 45, 12, 191, 197, 2, 119, 64, 50, 124, 17, 64,
                    46, 249, 70, 191, 73, 202, 141, 190, 227, 78, 43, 192, 239, 12, 233, 63,
                    13, 1, 246, 63, 7, 160, 204, 190, 234, 16, 160, 62, 192, 205, 182, 191,
                    155, 9, 132, 63, 245, 161, 33, 190, 159, 22, 250, 191, 103, 106, 80, 190,
                    63, 38, 197, 61, 235, 130, 50, 64, 160, 93, 8, 63, 213, 14, 223, 191,
                    159, 158, 53, 62, 205, 33, 6, 63, 130, 72, 178, 63, 174, 116, 193, 62,
                    132, 139, 153, 62, 18, 95, 231, 189, 210, 245, 161, 189, 124, 102, 210, 189,
                    239, 12, 181, 62, 214, 85, 227, 190, 164, 13, 78, 63, 161, 214, 155, 190,
                    133, 53, 108, 192, 19, 164, 234, 62, 135, 168, 15, 63, 191, 99, 119, 188,
                    165, 170, 115, 191, 168, 239, 145, 192, 147, 21, 75, 63, 82, 176, 154, 59,
                    29, 109, 52, 63, 95, 104, 169, 64, 16, 144, 150, 191, 112, 142, 36, 190,
                    75, 46, 158, 64, 135, 195, 100, 191, 159, 156, 150, 189, 254, 146, 69, 192,
                    58, 128, 14, 63, 247, 238, 56, 63, 222, 69, 38, 191, 2, 170, 208, 191,
                    144, 123, 135, 63, 252, 34, 156, 61, 198, 21, 215, 190, 183, 116, 62, 60,
                    185, 34, 11, 188, 18, 155, 134, 189, 62, 244, 175, 61, 56, 217, 79, 62,
                    147, 134, 7, 61, 108, 217, 177, 62, 181, 8, 67, 63, 55, 232, 205, 191,
                    148, 46, 156, 191, 149, 209, 198, 190, 130, 237, 168, 63, 10, 22, 1, 191,
                    249, 76, 91, 190, 43, 5, 215, 59, 249, 88, 165, 191, 159, 214, 234, 62,
                    66, 50, 167, 63, 47, 69, 168, 190, 64, 43, 164, 62, 142, 92, 208, 63,
                    133, 149, 101, 62, 149, 106, 216, 62, 245, 52, 49, 191, 26, 252, 42, 192,
                    28, 117, 216, 190, 204, 187, 58, 190, 121, 9, 92, 63, 252, 97, 218, 60,
                    71, 53, 225, 61, 156, 3, 143, 60, 204, 78, 201, 59, 243, 126, 27, 61,
                    89, 75, 152, 62, 158, 157, 158, 61, 212, 242, 1, 191, 235, 58, 116, 63,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 16, 30>;
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
                    232, 129, 193, 189, 74, 53, 35, 189, 152, 159, 3, 190, 99, 130, 250, 189,
                    79, 96, 20, 190, 254, 197, 207, 186, 27, 174, 83, 60, 43, 254, 22, 61,
                    38, 70, 156, 190, 203, 119, 234, 189, 203, 211, 250, 61, 9, 69, 192, 189,
                    172, 175, 75, 60, 3, 83, 190, 188, 36, 195, 56, 62, 92, 176, 63, 62,
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
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 30>;
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
                    191, 11, 68, 63, 244, 63, 56, 189, 5, 19, 139, 189, 91, 206, 212, 190,
                    174, 105, 156, 190, 199, 194, 72, 191, 182, 250, 12, 189, 242, 219, 179, 62,
                    79, 115, 98, 189, 38, 202, 9, 190, 2, 220, 223, 189, 157, 198, 14, 191,
                    136, 226, 37, 190, 211, 110, 73, 188, 67, 23, 194, 62, 114, 26, 54, 62,
                    94, 130, 73, 190, 251, 189, 70, 61, 182, 197, 143, 61, 110, 57, 142, 190,
                    119, 134, 44, 63, 238, 245, 246, 190, 146, 25, 0, 191, 102, 138, 106, 188,
                    30, 155, 163, 60, 94, 1, 239, 188, 79, 165, 21, 62, 28, 37, 85, 62,
                    167, 190, 139, 190, 203, 203, 103, 190, 187, 93, 96, 189, 189, 98, 40, 61,
                    22, 232, 127, 191, 131, 230, 87, 63, 138, 62, 249, 61, 128, 104, 195, 190,
                    114, 188, 189, 62, 253, 236, 185, 61, 82, 97, 141, 62, 97, 56, 13, 61,
                    88, 179, 211, 190, 145, 236, 165, 62, 195, 67, 190, 190, 228, 158, 181, 190,
                    161, 57, 183, 189, 249, 35, 167, 62, 99, 202, 17, 191, 104, 233, 72, 62,
                    76, 20, 50, 61, 66, 95, 145, 61, 105, 49, 70, 190, 103, 41, 253, 59,
                    210, 218, 164, 190, 180, 217, 32, 190, 73, 196, 179, 61, 15, 121, 148, 62,
                    3, 63, 237, 189, 159, 168, 172, 62, 82, 161, 139, 190, 152, 84, 108, 60,
                    203, 179, 54, 62, 163, 201, 77, 188, 170, 234, 170, 189, 12, 26, 249, 62,
                    68, 130, 74, 62, 149, 146, 181, 190, 174, 94, 36, 189, 141, 22, 143, 62,
                    118, 243, 218, 61, 196, 234, 189, 61, 89, 128, 244, 62, 91, 114, 210, 190,
                    83, 116, 119, 190, 59, 100, 167, 62, 253, 32, 33, 62, 137, 55, 156, 62,
                    137, 142, 170, 62, 120, 108, 141, 190, 190, 81, 191, 61, 14, 101, 165, 62,
                    9, 17, 26, 190, 37, 150, 117, 61, 12, 165, 242, 190, 230, 195, 145, 190,
                    248, 237, 12, 62, 180, 158, 6, 191, 119, 9, 213, 190, 50, 55, 247, 61,
                    231, 213, 224, 189, 236, 185, 156, 190, 63, 246, 155, 190, 213, 39, 138, 59,
                    216, 94, 28, 190, 111, 29, 123, 190, 207, 38, 152, 190, 144, 210, 85, 190,
                    105, 234, 167, 60, 77, 255, 146, 189, 235, 76, 149, 62, 196, 104, 0, 63,
                    218, 15, 124, 62, 91, 191, 142, 62, 139, 97, 26, 190, 79, 23, 92, 62,
                    18, 205, 182, 61, 172, 167, 238, 189, 251, 4, 153, 62, 254, 147, 189, 189,
                    114, 233, 17, 191, 48, 52, 117, 62, 62, 80, 218, 187, 238, 95, 184, 62,
                    245, 253, 131, 191, 146, 177, 18, 62, 91, 2, 127, 61, 223, 52, 10, 60,
                    96, 67, 80, 190, 179, 60, 129, 191, 204, 176, 135, 190, 175, 232, 219, 189,
                    215, 74, 14, 62, 166, 106, 26, 189, 82, 163, 197, 62, 4, 5, 109, 190,
                    99, 86, 115, 62, 167, 210, 13, 190, 73, 48, 6, 190, 151, 173, 162, 61,
                    168, 246, 76, 189, 104, 213, 44, 191, 28, 87, 26, 189, 115, 163, 248, 61,
                    174, 89, 26, 62, 216, 185, 77, 191, 22, 205, 16, 63, 54, 98, 149, 62,
                    99, 87, 164, 190, 38, 196, 166, 188, 127, 179, 95, 62, 230, 175, 239, 62,
                    138, 104, 94, 190, 137, 237, 254, 190, 95, 69, 145, 62, 41, 176, 88, 62,
                    133, 63, 237, 190, 6, 250, 206, 190, 194, 143, 210, 62, 136, 111, 135, 62,
                    189, 14, 166, 62, 215, 13, 64, 190, 248, 106, 47, 63, 33, 38, 24, 189,
                    223, 94, 71, 62, 21, 132, 0, 191, 180, 204, 55, 191, 211, 42, 121, 190,
                    242, 101, 52, 62, 213, 149, 28, 63, 155, 228, 96, 62, 214, 93, 4, 190,
                    198, 74, 46, 190, 70, 122, 141, 62, 181, 133, 136, 190, 240, 96, 56, 190,
                    109, 179, 94, 62, 112, 239, 134, 190, 145, 139, 228, 62, 152, 23, 94, 62,
                    76, 220, 103, 62, 61, 123, 255, 189, 68, 46, 221, 62, 124, 179, 210, 62,
                    4, 150, 102, 188, 49, 225, 86, 62, 203, 36, 141, 190, 150, 155, 129, 190,
                    248, 124, 89, 190, 234, 70, 246, 60, 174, 102, 128, 190, 31, 156, 38, 190,
                    205, 113, 247, 189, 213, 216, 227, 62, 115, 169, 73, 62, 185, 248, 37, 190,
                    210, 236, 247, 61, 197, 115, 196, 190, 124, 85, 19, 62, 30, 40, 116, 60,
                    67, 172, 11, 190, 159, 49, 4, 190, 191, 120, 14, 190, 128, 35, 244, 61,
                    107, 123, 204, 189, 219, 166, 18, 62, 139, 229, 137, 190, 103, 167, 103, 189,
                    152, 164, 149, 61, 247, 250, 142, 189, 244, 143, 158, 62, 215, 129, 2, 62,
                    199, 137, 234, 189, 162, 38, 238, 190, 162, 233, 84, 63, 122, 88, 163, 190,
                    150, 146, 252, 60, 31, 105, 5, 63, 90, 208, 101, 61, 105, 169, 12, 190,
                    93, 107, 201, 61, 82, 235, 61, 62, 130, 129, 205, 190, 172, 120, 85, 190,
                    140, 70, 215, 190, 223, 144, 40, 190, 1, 63, 132, 62, 5, 154, 144, 62,
                    190, 152, 221, 189, 162, 110, 62, 190, 42, 70, 18, 191, 96, 224, 94, 61,
                    57, 189, 217, 62, 74, 83, 81, 190, 152, 174, 131, 62, 45, 13, 216, 189,
                    218, 195, 237, 188, 243, 49, 58, 189, 86, 198, 252, 190, 35, 145, 5, 63,
                    2, 65, 235, 62, 96, 186, 116, 191, 112, 94, 156, 188, 183, 87, 14, 191,
                    246, 233, 133, 61, 230, 244, 242, 62, 52, 238, 215, 190, 235, 11, 143, 190,
                    28, 30, 104, 61, 36, 213, 194, 189, 244, 113, 158, 190, 190, 73, 7, 190,
                    147, 2, 178, 190, 199, 135, 8, 189, 210, 250, 135, 190, 251, 171, 197, 188,
                    56, 38, 164, 62, 174, 49, 114, 62, 169, 166, 140, 190, 153, 217, 10, 63,
                    52, 86, 47, 61, 208, 219, 90, 62, 220, 198, 15, 63, 75, 157, 171, 190,
                    50, 27, 234, 62, 44, 12, 220, 62, 96, 68, 153, 189, 74, 232, 29, 189,
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
                    184, 232, 183, 189, 151, 196, 73, 189, 86, 13, 227, 61, 32, 77, 194, 61,
                    40, 36, 223, 189, 162, 23, 254, 60, 163, 41, 65, 189, 48, 116, 21, 59,
                    96, 141, 174, 188, 160, 121, 3, 62, 207, 129, 7, 62, 102, 112, 90, 190,
                    222, 111, 13, 60, 241, 191, 178, 61, 119, 64, 115, 62, 221, 237, 134, 62,
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
    namespace layer_2 {
        namespace weights_input {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    72, 206, 31, 190, 111, 100, 42, 63, 119, 42, 102, 188, 26, 122, 45, 191,
                    103, 122, 222, 191, 75, 137, 32, 63, 42, 106, 103, 62, 182, 8, 85, 191,
                    99, 52, 76, 191, 241, 6, 42, 64, 45, 187, 252, 62, 203, 227, 6, 64,
                    90, 21, 239, 190, 124, 157, 194, 189, 246, 128, 12, 192, 189, 223, 43, 191,
                    244, 68, 175, 62, 121, 186, 98, 191, 208, 8, 140, 62, 210, 244, 171, 61,
                    149, 100, 61, 191, 59, 116, 118, 190, 124, 246, 218, 61, 107, 227, 159, 190,
                    103, 191, 103, 190, 152, 70, 41, 189, 202, 43, 13, 63, 141, 241, 17, 63,
                    0, 249, 59, 63, 57, 141, 85, 62, 52, 26, 243, 189, 95, 251, 210, 61,
                    153, 87, 148, 61, 14, 66, 142, 188, 143, 169, 150, 62, 188, 184, 34, 62,
                    62, 216, 216, 190, 135, 33, 187, 62, 214, 237, 161, 190, 35, 99, 72, 190,
                    78, 217, 148, 189, 12, 62, 6, 63, 81, 217, 245, 59, 4, 93, 29, 63,
                    232, 78, 189, 62, 224, 49, 213, 189, 12, 27, 134, 190, 135, 237, 160, 62,
                    67, 195, 132, 63, 192, 118, 165, 190, 228, 91, 70, 60, 87, 29, 140, 63,
                    32, 116, 93, 191, 190, 110, 9, 192, 221, 181, 21, 63, 135, 1, 210, 190,
                    245, 229, 1, 192, 70, 148, 152, 191, 196, 5, 80, 62, 80, 209, 169, 191,
                    13, 143, 199, 63, 212, 249, 21, 190, 95, 28, 77, 191, 209, 251, 105, 63,
                    9, 116, 196, 190, 169, 227, 60, 191, 51, 123, 1, 62, 137, 99, 135, 63,
                    169, 43, 91, 63, 120, 217, 131, 61, 242, 235, 176, 190, 242, 106, 11, 63,
                    101, 123, 98, 63, 63, 1, 16, 190, 234, 163, 168, 62, 22, 114, 59, 191,
                    175, 244, 116, 63, 105, 222, 7, 61, 173, 58, 166, 63, 161, 30, 71, 63,
                    76, 206, 133, 63, 120, 61, 31, 190, 105, 41, 112, 191, 202, 205, 174, 191,
                    219, 139, 2, 63, 157, 204, 200, 190, 79, 190, 247, 187, 33, 82, 159, 61,
                    89, 43, 190, 189, 220, 128, 248, 62, 238, 184, 87, 191, 122, 154, 18, 192,
                    223, 105, 13, 192, 79, 235, 167, 191, 133, 235, 255, 63, 15, 4, 239, 191,
                    34, 247, 201, 189, 163, 115, 67, 63, 169, 185, 204, 62, 100, 10, 254, 190,
                    104, 21, 243, 62, 149, 128, 221, 62, 172, 22, 133, 191, 37, 25, 105, 191,
                    68, 8, 89, 63, 135, 15, 1, 63, 248, 88, 152, 63, 71, 169, 52, 62,
                    34, 2, 240, 190, 60, 178, 196, 63, 13, 186, 252, 62, 205, 42, 185, 191,
                    137, 160, 22, 62, 122, 59, 156, 62, 0, 231, 124, 62, 200, 143, 39, 62,
                    53, 215, 11, 191, 99, 142, 81, 189, 188, 167, 40, 63, 25, 146, 137, 63,
                    162, 117, 121, 188, 145, 35, 5, 63, 160, 166, 20, 191, 178, 157, 144, 62,
                    28, 214, 29, 191, 88, 147, 131, 63, 113, 63, 134, 191, 174, 18, 184, 191,
                    214, 196, 91, 63, 198, 18, 62, 191, 53, 244, 23, 62, 39, 211, 154, 63,
                    203, 174, 51, 190, 150, 251, 2, 63, 143, 207, 225, 190, 239, 89, 53, 62,
                    30, 223, 71, 62, 132, 117, 94, 191, 118, 76, 85, 62, 252, 30, 154, 60,
                    67, 112, 194, 190, 9, 57, 99, 63, 175, 111, 237, 62, 81, 176, 200, 190,
                    68, 231, 248, 62, 140, 131, 19, 191, 179, 170, 5, 191, 101, 83, 35, 190,
                    91, 170, 30, 191, 243, 3, 3, 190, 176, 231, 184, 191, 252, 41, 227, 63,
                    145, 18, 109, 191, 22, 193, 201, 188, 6, 199, 112, 190, 52, 84, 33, 191,
                    47, 125, 7, 63, 64, 9, 210, 63, 34, 12, 51, 191, 233, 16, 180, 63,
                    230, 148, 79, 191, 132, 210, 12, 57, 205, 193, 186, 63, 146, 38, 112, 191,
                    73, 169, 153, 62, 106, 64, 15, 192, 91, 145, 57, 64, 8, 67, 191, 191,
                    214, 169, 209, 191, 224, 90, 185, 63, 243, 20, 12, 63, 33, 42, 38, 63,
                    224, 81, 221, 62, 59, 94, 13, 192, 242, 162, 31, 188, 170, 237, 75, 62,
                    35, 136, 114, 191, 42, 103, 165, 191, 145, 115, 45, 63, 115, 226, 141, 63,
                    212, 52, 191, 191, 11, 59, 151, 62, 202, 66, 37, 190, 29, 111, 227, 190,
                    83, 225, 162, 191, 39, 65, 71, 63, 182, 170, 121, 191, 76, 228, 40, 191,
                    22, 163, 1, 191, 109, 117, 138, 62, 68, 117, 33, 191, 28, 254, 55, 63,
                    218, 160, 90, 63, 108, 251, 158, 59, 113, 40, 177, 191, 235, 84, 236, 189,
                    26, 162, 42, 62, 233, 121, 234, 190, 232, 178, 164, 61, 177, 166, 191, 60,
                    20, 40, 108, 63, 43, 208, 150, 190, 61, 56, 176, 188, 183, 61, 78, 189,
                    93, 41, 12, 191, 145, 159, 59, 189, 204, 215, 7, 189, 169, 28, 223, 188,
                    209, 232, 255, 62, 19, 188, 207, 190, 18, 41, 152, 191, 92, 120, 129, 190,
                    28, 246, 102, 189, 44, 154, 187, 190, 2, 99, 61, 62, 142, 63, 65, 191,
                    180, 220, 148, 63, 209, 230, 4, 191, 41, 197, 55, 63, 47, 30, 1, 61,
                    125, 73, 3, 63, 69, 236, 187, 62, 202, 189, 15, 190, 121, 249, 96, 191,
                    229, 61, 242, 190, 223, 142, 142, 61, 3, 95, 92, 63, 126, 36, 103, 62,
                    176, 161, 241, 189, 16, 204, 165, 190, 33, 3, 200, 189, 41, 126, 19, 62,
                    45, 24, 209, 190, 65, 101, 215, 61, 93, 6, 127, 63, 74, 30, 79, 62,
                    149, 254, 102, 59, 227, 53, 251, 190, 153, 133, 144, 62, 111, 63, 111, 190,
                    68, 222, 175, 62, 32, 22, 200, 63, 25, 98, 71, 63, 216, 231, 98, 63,
                    173, 246, 77, 63, 119, 247, 176, 63, 123, 69, 29, 190, 247, 102, 19, 62,
                    189, 5, 70, 62, 155, 84, 211, 62, 155, 118, 177, 190, 35, 240, 152, 191,
                    242, 208, 110, 62, 193, 26, 150, 190, 152, 17, 165, 63, 96, 5, 186, 62,
                    4, 70, 67, 63, 82, 130, 67, 63, 136, 225, 187, 63, 253, 12, 228, 63,
                    18, 243, 182, 191, 231, 46, 211, 63, 21, 215, 146, 63, 201, 170, 9, 191,
                    230, 30, 88, 191, 10, 47, 104, 61, 145, 181, 106, 63, 73, 250, 240, 63,
                    153, 219, 2, 63, 227, 244, 215, 190, 146, 19, 139, 190, 104, 206, 3, 192,
                    172, 248, 15, 189, 118, 56, 36, 191, 91, 114, 50, 190, 6, 114, 6, 191,
                    116, 206, 169, 62, 141, 200, 228, 190, 81, 180, 22, 190, 131, 27, 11, 191,
                    36, 59, 146, 191, 127, 196, 226, 62, 196, 233, 226, 190, 37, 123, 209, 62,
                    146, 209, 22, 190, 109, 106, 196, 190, 154, 39, 182, 62, 255, 120, 142, 191,
                    74, 232, 82, 60, 222, 31, 251, 62, 226, 42, 167, 189, 161, 101, 20, 60,
                    238, 208, 16, 63, 134, 216, 13, 63, 125, 221, 179, 190, 155, 152, 231, 63,
                    232, 160, 156, 63, 168, 123, 137, 63, 199, 255, 241, 190, 189, 193, 80, 61,
                    86, 32, 92, 191, 249, 106, 144, 63, 93, 68, 138, 191, 106, 183, 130, 61,
                    241, 64, 44, 191, 56, 147, 30, 192, 46, 248, 77, 191, 178, 33, 143, 63,
                    239, 213, 227, 190, 173, 134, 205, 191, 54, 252, 159, 62, 173, 140, 10, 62,
                    163, 246, 244, 190, 139, 148, 90, 63, 48, 191, 151, 62, 132, 70, 253, 191,
                    76, 99, 104, 63, 40, 214, 39, 63, 16, 40, 138, 62, 202, 88, 126, 63,
                    37, 254, 22, 191, 195, 119, 54, 191, 13, 103, 15, 63, 85, 93, 233, 61,
                    197, 180, 92, 63, 97, 206, 4, 190, 49, 82, 134, 63, 24, 156, 5, 63,
                    152, 114, 3, 62, 27, 69, 15, 63, 128, 126, 165, 190, 161, 100, 170, 63,
                    126, 227, 169, 63, 52, 231, 57, 61, 14, 136, 149, 191, 181, 46, 7, 191,
                    240, 230, 64, 191, 137, 228, 73, 63, 60, 237, 205, 190, 251, 103, 116, 190,
                    30, 65, 111, 191, 184, 47, 33, 191, 91, 109, 122, 63, 241, 42, 170, 191,
                    15, 85, 117, 62, 143, 250, 116, 60, 5, 103, 86, 63, 244, 41, 26, 191,
                    71, 48, 46, 189, 254, 246, 96, 191, 60, 69, 96, 191, 237, 11, 41, 63,
                    39, 79, 191, 191, 163, 219, 128, 62, 181, 68, 244, 190, 0, 212, 230, 191,
                    205, 50, 199, 61, 172, 150, 101, 63, 9, 119, 171, 63, 122, 241, 33, 64,
                    106, 62, 163, 63, 48, 222, 145, 63, 79, 225, 131, 191, 195, 49, 93, 191,
                    58, 91, 22, 192, 71, 38, 54, 191, 57, 211, 161, 63, 192, 203, 133, 191,
                    36, 145, 196, 62, 93, 11, 182, 191, 53, 113, 233, 190, 131, 192, 95, 190,
                    73, 249, 156, 190, 157, 44, 141, 191, 31, 110, 187, 63, 161, 163, 97, 191,
                    50, 214, 184, 61, 186, 244, 4, 190, 226, 39, 152, 191, 3, 167, 9, 63,
                    83, 145, 5, 191, 221, 217, 175, 190, 210, 55, 82, 191, 221, 113, 50, 192,
                    63, 57, 29, 191, 93, 57, 187, 62, 212, 248, 234, 189, 229, 252, 129, 190,
                    116, 56, 204, 63, 175, 217, 6, 191, 86, 104, 59, 63, 37, 137, 132, 191,
                    47, 198, 156, 62, 146, 53, 196, 63, 59, 29, 60, 190, 127, 8, 32, 63,
                    130, 219, 165, 191, 144, 147, 0, 62, 186, 98, 98, 190, 171, 8, 235, 191,
                    237, 88, 86, 188, 203, 55, 238, 189, 174, 151, 166, 61, 60, 101, 145, 61,
                    61, 209, 233, 53, 121, 255, 19, 63, 192, 98, 233, 190, 130, 189, 129, 63,
                    22, 248, 232, 190, 7, 114, 216, 190, 0, 240, 196, 62, 252, 1, 127, 62,
                    7, 110, 205, 62, 230, 112, 80, 62, 111, 21, 32, 62, 24, 170, 166, 61,
                    231, 174, 100, 61, 101, 15, 47, 190, 218, 194, 32, 62, 116, 168, 109, 60,
                    87, 112, 218, 190, 222, 153, 8, 191, 243, 136, 100, 190, 233, 188, 190, 62,
                    127, 142, 25, 63, 231, 83, 154, 190, 11, 189, 210, 63, 68, 252, 142, 190,
                    19, 13, 129, 63, 32, 71, 65, 62, 54, 232, 183, 60, 182, 120, 52, 63,
                    52, 197, 34, 63, 60, 16, 208, 62, 7, 100, 79, 191, 43, 122, 139, 62,
                    135, 3, 175, 191, 127, 26, 149, 62, 68, 206, 7, 62, 216, 108, 30, 191,
                    12, 221, 189, 63, 57, 68, 12, 62, 81, 236, 231, 62, 182, 42, 38, 191,
                    53, 53, 186, 190, 228, 179, 16, 63, 131, 83, 155, 191, 250, 116, 222, 191,
                    151, 180, 169, 190, 114, 38, 31, 191, 65, 188, 198, 62, 149, 90, 26, 189,
                    2, 102, 165, 63, 100, 179, 189, 191, 21, 210, 40, 63, 164, 247, 109, 191,
                    40, 125, 26, 190, 51, 155, 54, 191, 211, 186, 236, 58, 224, 106, 78, 191,
                    219, 179, 205, 190, 20, 113, 150, 190, 145, 144, 75, 62, 220, 126, 205, 62,
                    95, 173, 8, 191, 148, 28, 166, 63, 31, 67, 169, 190, 31, 148, 204, 191,
                    122, 230, 170, 190, 197, 250, 12, 64, 44, 109, 144, 61, 254, 194, 150, 63,
                    172, 122, 157, 63, 242, 63, 131, 190, 47, 109, 80, 191, 226, 241, 156, 61,
                    7, 48, 139, 191, 108, 122, 181, 190, 110, 184, 14, 63, 174, 33, 169, 191,
                    203, 235, 64, 63, 126, 127, 129, 62, 98, 29, 57, 63, 209, 41, 255, 62,
                    207, 55, 159, 190, 6, 6, 192, 62, 8, 10, 216, 190, 107, 252, 231, 62,
                    17, 54, 196, 190, 0, 162, 247, 190, 83, 41, 67, 191, 158, 126, 225, 189,
                    69, 179, 122, 191, 181, 158, 138, 61, 41, 129, 165, 62, 253, 87, 201, 191,
                    28, 232, 3, 192, 169, 241, 90, 192, 181, 72, 72, 191, 84, 4, 249, 62,
                    206, 26, 152, 62, 252, 204, 233, 191, 124, 55, 212, 190, 150, 192, 89, 191,
                    13, 88, 97, 190, 224, 79, 37, 190, 142, 15, 186, 190, 250, 35, 243, 190,
                    144, 45, 165, 189, 59, 225, 195, 62, 249, 242, 21, 190, 102, 207, 197, 63,
                    240, 112, 207, 190, 102, 194, 8, 191, 73, 19, 62, 191, 31, 22, 167, 62,
                    239, 249, 63, 191, 233, 88, 246, 62, 136, 148, 247, 190, 190, 162, 172, 190,
                    62, 220, 129, 190, 86, 221, 189, 190, 118, 47, 65, 190, 21, 81, 51, 191,
                    33, 177, 54, 190, 196, 45, 245, 189, 4, 103, 192, 63, 128, 57, 227, 189,
                    105, 148, 34, 190, 186, 67, 16, 188, 151, 239, 89, 61, 253, 186, 6, 190,
                    31, 104, 115, 62, 165, 112, 143, 189, 29, 248, 0, 190, 181, 96, 183, 62,
                    167, 250, 117, 62, 149, 219, 141, 62, 120, 87, 102, 61, 90, 68, 83, 189,
                    239, 214, 11, 62, 35, 211, 124, 190, 212, 6, 42, 63, 118, 22, 66, 189,
                    51, 217, 39, 191, 193, 169, 6, 189, 82, 75, 133, 190, 139, 62, 166, 190,
                    7, 189, 110, 188, 3, 132, 8, 62, 128, 212, 132, 62, 163, 44, 228, 190,
                    80, 157, 76, 190, 29, 5, 198, 189, 101, 195, 55, 62, 117, 205, 198, 61,
                    186, 147, 144, 60, 134, 246, 14, 191, 82, 15, 219, 62, 167, 14, 160, 189,
                    217, 226, 252, 190, 235, 61, 120, 63, 15, 206, 132, 62, 207, 83, 21, 191,
                    210, 237, 198, 190, 98, 149, 18, 64, 106, 223, 73, 191, 150, 189, 165, 62,
                    224, 61, 61, 63, 148, 125, 130, 191, 3, 251, 82, 63, 121, 115, 19, 191,
                    36, 22, 40, 188, 20, 174, 197, 61, 100, 183, 58, 63, 227, 32, 7, 188,
                    0, 76, 134, 188, 102, 218, 134, 62, 60, 250, 146, 62, 245, 70, 163, 61,
                    154, 111, 35, 62, 84, 48, 205, 188, 196, 136, 186, 189, 114, 223, 5, 62,
                    108, 69, 182, 190, 185, 222, 28, 191, 138, 236, 101, 190, 3, 169, 176, 189,
                    51, 36, 144, 190, 148, 128, 54, 190, 82, 2, 198, 62, 174, 147, 23, 61,
                    178, 48, 100, 62, 111, 27, 134, 61, 92, 15, 114, 63, 175, 146, 65, 63,
                    171, 183, 32, 191, 110, 81, 186, 190, 7, 188, 231, 62, 173, 51, 46, 191,
                    0, 148, 21, 191, 136, 214, 189, 63, 75, 31, 80, 62, 97, 51, 132, 63,
                    143, 22, 95, 63, 113, 248, 203, 188, 61, 195, 221, 191, 116, 186, 21, 63,
                    79, 132, 4, 190, 175, 43, 105, 61, 2, 244, 38, 191, 92, 36, 193, 190,
                    75, 2, 19, 62, 67, 227, 177, 188, 188, 31, 168, 61, 104, 132, 44, 62,
                    142, 24, 153, 190, 171, 229, 183, 62, 82, 64, 168, 189, 230, 22, 142, 190,
                    146, 25, 250, 189, 176, 198, 140, 190, 54, 90, 127, 63, 236, 215, 8, 190,
                    242, 82, 173, 191, 12, 194, 252, 190, 42, 83, 9, 63, 244, 184, 167, 62,
                    109, 0, 104, 63, 62, 59, 82, 61, 7, 252, 17, 191, 93, 170, 21, 191,
                    208, 187, 199, 62, 3, 238, 2, 63, 88, 160, 173, 62, 114, 115, 187, 62,
                    192, 156, 97, 191, 191, 227, 18, 63, 136, 18, 197, 62, 132, 104, 171, 190,
                    167, 139, 46, 62, 44, 244, 185, 190, 9, 206, 222, 191, 234, 203, 44, 190,
                    190, 110, 197, 190, 141, 44, 31, 62, 164, 168, 91, 190, 98, 75, 122, 61,
                    120, 60, 47, 62, 7, 83, 140, 190, 124, 201, 148, 62, 5, 107, 79, 61,
                    104, 110, 197, 188, 201, 180, 16, 63, 18, 49, 15, 189, 38, 94, 142, 190,
                    111, 156, 172, 62, 201, 199, 54, 63, 46, 5, 68, 63, 51, 140, 10, 190,
                    119, 114, 243, 189, 68, 239, 145, 62, 240, 20, 171, 190, 245, 192, 200, 190,
                    139, 164, 142, 187, 40, 120, 100, 187, 105, 92, 226, 62, 196, 111, 74, 188,
                    93, 58, 60, 190, 3, 191, 16, 62, 160, 234, 145, 62, 27, 242, 45, 190,
                    62, 244, 74, 190, 254, 172, 39, 191, 90, 161, 128, 190, 15, 128, 119, 190,
                    99, 167, 210, 61, 217, 37, 144, 191, 130, 39, 137, 63, 175, 200, 45, 62,
                    242, 137, 24, 191, 12, 59, 166, 63, 6, 3, 97, 191, 126, 114, 206, 190,
                    126, 226, 18, 63, 110, 98, 66, 191, 161, 111, 77, 63, 10, 151, 163, 63,
                    123, 228, 165, 189, 112, 158, 119, 62, 100, 179, 182, 190, 8, 194, 33, 61,
                    72, 85, 11, 190, 62, 197, 186, 62, 101, 99, 198, 62, 161, 217, 7, 63,
                    184, 196, 89, 63, 246, 251, 190, 189, 92, 118, 235, 189, 159, 77, 58, 61,
                    162, 216, 156, 189, 36, 162, 158, 189, 245, 14, 137, 189, 2, 189, 41, 60,
                    49, 134, 31, 191, 8, 56, 67, 61, 1, 182, 255, 61, 195, 46, 86, 190,
                    63, 204, 56, 62, 235, 205, 93, 190, 214, 142, 69, 190, 173, 120, 56, 62,
                    139, 21, 3, 190, 130, 176, 41, 191, 170, 109, 182, 62, 36, 234, 108, 62,
                    97, 233, 137, 62, 147, 212, 94, 190, 163, 9, 7, 191, 158, 89, 190, 61,
                    132, 209, 169, 189, 84, 191, 83, 189, 139, 201, 198, 190, 58, 249, 163, 61,
                    249, 51, 125, 62, 84, 118, 76, 190, 180, 254, 31, 190, 4, 171, 136, 61,
                    179, 16, 157, 62, 103, 128, 146, 62, 154, 104, 25, 189, 143, 12, 185, 60,
                    91, 203, 10, 190, 85, 179, 225, 187, 10, 27, 109, 190, 116, 128, 237, 188,
                    147, 11, 47, 62, 192, 34, 74, 190, 254, 80, 16, 191, 163, 10, 14, 62,
                    61, 16, 164, 188, 29, 246, 208, 60, 146, 111, 182, 189, 40, 236, 178, 189,
                    65, 111, 7, 189, 122, 98, 123, 190, 222, 13, 142, 189, 14, 166, 68, 189,
                    212, 172, 154, 60, 168, 216, 79, 61, 57, 214, 245, 61, 25, 72, 68, 189,
                    246, 204, 152, 63, 63, 31, 144, 190, 201, 98, 25, 190, 122, 229, 139, 63,
                    68, 129, 254, 191, 83, 70, 183, 63, 88, 79, 144, 190, 62, 127, 136, 190,
                    75, 4, 152, 62, 6, 127, 170, 191, 39, 54, 49, 63, 217, 118, 119, 63,
                    30, 64, 148, 63, 166, 171, 148, 63, 100, 88, 84, 191, 153, 36, 109, 191,
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
                    177, 253, 91, 62, 118, 211, 44, 64, 79, 0, 37, 63, 124, 152, 143, 63,
                    146, 179, 71, 64, 89, 185, 78, 62, 216, 151, 236, 190, 106, 82, 43, 191,
                    17, 154, 110, 191, 223, 227, 235, 191, 48, 65, 151, 191, 7, 188, 159, 63,
                    69, 192, 117, 62, 178, 119, 21, 191, 5, 28, 97, 63, 183, 206, 174, 60,
                    189, 45, 120, 189, 47, 143, 47, 63, 122, 32, 130, 63, 135, 135, 165, 62,
                    210, 107, 201, 190, 216, 17, 238, 61, 198, 245, 236, 190, 126, 133, 248, 62,
                    203, 36, 169, 62, 141, 67, 32, 63, 185, 193, 17, 191, 233, 190, 49, 63,
                    57, 123, 211, 60, 225, 86, 177, 190, 88, 75, 96, 63, 69, 20, 84, 191,
                    46, 187, 183, 61, 110, 203, 196, 61, 169, 115, 165, 189, 81, 250, 204, 190,
                    20, 146, 236, 62, 205, 75, 245, 189, 173, 198, 28, 191, 195, 229, 169, 62,
                    187, 16, 15, 63, 72, 79, 171, 191, 214, 24, 150, 62, 134, 181, 151, 190,
                    13, 113, 105, 189, 142, 58, 131, 190, 251, 91, 147, 190, 161, 117, 136, 62,
                    8, 81, 49, 191, 109, 46, 199, 62, 82, 140, 143, 191, 247, 89, 67, 63,
                    96, 1, 109, 63, 54, 247, 143, 191, 250, 249, 143, 191, 252, 200, 221, 62,
                    210, 242, 167, 191, 86, 71, 202, 190, 44, 17, 7, 63, 94, 87, 114, 62,
                    67, 193, 165, 61, 35, 161, 241, 191, 26, 74, 99, 191, 172, 49, 165, 63,
                    26, 221, 47, 188, 167, 131, 101, 191, 53, 149, 154, 189, 179, 143, 139, 191,
                    254, 51, 61, 191, 4, 194, 13, 62, 83, 90, 77, 191, 75, 76, 171, 190,
                    188, 52, 235, 190, 254, 54, 218, 191, 222, 102, 40, 63, 1, 251, 39, 62,
                    120, 168, 100, 62, 84, 87, 75, 191, 130, 87, 176, 191, 137, 198, 248, 190,
                    164, 166, 246, 63, 121, 203, 53, 64, 101, 153, 27, 191, 228, 152, 217, 63,
                    17, 168, 99, 63, 138, 115, 80, 63, 30, 82, 27, 64, 39, 67, 204, 63,
                    27, 50, 81, 63, 138, 119, 158, 63, 155, 202, 242, 190, 206, 155, 20, 62,
                    149, 220, 147, 190, 230, 37, 165, 63, 241, 163, 29, 61, 197, 149, 129, 63,
                    80, 88, 154, 190, 208, 233, 175, 61, 58, 91, 47, 191, 184, 187, 148, 63,
                    166, 85, 17, 63, 187, 253, 169, 63, 244, 101, 72, 190, 12, 217, 157, 63,
                    150, 136, 215, 190, 13, 26, 52, 64, 139, 152, 213, 63, 65, 157, 38, 63,
                    28, 167, 236, 62, 148, 186, 100, 191, 37, 62, 212, 190, 194, 224, 228, 62,
                    173, 161, 131, 190, 254, 67, 11, 64, 206, 68, 192, 63, 97, 39, 93, 192,
                    242, 181, 76, 63, 100, 50, 216, 63, 171, 218, 203, 191, 101, 150, 8, 192,
                    251, 142, 195, 191, 17, 239, 117, 191, 165, 101, 202, 63, 68, 190, 108, 62,
                    160, 138, 219, 191, 211, 203, 48, 190, 35, 153, 227, 188, 177, 141, 237, 191,
                    170, 84, 227, 190, 155, 106, 181, 62, 227, 146, 17, 189, 85, 41, 151, 63,
                    203, 222, 205, 62, 182, 57, 154, 191, 248, 94, 43, 63, 96, 233, 35, 64,
                    27, 38, 197, 61, 64, 7, 142, 63, 161, 251, 252, 188, 26, 236, 63, 191,
                    220, 194, 149, 63, 192, 217, 197, 190, 154, 66, 130, 191, 175, 247, 137, 63,
                    131, 237, 243, 190, 74, 212, 55, 191, 149, 122, 67, 63, 155, 219, 185, 60,
                    186, 65, 94, 191, 217, 20, 168, 190, 17, 159, 96, 191, 17, 166, 49, 190,
                    140, 30, 34, 63, 129, 63, 206, 63, 1, 122, 227, 62, 74, 252, 235, 63,
                    136, 27, 47, 63, 84, 120, 123, 63, 101, 73, 191, 189, 63, 154, 180, 62,
                    205, 135, 49, 190, 212, 176, 129, 191, 123, 224, 201, 62, 13, 241, 128, 191,
                    138, 57, 125, 61, 64, 88, 141, 63, 20, 165, 250, 62, 56, 192, 42, 62,
                    225, 98, 141, 191, 197, 110, 125, 191, 150, 16, 187, 63, 47, 111, 180, 191,
                    6, 237, 191, 63, 37, 5, 3, 64, 109, 151, 35, 191, 252, 106, 189, 190,
                    0, 161, 48, 62, 185, 24, 84, 190, 4, 237, 92, 63, 36, 125, 131, 189,
                    106, 112, 172, 190, 26, 199, 154, 62, 156, 230, 223, 61, 109, 100, 19, 63,
                    116, 137, 102, 190, 42, 88, 237, 190, 167, 127, 56, 191, 230, 180, 232, 63,
                    38, 72, 115, 191, 65, 252, 134, 63, 120, 104, 4, 62, 155, 151, 250, 190,
                    141, 225, 114, 190, 47, 195, 243, 62, 207, 133, 139, 63, 211, 129, 113, 191,
                    138, 50, 228, 190, 28, 76, 179, 62, 95, 41, 14, 191, 51, 171, 46, 191,
                    32, 152, 130, 62, 111, 243, 23, 62, 84, 76, 56, 63, 23, 166, 56, 191,
                    75, 123, 105, 63, 98, 158, 154, 191, 178, 24, 83, 191, 7, 77, 52, 191,
                    48, 255, 56, 189, 235, 115, 177, 62, 177, 254, 118, 63, 146, 40, 32, 62,
                    28, 153, 93, 63, 41, 240, 49, 61, 20, 122, 42, 191, 189, 19, 156, 61,
                    38, 190, 75, 61, 225, 83, 7, 64, 17, 219, 233, 63, 216, 98, 98, 191,
                    14, 110, 102, 63, 184, 115, 166, 191, 136, 109, 146, 191, 145, 68, 47, 191,
                    221, 166, 208, 190, 33, 15, 240, 62, 52, 90, 42, 190, 65, 13, 106, 63,
                    66, 130, 72, 191, 19, 64, 76, 63, 228, 254, 51, 62, 226, 85, 14, 191,
                    102, 160, 37, 191, 13, 100, 231, 191, 48, 30, 41, 191, 29, 29, 19, 62,
                    106, 26, 1, 192, 98, 151, 156, 190, 245, 112, 21, 63, 185, 150, 102, 188,
                    86, 91, 212, 63, 47, 104, 65, 62, 153, 216, 115, 191, 75, 142, 53, 64,
                    49, 110, 79, 63, 133, 3, 56, 192, 180, 108, 11, 189, 87, 130, 74, 62,
                    220, 98, 185, 63, 174, 51, 129, 58, 33, 225, 58, 62, 163, 140, 120, 191,
                    182, 109, 88, 63, 2, 235, 214, 62, 185, 146, 144, 191, 34, 150, 17, 64,
                    160, 121, 85, 191, 99, 216, 76, 191, 74, 148, 121, 188, 140, 17, 177, 63,
                    32, 13, 6, 63, 225, 0, 146, 64, 249, 125, 149, 190, 118, 73, 28, 63,
                    248, 175, 239, 62, 200, 229, 230, 63, 145, 114, 133, 63, 180, 99, 156, 191,
                    63, 50, 83, 191, 186, 34, 151, 191, 14, 55, 214, 191, 221, 40, 248, 61,
                    234, 131, 81, 191, 124, 8, 210, 63, 135, 1, 69, 63, 160, 153, 32, 192,
                    125, 194, 24, 191, 151, 240, 1, 64, 137, 2, 161, 191, 30, 218, 139, 192,
                    215, 247, 23, 192, 190, 245, 148, 191, 224, 49, 83, 62, 246, 8, 162, 63,
                    100, 255, 133, 191, 244, 136, 5, 63, 144, 84, 239, 62, 253, 212, 142, 191,
                    110, 208, 73, 191, 164, 107, 165, 60, 18, 54, 61, 61, 62, 217, 161, 191,
                    157, 133, 61, 63, 238, 120, 162, 191, 9, 171, 146, 191, 24, 47, 9, 62,
                    52, 5, 233, 62, 96, 113, 28, 64, 71, 82, 67, 191, 134, 254, 184, 61,
                    172, 132, 177, 63, 28, 177, 188, 191, 105, 143, 152, 191, 174, 82, 12, 62,
                    188, 129, 49, 64, 117, 93, 225, 190, 80, 18, 52, 191, 173, 146, 35, 64,
                    211, 79, 154, 189, 34, 91, 13, 192, 79, 29, 39, 63, 210, 133, 76, 63,
                    66, 11, 133, 63, 233, 56, 81, 63, 73, 33, 204, 191, 204, 157, 128, 191,
                    119, 114, 139, 191, 179, 209, 163, 190, 102, 105, 214, 63, 108, 78, 225, 63,
                    33, 132, 5, 64, 166, 175, 68, 62, 36, 214, 20, 191, 184, 220, 45, 192,
                    159, 228, 51, 62, 234, 14, 39, 64, 155, 121, 188, 63, 38, 108, 177, 191,
                    67, 233, 115, 191, 60, 222, 218, 63, 223, 234, 97, 191, 173, 231, 246, 189,
                    251, 146, 244, 191, 240, 189, 126, 189, 115, 41, 35, 64, 198, 37, 24, 192,
                    96, 221, 7, 190, 151, 209, 160, 63, 117, 23, 46, 63, 149, 250, 245, 63,
                    83, 130, 136, 189, 236, 142, 117, 191, 196, 209, 247, 190, 193, 227, 62, 63,
                    238, 102, 102, 63, 164, 202, 105, 63, 23, 26, 77, 62, 217, 155, 101, 190,
                    223, 248, 195, 191, 32, 95, 5, 191, 224, 240, 51, 191, 241, 92, 128, 62,
                    165, 131, 12, 192, 66, 60, 27, 64, 46, 85, 20, 63, 103, 155, 7, 189,
                    110, 36, 17, 191, 167, 243, 50, 191, 150, 248, 138, 62, 119, 27, 22, 191,
                    130, 77, 81, 192, 230, 143, 125, 61, 255, 5, 15, 192, 146, 111, 132, 191,
                    65, 38, 236, 63, 74, 170, 95, 192, 207, 123, 49, 192, 164, 206, 133, 189,
                    20, 145, 77, 63, 152, 197, 148, 63, 106, 78, 6, 192, 154, 89, 4, 64,
                    162, 23, 5, 63, 243, 184, 1, 64, 122, 190, 158, 63, 138, 230, 113, 190,
                    114, 76, 146, 63, 65, 60, 94, 63, 29, 78, 37, 192, 173, 33, 153, 63,
                    187, 231, 60, 191, 203, 150, 216, 191, 104, 155, 29, 61, 129, 168, 243, 190,
                    227, 47, 52, 191, 206, 42, 237, 190, 36, 110, 25, 64, 9, 41, 119, 192,
                    152, 134, 57, 63, 247, 161, 136, 191, 74, 75, 208, 191, 139, 142, 72, 191,
                    94, 245, 252, 62, 101, 229, 132, 63, 119, 59, 244, 191, 47, 6, 212, 60,
                    131, 93, 203, 190, 36, 196, 25, 64, 178, 252, 103, 190, 244, 12, 5, 192,
                    227, 10, 190, 62, 66, 184, 218, 188, 74, 19, 121, 191, 234, 126, 12, 190,
                    108, 69, 119, 191, 87, 201, 17, 191, 138, 177, 120, 191, 202, 54, 175, 61,
                    242, 11, 87, 62, 101, 228, 217, 189, 183, 78, 73, 63, 66, 43, 249, 62,
                    227, 253, 19, 191, 195, 6, 101, 63, 107, 241, 128, 190, 111, 94, 202, 59,
                    108, 198, 98, 61, 148, 189, 226, 61, 93, 6, 4, 191, 71, 147, 27, 64,
                    233, 208, 171, 188, 168, 17, 229, 191, 3, 254, 228, 188, 85, 221, 2, 63,
                    87, 32, 242, 62, 252, 189, 32, 191, 180, 95, 156, 190, 253, 89, 11, 63,
                    54, 65, 90, 190, 37, 3, 28, 191, 68, 31, 147, 62, 26, 232, 125, 190,
                    206, 74, 141, 191, 160, 151, 183, 63, 32, 219, 227, 63, 43, 4, 119, 192,
                    187, 234, 248, 63, 213, 209, 133, 63, 236, 143, 60, 63, 80, 19, 132, 191,
                    153, 80, 144, 190, 47, 217, 110, 63, 131, 130, 163, 191, 81, 208, 10, 190,
                    168, 119, 129, 190, 1, 4, 40, 191, 52, 225, 146, 191, 209, 216, 28, 62,
                    144, 129, 172, 62, 242, 185, 186, 63, 243, 0, 233, 63, 239, 7, 26, 192,
                    117, 190, 131, 62, 255, 36, 24, 63, 23, 17, 25, 191, 84, 202, 18, 61,
                    73, 224, 138, 191, 7, 115, 101, 62, 25, 184, 3, 63, 209, 88, 104, 63,
                    133, 1, 9, 63, 177, 231, 146, 63, 244, 171, 73, 63, 223, 151, 42, 192,
                    109, 31, 186, 191, 40, 64, 240, 191, 8, 40, 182, 191, 21, 156, 10, 64,
                    68, 151, 168, 191, 104, 151, 103, 63, 195, 8, 241, 62, 145, 52, 19, 62,
                    36, 248, 43, 192, 106, 167, 65, 63, 19, 137, 2, 189, 89, 74, 206, 190,
                    11, 200, 60, 62, 154, 33, 3, 191, 22, 189, 142, 62, 111, 172, 204, 61,
                    164, 155, 207, 63, 78, 171, 23, 191, 108, 143, 235, 190, 103, 252, 224, 191,
                    33, 251, 200, 191, 221, 215, 152, 62, 29, 147, 148, 191, 45, 121, 18, 190,
                    75, 148, 7, 63, 30, 243, 44, 63, 98, 165, 123, 191, 13, 133, 15, 64,
                    92, 141, 77, 63, 96, 229, 187, 63, 156, 36, 179, 190, 238, 45, 199, 189,
                    31, 165, 34, 61, 85, 226, 203, 62, 225, 17, 104, 63, 245, 225, 215, 63,
                    53, 230, 91, 190, 149, 199, 253, 191, 136, 105, 109, 63, 167, 132, 161, 63,
                    192, 34, 102, 63, 151, 154, 176, 191, 205, 48, 7, 64, 225, 181, 10, 63,
                    225, 210, 46, 63, 127, 32, 224, 191, 241, 47, 30, 63, 173, 239, 204, 62,
                    88, 81, 145, 191, 106, 122, 17, 63, 38, 233, 152, 191, 16, 100, 121, 191,
                    25, 202, 54, 191, 167, 56, 109, 190, 171, 214, 95, 62, 157, 174, 164, 63,
                    99, 132, 208, 189, 31, 105, 78, 63, 140, 14, 56, 63, 67, 129, 9, 192,
                    145, 102, 244, 63, 232, 90, 211, 190, 48, 1, 243, 191, 54, 207, 94, 62,
                    219, 140, 210, 189, 219, 215, 60, 191, 75, 220, 154, 191, 175, 239, 158, 188,
                    168, 160, 192, 61, 35, 233, 15, 62, 119, 163, 52, 191, 68, 46, 21, 190,
                    230, 232, 190, 190, 205, 215, 10, 191, 42, 231, 12, 63, 83, 51, 88, 62,
                    201, 238, 183, 58, 196, 6, 11, 190, 61, 57, 46, 190, 40, 136, 161, 62,
                    68, 105, 163, 61, 235, 244, 54, 63, 113, 135, 242, 62, 173, 64, 163, 62,
                    112, 245, 106, 63, 29, 50, 171, 190, 101, 99, 25, 62, 116, 112, 150, 190,
                    188, 84, 87, 190, 79, 16, 14, 62, 206, 70, 122, 190, 110, 255, 8, 62,
                    12, 146, 145, 191, 105, 21, 157, 62, 114, 208, 124, 61, 64, 71, 14, 190,
                    166, 174, 105, 190, 88, 185, 105, 63, 4, 149, 38, 191, 117, 127, 236, 63,
                    194, 123, 53, 63, 16, 147, 29, 191, 91, 130, 70, 189, 71, 246, 109, 189,
                    184, 190, 222, 62, 107, 195, 64, 190, 146, 221, 132, 62, 6, 242, 69, 189,
                    17, 97, 103, 191, 28, 48, 248, 191, 33, 141, 163, 63, 28, 139, 148, 63,
                    192, 219, 225, 189, 135, 101, 235, 62, 248, 212, 198, 190, 45, 190, 10, 63,
                    191, 84, 127, 191, 150, 61, 51, 189, 66, 211, 14, 191, 64, 123, 72, 190,
                    232, 6, 193, 61, 91, 170, 210, 62, 154, 55, 34, 63, 135, 248, 177, 190,
                    66, 5, 202, 62, 54, 171, 40, 189, 196, 224, 66, 190, 51, 46, 166, 61,
                    178, 249, 142, 190, 195, 16, 75, 63, 46, 146, 147, 63, 239, 36, 12, 192,
                    61, 124, 170, 191, 45, 193, 239, 63, 184, 161, 227, 188, 93, 40, 146, 190,
                    147, 221, 61, 191, 81, 65, 31, 62, 124, 191, 159, 190, 192, 222, 20, 191,
                    38, 191, 204, 191, 226, 7, 19, 63, 165, 154, 228, 190, 149, 24, 2, 192,
                    144, 35, 2, 190, 26, 85, 9, 61, 182, 109, 79, 191, 245, 54, 193, 187,
                    23, 148, 26, 190, 98, 88, 95, 63, 82, 40, 193, 189, 136, 201, 71, 191,
                    91, 233, 221, 189, 130, 116, 94, 190, 245, 86, 111, 63, 153, 41, 68, 63,
                    96, 14, 152, 186, 162, 78, 175, 62, 199, 175, 4, 62, 193, 59, 51, 63,
                    60, 20, 14, 191, 40, 25, 99, 191, 182, 124, 161, 188, 12, 27, 236, 190,
                    211, 220, 4, 64, 104, 18, 73, 191, 20, 158, 34, 191, 100, 61, 140, 191,
                    120, 169, 22, 191, 136, 255, 23, 191, 108, 246, 191, 188, 71, 65, 190, 190,
                    224, 161, 191, 191, 200, 116, 175, 190, 46, 254, 23, 63, 41, 45, 73, 61,
                    99, 164, 26, 62, 229, 163, 61, 63, 144, 45, 39, 63, 51, 117, 136, 62,
                    241, 7, 182, 63, 226, 96, 134, 191, 20, 78, 168, 63, 211, 21, 135, 190,
                    54, 25, 116, 190, 1, 204, 175, 63, 131, 17, 91, 62, 136, 38, 54, 191,
                    131, 110, 23, 63, 185, 215, 154, 189, 35, 61, 12, 192, 242, 205, 65, 63,
                    12, 70, 23, 191, 211, 122, 176, 63, 180, 38, 198, 61, 231, 10, 57, 191,
                    116, 222, 49, 63, 204, 31, 50, 63, 26, 128, 27, 191, 100, 198, 4, 191,
                    131, 2, 118, 189, 47, 221, 196, 191, 193, 195, 59, 63, 161, 106, 67, 62,
                    179, 209, 18, 62, 30, 164, 145, 190, 104, 211, 0, 191, 188, 183, 228, 190,
                    101, 69, 124, 191, 92, 88, 187, 187, 24, 229, 90, 60, 184, 114, 4, 191,
                    49, 82, 193, 191, 95, 51, 174, 189, 42, 139, 109, 191, 194, 41, 144, 191,
                    207, 34, 27, 192, 154, 241, 35, 191, 242, 4, 180, 191, 192, 211, 162, 190,
                    171, 80, 50, 63, 118, 61, 193, 190, 237, 0, 6, 192, 135, 221, 128, 189,
                    84, 136, 57, 188, 14, 203, 94, 191, 108, 192, 70, 191, 72, 251, 136, 62,
                    164, 154, 166, 62, 56, 178, 32, 62, 195, 141, 135, 191, 253, 121, 214, 189,
                    29, 40, 171, 189, 252, 75, 127, 190, 156, 98, 144, 63, 3, 76, 83, 190,
                    31, 128, 203, 61, 93, 249, 72, 62, 67, 176, 183, 62, 117, 49, 78, 190,
                    232, 83, 90, 62, 234, 154, 214, 190, 218, 138, 156, 62, 233, 179, 197, 188,
                    34, 122, 152, 191, 0, 38, 33, 63, 196, 21, 181, 63, 47, 83, 102, 190,
                    219, 253, 158, 62, 218, 220, 24, 63, 170, 59, 139, 61, 145, 126, 242, 62,
                    56, 7, 198, 190, 160, 212, 211, 190, 106, 96, 96, 191, 98, 73, 61, 191,
                    171, 171, 178, 62, 144, 185, 32, 190, 37, 170, 172, 190, 55, 51, 188, 188,
                    74, 252, 20, 63, 5, 213, 186, 190, 144, 216, 179, 62, 254, 231, 44, 61,
                    109, 111, 139, 62, 20, 241, 48, 63, 70, 209, 0, 63, 199, 3, 36, 190,
                    110, 17, 180, 62, 246, 107, 60, 191, 101, 188, 23, 191, 190, 165, 191, 61,
                    167, 95, 246, 190, 169, 38, 141, 63, 96, 181, 253, 62, 219, 68, 93, 191,
                    204, 114, 154, 190, 199, 127, 43, 59, 198, 23, 58, 190, 189, 13, 225, 61,
                    213, 194, 38, 190, 206, 19, 133, 63, 89, 100, 235, 62, 105, 197, 27, 190,
                    30, 72, 210, 62, 86, 132, 18, 189, 179, 149, 43, 191, 30, 184, 134, 190,
                    212, 197, 5, 63, 76, 204, 149, 191, 15, 52, 44, 191, 61, 187, 244, 63,
                    228, 3, 186, 60, 73, 124, 149, 191, 82, 155, 69, 63, 142, 175, 178, 62,
                    254, 39, 194, 62, 179, 100, 181, 62, 37, 211, 17, 191, 170, 136, 31, 63,
                    247, 71, 183, 190, 216, 142, 227, 188, 26, 154, 208, 190, 19, 219, 179, 191,
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
                    244, 248, 29, 191, 37, 118, 68, 63, 131, 235, 157, 62, 63, 188, 175, 63,
                    155, 254, 209, 62, 247, 64, 84, 63, 106, 221, 71, 190, 233, 33, 7, 192,
                    121, 30, 4, 64, 44, 178, 12, 62, 122, 225, 6, 63, 214, 19, 124, 190,
                    82, 112, 93, 190, 49, 56, 250, 189, 21, 61, 128, 190, 237, 235, 128, 63,
                    194, 233, 60, 191, 138, 191, 126, 192, 115, 2, 149, 191, 131, 245, 77, 64,
                    52, 198, 132, 192, 205, 20, 32, 64, 221, 5, 223, 191, 228, 136, 192, 190,
                    221, 53, 107, 191, 194, 226, 180, 190, 77, 216, 27, 64, 236, 175, 91, 192,
                    21, 185, 9, 192, 163, 199, 12, 192, 151, 170, 240, 191, 26, 57, 4, 64,
                    74, 123, 206, 63, 224, 181, 191, 61, 88, 88, 60, 191, 107, 155, 68, 64,
                    236, 59, 107, 190, 230, 152, 212, 191, 173, 218, 176, 63, 93, 12, 18, 63,
                    235, 63, 165, 63, 64, 73, 17, 63, 247, 252, 158, 190, 105, 34, 24, 62,
                    200, 219, 252, 62, 161, 132, 213, 190, 231, 235, 164, 62, 242, 55, 208, 63,
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
                    47, 164, 243, 190, 123, 166, 61, 63, 85, 63, 232, 189, 209, 75, 190, 63,
                    47, 184, 121, 61, 72, 239, 211, 62, 232, 43, 243, 190, 108, 216, 7, 192,
                    197, 12, 236, 63, 241, 80, 54, 190, 114, 235, 62, 62, 187, 21, 125, 189,
                    161, 31, 231, 190, 70, 249, 226, 189, 173, 59, 173, 190, 149, 205, 162, 63,
                    245, 118, 112, 191, 230, 159, 116, 192, 243, 185, 181, 191, 68, 106, 59, 64,
                    65, 238, 129, 192, 87, 33, 15, 64, 238, 112, 216, 191, 88, 56, 171, 190,
                    30, 56, 152, 191, 185, 229, 237, 190, 213, 224, 3, 64, 91, 173, 77, 192,
                    101, 169, 22, 192, 68, 122, 8, 192, 33, 94, 249, 191, 84, 107, 237, 63,
                    171, 170, 36, 187, 140, 38, 39, 189, 183, 185, 152, 190, 190, 29, 223, 63,
                    144, 43, 208, 62, 197, 55, 159, 191, 223, 78, 149, 63, 152, 200, 232, 63,
                    192, 217, 71, 63, 18, 199, 131, 62, 84, 81, 153, 191, 161, 219, 131, 190,
                    234, 184, 155, 190, 53, 118, 144, 61, 137, 236, 102, 191, 50, 201, 141, 63,
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
    namespace layer_3 {
        namespace weights {
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            namespace parameters_memory {
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {
                    103, 111, 14, 64, 33, 77, 149, 192, 31, 68, 18, 192, 248, 20, 154, 62,
                    216, 15, 46, 63, 183, 115, 15, 192, 100, 20, 139, 62, 34, 26, 215, 63,
                    180, 242, 115, 64, 108, 180, 204, 191, 137, 59, 138, 62, 87, 122, 250, 62,
                    50, 142, 7, 64, 68, 81, 168, 63, 117, 42, 228, 64, 234, 204, 1, 64,
                    183, 0, 50, 64, 201, 22, 139, 64, 180, 239, 249, 191, 118, 192, 62, 64,
                    78, 78, 118, 64, 212, 223, 231, 191, 199, 147, 96, 64, 60, 195, 150, 63,
                    130, 108, 165, 63, 185, 24, 73, 64, 18, 201, 53, 190, 79, 142, 37, 63,
                    45, 180, 77, 64, 223, 26, 195, 192, 141, 165, 35, 191, 26, 79, 49, 64,
                    137, 95, 60, 64, 126, 184, 205, 63, 158, 96, 252, 191, 116, 104, 148, 63,
                    63, 174, 240, 191, 198, 72, 54, 192, 114, 172, 139, 64, 228, 0, 110, 63,
                    21, 19, 56, 64, 105, 224, 182, 63, 95, 173, 73, 63, 215, 199, 6, 64,
                    194, 5, 54, 192, 179, 123, 19, 63, 174, 254, 153, 64, 229, 28, 228, 63,
                    37, 43, 18, 64, 152, 181, 170, 192, 170, 136, 144, 192, 121, 130, 9, 64,
                    157, 52, 40, 192, 219, 100, 231, 191, 43, 191, 208, 63, 186, 52, 212, 63,
                    89, 86, 228, 63, 54, 124, 30, 64, 193, 80, 219, 191, 23, 218, 121, 192,
                    38, 88, 144, 63, 76, 62, 215, 192, 12, 198, 155, 62, 24, 172, 87, 64,
                    47, 23, 241, 63, 7, 253, 160, 64, 82, 127, 191, 191, 10, 214, 24, 64,
                    10, 158, 246, 191, 22, 70, 189, 191, 241, 165, 8, 64, 114, 71, 61, 64,
                    158, 248, 204, 62, 21, 234, 42, 64, 97, 208, 11, 190, 254, 249, 134, 62,
                    209, 168, 126, 63, 32, 129, 72, 191, 229, 8, 186, 192, 209, 236, 236, 63,
                    77, 5, 90, 63, 76, 210, 72, 192, 69, 201, 164, 191, 13, 183, 170, 62,
                    187, 176, 160, 192, 162, 180, 83, 191, 88, 186, 155, 188, 189, 61, 44, 64,
                    58, 53, 1, 64, 6, 114, 88, 62, 179, 203, 146, 191, 228, 41, 156, 191,
                    83, 33, 26, 63, 56, 98, 68, 64, 58, 173, 213, 63, 56, 215, 17, 63,
                    16, 128, 73, 63, 48, 116, 49, 192, 102, 170, 168, 191, 172, 199, 0, 64,
                    243, 168, 117, 191, 170, 214, 150, 190, 191, 103, 0, 189, 246, 214, 33, 64,
                    255, 76, 69, 63, 180, 6, 131, 62, 197, 154, 3, 192, 177, 195, 166, 191,
                    171, 234, 203, 64, 84, 124, 178, 190, 74, 98, 73, 192, 116, 173, 189, 63,
                    88, 155, 199, 63, 199, 14, 119, 64, 39, 204, 173, 190, 251, 149, 134, 63,
                    219, 213, 64, 191, 158, 72, 164, 191, 96, 18, 210, 63, 13, 29, 60, 64,
                    214, 12, 171, 63, 16, 78, 122, 191, 47, 50, 135, 189, 6, 20, 104, 64,
                    189, 61, 50, 64, 193, 65, 98, 64, 126, 182, 30, 63, 169, 5, 10, 63,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 8, 16>;
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
                    103, 1, 8, 64, 90, 167, 35, 64, 251, 178, 224, 63, 71, 194, 46, 64,
                    249, 253, 215, 63, 35, 216, 112, 63, 139, 51, 238, 63, 83, 31, 129, 63,
                };
                using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 8>;
                using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                const CONTAINER_TYPE container = {(const float*)memory};
            }
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, unsigned long, parameters_memory::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {parameters_memory::container};
        }
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 8, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::IDENTITY, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal>;
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
    namespace layer_4 {
        using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
        using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::checkpoint::beta_mean::Configuration<TYPE_POLICY, unsigned long, 4>;
        using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::checkpoint::beta_mean::BindConfiguration<CONFIG>;
        using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 8>;
        using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::checkpoint::beta_mean::LayerForward<RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::checkpoint::beta_mean::Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>;
        const TYPE module = {};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory = {};
        template <typename T_TYPE = TYPE>
        const T_TYPE factory_function(){return T_TYPE{};}
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
        0, 0, 128, 191, 71, 88, 110, 191, 141, 176, 92, 191, 212, 8, 75, 191,
        26, 97, 57, 191, 97, 185, 39, 191, 168, 17, 22, 191, 238, 105, 4, 191,
        106, 132, 229, 190, 247, 52, 194, 190, 132, 229, 158, 190, 36, 44, 119, 190,
        62, 141, 48, 190, 177, 220, 211, 189, 204, 61, 13, 189, 204, 61, 13, 61,
        177, 220, 211, 61, 62, 141, 48, 62, 36, 44, 119, 62, 132, 229, 158, 62,
        247, 52, 194, 62, 106, 132, 229, 62, 238, 105, 4, 63, 168, 17, 22, 63,
        97, 185, 39, 63, 26, 97, 57, 63, 212, 8, 75, 63, 141, 176, 92, 63,
        71, 88, 110, 63, 0, 0, 128, 63,
    };
    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 30>;
    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
    const CONTAINER_TYPE container = {(const float*)memory};
}
namespace rl_tools::checkpoint::example::output {
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {
        177, 143, 234, 61, 152, 215, 239, 62, 42, 24, 175, 62, 98, 71, 210, 61,
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
