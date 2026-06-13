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
                    213, 171, 108, 64, 210, 253, 40, 64, 249, 218, 238, 191, 125, 225, 59, 189,
                    27, 79, 201, 191, 45, 38, 146, 63, 198, 125, 182, 63, 90, 208, 135, 188,
                    124, 58, 219, 191, 131, 111, 169, 191, 11, 183, 220, 63, 192, 42, 172, 189,
                    210, 231, 135, 191, 127, 128, 190, 63, 100, 196, 48, 190, 223, 32, 78, 64,
                    235, 169, 18, 64, 199, 111, 49, 191, 242, 60, 140, 63, 10, 189, 117, 190,
                    18, 144, 248, 62, 87, 69, 68, 61, 46, 57, 42, 61, 99, 230, 198, 188,
                    246, 187, 249, 188, 243, 130, 41, 191, 79, 159, 48, 62, 176, 212, 234, 189,
                    161, 19, 129, 59, 223, 218, 33, 61, 184, 76, 94, 63, 240, 131, 154, 62,
                    154, 79, 14, 63, 143, 124, 222, 190, 221, 32, 208, 62, 62, 253, 156, 191,
                    165, 178, 28, 63, 225, 55, 2, 190, 228, 223, 132, 191, 8, 29, 140, 63,
                    69, 128, 197, 63, 207, 88, 61, 190, 149, 53, 178, 63, 84, 209, 30, 63,
                    3, 106, 207, 189, 239, 2, 68, 191, 69, 12, 76, 191, 56, 65, 15, 191,
                    199, 191, 27, 192, 173, 172, 122, 191, 28, 38, 50, 62, 186, 64, 139, 62,
                    4, 164, 2, 191, 146, 88, 27, 190, 55, 192, 106, 61, 180, 62, 242, 62,
                    87, 84, 56, 63, 27, 28, 92, 191, 234, 98, 227, 190, 162, 24, 0, 61,
                    4, 34, 24, 63, 39, 61, 235, 63, 8, 254, 224, 192, 18, 231, 224, 61,
                    66, 30, 218, 61, 61, 78, 14, 63, 107, 202, 74, 63, 163, 152, 64, 190,
                    239, 191, 177, 63, 173, 10, 190, 190, 172, 154, 95, 191, 147, 178, 63, 190,
                    221, 163, 17, 191, 58, 7, 206, 191, 51, 126, 239, 61, 251, 145, 215, 60,
                    224, 246, 137, 63, 230, 49, 151, 192, 68, 115, 237, 190, 161, 76, 207, 62,
                    10, 123, 160, 189, 26, 105, 197, 189, 87, 169, 179, 61, 69, 146, 160, 61,
                    18, 143, 8, 188, 36, 11, 162, 190, 105, 31, 21, 63, 53, 18, 5, 191,
                    216, 64, 149, 62, 109, 178, 28, 62, 77, 71, 7, 192, 95, 241, 60, 191,
                    73, 212, 143, 190, 176, 248, 170, 59, 166, 26, 194, 190, 159, 206, 87, 192,
                    188, 131, 64, 62, 115, 146, 209, 60, 180, 140, 109, 191, 116, 30, 101, 64,
                    213, 10, 132, 63, 8, 188, 196, 61, 164, 14, 115, 64, 169, 157, 195, 62,
                    242, 184, 145, 189, 99, 67, 27, 192, 174, 235, 186, 190, 111, 215, 61, 191,
                    83, 74, 159, 62, 3, 18, 225, 191, 119, 221, 211, 63, 79, 19, 30, 62,
                    10, 155, 82, 190, 173, 17, 212, 60, 50, 8, 66, 189, 221, 180, 74, 61,
                    105, 105, 58, 61, 48, 207, 19, 60, 194, 165, 133, 190, 163, 42, 128, 62,
                    159, 204, 144, 62, 165, 38, 62, 64, 242, 154, 44, 64, 240, 67, 173, 189,
                    12, 38, 149, 190, 23, 181, 59, 190, 148, 99, 89, 63, 37, 39, 4, 189,
                    41, 22, 144, 64, 13, 231, 232, 62, 36, 51, 161, 192, 83, 143, 230, 61,
                    106, 177, 221, 62, 7, 231, 147, 192, 81, 2, 199, 187, 150, 16, 163, 62,
                    119, 234, 92, 64, 210, 118, 6, 64, 248, 59, 16, 192, 32, 226, 83, 190,
                    146, 85, 60, 63, 56, 68, 71, 188, 222, 25, 36, 190, 110, 73, 17, 188,
                    41, 111, 201, 60, 231, 232, 29, 62, 220, 212, 143, 190, 217, 3, 209, 62,
                    11, 190, 188, 62, 130, 143, 129, 187, 211, 212, 128, 64, 158, 228, 82, 191,
                    196, 122, 140, 190, 32, 9, 77, 62, 112, 47, 78, 62, 17, 76, 58, 64,
                    167, 82, 140, 190, 133, 85, 238, 189, 168, 105, 212, 190, 8, 90, 128, 192,
                    208, 194, 187, 190, 211, 34, 13, 190, 85, 204, 96, 192, 137, 151, 72, 62,
                    28, 124, 18, 190, 210, 74, 111, 64, 116, 253, 106, 191, 25, 215, 63, 191,
                    215, 156, 185, 190, 100, 79, 143, 63, 32, 253, 204, 62, 107, 199, 186, 188,
                    136, 152, 129, 62, 252, 47, 100, 187, 92, 165, 124, 59, 54, 213, 167, 189,
                    47, 79, 80, 62, 72, 85, 16, 191, 200, 29, 200, 61, 70, 197, 131, 190,
                    147, 200, 162, 63, 128, 53, 86, 63, 34, 13, 226, 62, 210, 44, 204, 62,
                    102, 226, 129, 192, 39, 227, 61, 63, 236, 105, 124, 64, 77, 54, 133, 190,
                    242, 201, 2, 64, 133, 67, 83, 191, 138, 184, 10, 192, 192, 186, 142, 60,
                    174, 178, 51, 191, 245, 105, 252, 191, 48, 216, 45, 190, 154, 136, 8, 63,
                    58, 86, 204, 63, 110, 193, 7, 191, 160, 70, 60, 61, 92, 211, 154, 62,
                    92, 70, 140, 64, 38, 98, 35, 190, 112, 17, 21, 61, 252, 83, 188, 59,
                    96, 177, 187, 61, 147, 38, 90, 189, 253, 139, 164, 190, 185, 249, 138, 190,
                    2, 163, 108, 62, 5, 127, 4, 191, 125, 224, 1, 191, 89, 176, 68, 63,
                    164, 89, 22, 63, 149, 233, 20, 190, 106, 137, 37, 191, 40, 98, 222, 63,
                    53, 54, 4, 64, 193, 158, 162, 190, 143, 157, 52, 190, 8, 21, 151, 191,
                    207, 2, 39, 191, 133, 126, 4, 190, 142, 250, 236, 191, 46, 35, 17, 189,
                    87, 63, 130, 62, 78, 103, 65, 63, 243, 159, 87, 62, 74, 79, 110, 191,
                    143, 115, 145, 191, 131, 219, 3, 189, 201, 42, 145, 63, 18, 71, 26, 190,
                    129, 193, 42, 189, 239, 9, 149, 61, 105, 134, 41, 190, 171, 115, 123, 191,
                    7, 151, 58, 61, 10, 14, 77, 62, 178, 41, 24, 63, 176, 29, 106, 61,
                    217, 173, 170, 190, 86, 240, 182, 189, 162, 115, 138, 60, 205, 32, 175, 61,
                    143, 19, 139, 62, 184, 31, 44, 64, 85, 5, 208, 63, 130, 147, 32, 62,
                    167, 63, 85, 191, 242, 164, 94, 192, 250, 216, 246, 62, 208, 36, 155, 61,
                    89, 3, 72, 192, 57, 6, 137, 63, 205, 149, 219, 189, 250, 231, 177, 63,
                    33, 252, 86, 190, 172, 180, 2, 63, 228, 144, 40, 63, 32, 168, 173, 63,
                    125, 234, 171, 189, 85, 210, 176, 191, 16, 113, 147, 191, 202, 161, 32, 190,
                    207, 66, 119, 61, 50, 50, 241, 190, 78, 23, 135, 62, 146, 237, 130, 60,
                    142, 28, 77, 61, 231, 109, 225, 60, 96, 94, 82, 191, 69, 4, 247, 62,
                    121, 63, 186, 63, 59, 96, 234, 188, 178, 145, 91, 189, 135, 23, 220, 191,
                    228, 103, 21, 191, 220, 225, 107, 62, 246, 243, 129, 192, 110, 67, 3, 64,
                    28, 75, 132, 64, 212, 211, 42, 60, 180, 28, 6, 64, 42, 193, 112, 64,
                    164, 121, 70, 189, 145, 133, 134, 191, 197, 155, 29, 190, 84, 145, 52, 191,
                    3, 101, 25, 64, 200, 200, 149, 191, 30, 247, 106, 191, 111, 51, 170, 190,
                    69, 187, 105, 190, 253, 10, 51, 188, 146, 201, 195, 61, 6, 91, 145, 191,
                    126, 15, 146, 62, 38, 94, 3, 190, 81, 15, 13, 191, 140, 129, 255, 60,
                    173, 153, 152, 191, 69, 243, 212, 191, 91, 253, 64, 191, 13, 180, 242, 190,
                    40, 168, 189, 60, 48, 246, 145, 190, 93, 202, 178, 62, 134, 186, 66, 190,
                    111, 10, 200, 60, 168, 240, 139, 190, 8, 170, 54, 191, 122, 179, 220, 190,
                    231, 26, 95, 61, 157, 180, 46, 62, 46, 69, 227, 62, 28, 221, 165, 191,
                    44, 128, 11, 191, 171, 32, 158, 62, 30, 191, 146, 191, 248, 11, 27, 64,
                    11, 147, 62, 63, 148, 232, 57, 62, 84, 39, 200, 62, 144, 2, 7, 59,
                    16, 179, 35, 60, 49, 46, 158, 63, 54, 44, 1, 63, 145, 60, 48, 191,
                    41, 179, 187, 190, 68, 171, 98, 63, 83, 207, 163, 61, 185, 250, 19, 64,
                    63, 80, 164, 63, 166, 162, 134, 186, 9, 168, 30, 64, 227, 136, 159, 62,
                    239, 78, 12, 192, 114, 244, 210, 189, 193, 80, 100, 64, 102, 154, 183, 191,
                    185, 218, 110, 192, 118, 17, 252, 61, 85, 12, 167, 189, 52, 54, 83, 192,
                    119, 213, 140, 189, 162, 219, 145, 63, 43, 7, 27, 64, 80, 125, 129, 191,
                    35, 79, 173, 190, 28, 42, 39, 63, 235, 231, 36, 192, 214, 9, 223, 188,
                    48, 29, 53, 188, 200, 41, 150, 188, 22, 251, 148, 187, 101, 154, 3, 62,
                    227, 123, 172, 190, 212, 102, 36, 63, 158, 98, 116, 62, 28, 40, 183, 190,
                    26, 76, 167, 63, 134, 227, 135, 63, 204, 150, 5, 192, 119, 109, 141, 190,
                    100, 6, 207, 63, 30, 210, 195, 63, 222, 2, 26, 191, 52, 226, 76, 190,
                    105, 125, 183, 61, 179, 198, 63, 191, 99, 9, 33, 191, 134, 44, 250, 190,
                    55, 127, 211, 191, 60, 181, 136, 190, 211, 22, 3, 63, 188, 217, 238, 63,
                    62, 211, 22, 59, 182, 28, 2, 192, 51, 241, 31, 191, 157, 237, 83, 63,
                    54, 4, 238, 63, 56, 97, 21, 191, 12, 136, 75, 191, 43, 57, 12, 62,
                    255, 173, 22, 190, 175, 30, 67, 190, 43, 82, 69, 62, 123, 53, 212, 62,
                    49, 63, 73, 63, 19, 224, 39, 63, 80, 208, 242, 189, 141, 105, 249, 63,
                    151, 157, 230, 63, 94, 61, 78, 190, 35, 24, 253, 191, 115, 2, 184, 63,
                    129, 66, 60, 63, 195, 1, 215, 189, 191, 149, 179, 190, 246, 156, 6, 192,
                    250, 226, 24, 63, 181, 47, 85, 190, 122, 2, 183, 191, 44, 89, 225, 62,
                    30, 157, 22, 62, 19, 41, 8, 63, 64, 95, 176, 63, 127, 196, 162, 63,
                    146, 48, 37, 63, 51, 52, 243, 63, 163, 244, 157, 190, 190, 30, 69, 62,
                    146, 133, 6, 63, 63, 222, 219, 61, 140, 29, 139, 189, 159, 189, 59, 191,
                    74, 197, 119, 190, 147, 123, 32, 189, 7, 242, 159, 62, 56, 244, 17, 191,
                    191, 2, 146, 190, 20, 140, 29, 192, 43, 220, 208, 62, 122, 139, 114, 190,
                    106, 194, 147, 190, 162, 17, 61, 63, 213, 6, 44, 62, 115, 201, 93, 189,
                    9, 1, 168, 192, 79, 237, 88, 191, 84, 231, 154, 64, 13, 51, 153, 189,
                    202, 126, 118, 191, 108, 251, 157, 64, 198, 129, 30, 190, 0, 87, 91, 63,
                    140, 60, 53, 192, 181, 179, 159, 63, 14, 51, 36, 64, 229, 53, 139, 61,
                    75, 204, 73, 190, 194, 45, 19, 189, 17, 43, 39, 189, 240, 93, 177, 58,
                    162, 65, 55, 59, 195, 179, 46, 191, 168, 172, 130, 61, 195, 107, 190, 189,
                    45, 88, 121, 189, 64, 15, 216, 62, 155, 221, 22, 63, 18, 101, 114, 63,
                    107, 73, 36, 62, 93, 106, 144, 191, 227, 207, 198, 63, 247, 124, 2, 63,
                    241, 19, 10, 192, 160, 209, 117, 190, 35, 95, 7, 190, 50, 104, 129, 191,
                    220, 139, 119, 63, 80, 6, 85, 191, 240, 123, 39, 191, 196, 189, 251, 62,
                    201, 50, 57, 63, 255, 135, 38, 192, 215, 229, 129, 62, 132, 198, 48, 63,
                    109, 153, 24, 63, 253, 138, 119, 63, 223, 175, 228, 190, 151, 206, 139, 191,
                    210, 198, 140, 190, 50, 0, 83, 62, 25, 97, 182, 61, 254, 246, 25, 191,
                    184, 190, 13, 62, 123, 46, 128, 62, 63, 16, 44, 63, 176, 120, 3, 191,
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
                    100, 245, 9, 190, 1, 71, 234, 61, 37, 191, 236, 61, 172, 212, 145, 59,
                    167, 18, 2, 190, 137, 61, 90, 189, 2, 128, 10, 60, 177, 75, 48, 62,
                    196, 63, 61, 62, 164, 100, 201, 189, 76, 138, 236, 189, 228, 83, 6, 190,
                    67, 194, 229, 62, 62, 134, 18, 190, 181, 215, 236, 188, 58, 24, 203, 190,
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
                    46, 180, 136, 61, 128, 22, 151, 62, 94, 44, 212, 188, 131, 11, 11, 63,
                    190, 54, 227, 188, 219, 35, 166, 190, 181, 246, 162, 62, 145, 11, 205, 62,
                    216, 187, 199, 61, 207, 195, 25, 63, 247, 154, 159, 61, 30, 216, 183, 189,
                    144, 142, 162, 190, 161, 164, 79, 62, 86, 151, 104, 189, 18, 211, 89, 190,
                    85, 205, 206, 190, 88, 238, 98, 61, 81, 31, 38, 62, 164, 155, 182, 62,
                    238, 25, 145, 62, 53, 96, 25, 63, 179, 179, 193, 190, 19, 146, 53, 61,
                    54, 78, 167, 62, 128, 20, 179, 62, 124, 32, 96, 62, 101, 97, 237, 190,
                    212, 41, 55, 190, 67, 78, 0, 188, 23, 46, 51, 188, 104, 120, 66, 189,
                    218, 62, 23, 190, 149, 64, 25, 61, 235, 23, 3, 61, 175, 124, 189, 188,
                    27, 68, 65, 191, 136, 121, 59, 191, 156, 215, 38, 191, 76, 129, 119, 61,
                    98, 184, 206, 190, 194, 12, 5, 63, 93, 251, 95, 189, 255, 255, 69, 63,
                    115, 15, 212, 190, 48, 196, 220, 189, 169, 190, 48, 190, 204, 108, 62, 190,
                    80, 104, 27, 63, 118, 119, 56, 62, 2, 141, 19, 63, 146, 73, 46, 191,
                    34, 206, 128, 190, 216, 48, 237, 190, 179, 207, 64, 62, 89, 252, 28, 190,
                    244, 44, 58, 61, 46, 14, 129, 190, 45, 134, 187, 61, 67, 152, 144, 62,
                    2, 53, 23, 190, 61, 167, 23, 191, 224, 211, 229, 190, 130, 113, 7, 190,
                    193, 247, 209, 62, 42, 253, 182, 189, 207, 106, 181, 190, 127, 19, 154, 188,
                    153, 117, 173, 62, 141, 68, 134, 190, 43, 162, 157, 188, 6, 64, 42, 189,
                    241, 13, 171, 190, 11, 36, 164, 62, 9, 239, 235, 61, 196, 33, 62, 62,
                    26, 216, 83, 62, 180, 218, 115, 190, 3, 91, 19, 63, 28, 87, 73, 61,
                    187, 89, 119, 190, 249, 15, 139, 61, 12, 185, 154, 190, 98, 17, 32, 188,
                    30, 31, 201, 61, 91, 84, 33, 191, 164, 120, 215, 62, 105, 51, 112, 60,
                    207, 56, 75, 62, 4, 87, 178, 190, 150, 226, 232, 60, 172, 18, 68, 190,
                    44, 133, 103, 62, 148, 8, 33, 61, 170, 80, 153, 189, 72, 242, 84, 189,
                    159, 228, 200, 59, 229, 148, 73, 190, 53, 146, 175, 190, 158, 0, 147, 62,
                    125, 183, 137, 190, 218, 213, 200, 62, 3, 45, 250, 188, 12, 206, 255, 62,
                    172, 180, 237, 58, 190, 46, 147, 188, 212, 135, 161, 189, 43, 1, 167, 62,
                    203, 229, 29, 190, 30, 194, 4, 190, 86, 9, 187, 190, 70, 136, 40, 60,
                    132, 42, 70, 191, 106, 155, 159, 61, 41, 210, 145, 190, 97, 21, 228, 62,
                    164, 65, 88, 191, 184, 113, 44, 191, 203, 31, 235, 190, 216, 161, 27, 190,
                    169, 167, 183, 190, 141, 115, 41, 190, 93, 238, 198, 61, 57, 106, 235, 190,
                    21, 177, 26, 189, 105, 255, 189, 190, 137, 73, 245, 62, 47, 185, 150, 190,
                    205, 131, 73, 190, 92, 226, 135, 187, 46, 165, 200, 190, 107, 205, 25, 63,
                    228, 239, 162, 63, 222, 29, 145, 191, 30, 81, 73, 189, 208, 1, 156, 188,
                    214, 244, 113, 191, 181, 242, 91, 189, 243, 171, 187, 188, 183, 163, 22, 62,
                    219, 187, 109, 189, 71, 66, 9, 190, 248, 205, 159, 190, 195, 233, 163, 190,
                    106, 171, 152, 190, 62, 208, 64, 190, 23, 3, 40, 62, 34, 239, 205, 190,
                    105, 117, 37, 61, 76, 87, 98, 60, 38, 214, 28, 61, 45, 176, 195, 62,
                    7, 88, 24, 61, 147, 164, 96, 189, 44, 42, 18, 191, 111, 79, 200, 190,
                    114, 244, 141, 190, 169, 69, 145, 62, 115, 34, 25, 190, 240, 120, 182, 62,
                    255, 223, 38, 190, 57, 168, 185, 62, 138, 205, 10, 63, 242, 15, 51, 62,
                    7, 80, 210, 62, 236, 210, 117, 191, 247, 234, 7, 63, 100, 76, 197, 61,
                    187, 249, 93, 190, 31, 243, 161, 190, 184, 218, 13, 190, 110, 55, 44, 62,
                    32, 155, 186, 189, 160, 119, 237, 189, 49, 4, 136, 191, 237, 203, 205, 190,
                    190, 5, 237, 190, 109, 97, 190, 189, 53, 225, 131, 191, 96, 8, 33, 191,
                    135, 224, 4, 63, 67, 161, 251, 61, 5, 64, 157, 62, 229, 141, 219, 61,
                    184, 108, 115, 62, 123, 95, 218, 190, 118, 196, 123, 62, 148, 75, 160, 61,
                    83, 27, 217, 190, 54, 64, 203, 189, 65, 112, 222, 62, 207, 202, 43, 62,
                    161, 98, 8, 191, 181, 194, 32, 63, 149, 121, 102, 63, 161, 59, 68, 63,
                    96, 204, 43, 191, 175, 66, 104, 190, 97, 173, 68, 62, 155, 82, 130, 61,
                    237, 24, 135, 190, 147, 127, 243, 62, 62, 28, 177, 62, 207, 110, 215, 190,
                    36, 173, 65, 190, 72, 197, 31, 189, 192, 206, 167, 61, 129, 142, 45, 191,
                    151, 37, 47, 189, 202, 145, 80, 61, 125, 116, 169, 62, 196, 167, 142, 188,
                    246, 153, 3, 191, 46, 120, 188, 62, 197, 235, 168, 62, 218, 77, 84, 62,
                    148, 224, 80, 191, 60, 121, 57, 191, 53, 255, 154, 190, 110, 23, 129, 190,
                    118, 201, 214, 62, 28, 42, 89, 191, 50, 66, 168, 188, 66, 186, 4, 62,
                    187, 8, 146, 61, 34, 182, 21, 190, 92, 232, 59, 191, 195, 91, 65, 63,
                    73, 139, 27, 190, 131, 138, 117, 190, 155, 115, 69, 63, 81, 58, 147, 188,
                    94, 211, 194, 190, 186, 239, 217, 62, 70, 87, 197, 189, 177, 54, 83, 191,
                    183, 16, 10, 189, 184, 179, 215, 61, 122, 67, 218, 62, 173, 62, 248, 190,
                    246, 70, 184, 188, 59, 111, 137, 189, 245, 39, 93, 191, 52, 61, 211, 188,
                    71, 32, 139, 62, 2, 111, 181, 189, 113, 43, 237, 190, 82, 94, 154, 62,
                    195, 22, 228, 190, 152, 49, 249, 189, 66, 0, 10, 191, 176, 105, 242, 61,
                    240, 178, 153, 62, 90, 135, 113, 60, 186, 233, 23, 190, 27, 157, 0, 63,
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
                    106, 216, 65, 189, 202, 213, 20, 61, 180, 247, 160, 188, 46, 107, 216, 189,
                    88, 173, 15, 60, 18, 169, 63, 61, 143, 58, 63, 190, 208, 206, 231, 188,
                    100, 203, 91, 190, 29, 196, 66, 189, 21, 195, 22, 190, 242, 215, 4, 189,
                    53, 251, 59, 188, 197, 128, 229, 61, 112, 241, 93, 62, 239, 214, 152, 62,
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
                    216, 51, 100, 191, 20, 233, 74, 62, 184, 163, 115, 62, 190, 83, 159, 190,
                    150, 138, 144, 190, 183, 206, 166, 191, 120, 175, 166, 61, 112, 102, 173, 191,
                    8, 208, 73, 191, 13, 2, 133, 62, 154, 154, 116, 191, 0, 79, 85, 190,
                    188, 152, 194, 189, 41, 52, 254, 190, 107, 230, 125, 190, 39, 250, 30, 191,
                    254, 38, 16, 190, 101, 72, 9, 190, 34, 119, 15, 63, 108, 15, 55, 190,
                    120, 132, 83, 191, 86, 208, 110, 191, 237, 44, 207, 187, 43, 34, 88, 191,
                    98, 197, 6, 191, 33, 130, 183, 63, 187, 60, 133, 62, 2, 213, 200, 62,
                    215, 158, 72, 63, 55, 118, 99, 63, 188, 248, 220, 187, 20, 193, 124, 191,
                    124, 113, 18, 63, 132, 202, 72, 63, 89, 97, 66, 63, 78, 172, 130, 191,
                    180, 138, 29, 63, 175, 125, 35, 191, 104, 58, 127, 190, 170, 41, 50, 63,
                    187, 227, 83, 63, 203, 38, 132, 190, 123, 44, 48, 191, 94, 93, 166, 191,
                    58, 131, 157, 189, 175, 88, 36, 191, 210, 139, 48, 63, 159, 17, 186, 62,
                    168, 125, 19, 191, 69, 54, 161, 61, 218, 4, 162, 63, 28, 118, 87, 62,
                    116, 233, 211, 189, 55, 221, 121, 63, 190, 227, 216, 62, 77, 76, 131, 63,
                    113, 41, 108, 62, 44, 125, 249, 62, 49, 240, 31, 62, 49, 214, 116, 191,
                    226, 54, 148, 61, 39, 38, 27, 190, 123, 113, 6, 191, 168, 102, 113, 63,
                    94, 64, 250, 190, 21, 84, 109, 189, 139, 167, 219, 190, 89, 52, 129, 190,
                    177, 77, 216, 60, 129, 43, 58, 191, 12, 115, 160, 62, 185, 121, 38, 62,
                    181, 31, 208, 61, 132, 206, 19, 63, 7, 240, 89, 191, 205, 98, 202, 189,
                    74, 187, 138, 189, 191, 81, 173, 189, 119, 205, 19, 61, 177, 255, 38, 191,
                    249, 131, 30, 189, 171, 182, 39, 191, 85, 174, 92, 63, 177, 185, 5, 63,
                    1, 114, 132, 62, 53, 157, 28, 190, 93, 168, 208, 62, 36, 234, 221, 61,
                    79, 40, 110, 191, 21, 53, 81, 191, 122, 34, 139, 191, 14, 87, 13, 191,
                    240, 31, 7, 192, 182, 89, 46, 192, 1, 7, 113, 189, 22, 33, 34, 190,
                    240, 202, 194, 63, 37, 4, 67, 191, 41, 91, 45, 191, 126, 27, 34, 63,
                    164, 49, 151, 189, 206, 246, 105, 63, 248, 23, 137, 191, 242, 86, 147, 61,
                    151, 149, 64, 63, 36, 182, 156, 191, 1, 188, 0, 64, 199, 201, 55, 191,
                    153, 92, 148, 63, 42, 196, 185, 63, 106, 82, 155, 190, 131, 21, 112, 191,
                    5, 142, 66, 63, 55, 54, 162, 190, 144, 215, 232, 61, 38, 109, 111, 191,
                    29, 116, 13, 63, 3, 25, 224, 61, 105, 59, 230, 190, 42, 210, 73, 62,
                    146, 200, 174, 191, 250, 146, 167, 190, 215, 234, 46, 191, 103, 180, 74, 190,
                    192, 122, 44, 190, 73, 87, 71, 191, 158, 233, 28, 191, 253, 73, 157, 62,
                    184, 121, 132, 63, 204, 223, 142, 191, 122, 167, 64, 62, 106, 91, 66, 63,
                    127, 148, 141, 63, 212, 154, 57, 191, 165, 180, 144, 62, 218, 66, 241, 190,
                    42, 208, 127, 190, 101, 8, 249, 62, 14, 212, 8, 191, 210, 210, 63, 63,
                    129, 180, 187, 190, 150, 33, 159, 62, 248, 156, 46, 62, 235, 228, 58, 63,
                    253, 30, 90, 191, 141, 245, 57, 190, 136, 89, 32, 190, 136, 39, 163, 191,
                    6, 241, 3, 190, 229, 181, 41, 59, 133, 199, 141, 63, 250, 46, 115, 63,
                    47, 202, 6, 63, 108, 239, 46, 62, 253, 234, 205, 190, 158, 232, 69, 63,
                    171, 147, 90, 62, 149, 12, 149, 63, 78, 154, 123, 191, 170, 1, 128, 63,
                    165, 190, 90, 62, 169, 25, 70, 60, 163, 27, 102, 62, 191, 195, 129, 63,
                    145, 65, 208, 59, 142, 143, 21, 62, 158, 74, 18, 61, 203, 71, 203, 61,
                    2, 134, 191, 190, 57, 51, 6, 192, 74, 213, 48, 191, 130, 154, 201, 60,
                    110, 184, 183, 189, 56, 149, 110, 62, 78, 114, 236, 62, 60, 122, 155, 191,
                    162, 66, 85, 191, 244, 20, 24, 62, 137, 69, 135, 191, 74, 169, 60, 189,
                    59, 78, 170, 190, 129, 110, 53, 63, 83, 180, 114, 190, 210, 108, 129, 59,
                    37, 135, 199, 191, 155, 112, 214, 61, 155, 116, 83, 60, 24, 225, 147, 62,
                    108, 32, 91, 62, 249, 85, 12, 62, 173, 172, 11, 192, 200, 95, 0, 63,
                    99, 236, 224, 190, 47, 105, 8, 63, 58, 46, 19, 191, 172, 240, 35, 190,
                    203, 234, 74, 61, 184, 108, 152, 191, 153, 148, 4, 191, 140, 41, 165, 190,
                    19, 248, 63, 63, 254, 185, 71, 190, 133, 108, 185, 186, 185, 47, 51, 191,
                    235, 195, 92, 63, 205, 5, 230, 189, 89, 115, 61, 188, 168, 203, 152, 190,
                    49, 99, 22, 63, 206, 11, 237, 190, 134, 250, 162, 190, 91, 2, 92, 191,
                    59, 11, 138, 191, 100, 186, 165, 190, 207, 250, 60, 191, 72, 141, 178, 62,
                    186, 44, 158, 63, 110, 23, 184, 61, 63, 70, 183, 63, 196, 212, 20, 62,
                    148, 221, 2, 64, 0, 157, 176, 63, 84, 146, 235, 62, 116, 202, 104, 186,
                    56, 210, 139, 190, 243, 155, 110, 191, 17, 30, 130, 190, 158, 255, 37, 63,
                    74, 194, 105, 62, 243, 152, 122, 191, 65, 180, 220, 59, 163, 205, 21, 191,
                    224, 88, 252, 63, 255, 52, 58, 191, 81, 80, 182, 63, 101, 248, 152, 191,
                    32, 221, 225, 63, 197, 44, 3, 62, 163, 88, 118, 191, 246, 155, 215, 189,
                    134, 245, 3, 63, 243, 223, 56, 63, 64, 30, 79, 62, 72, 162, 215, 190,
                    164, 201, 32, 190, 132, 22, 14, 191, 47, 196, 161, 62, 23, 254, 162, 190,
                    143, 99, 145, 191, 216, 98, 116, 61, 63, 200, 14, 191, 199, 195, 145, 189,
                    37, 198, 117, 62, 145, 243, 86, 191, 34, 44, 26, 61, 6, 196, 98, 190,
                    160, 25, 112, 191, 188, 118, 160, 62, 101, 248, 28, 191, 64, 46, 4, 190,
                    114, 90, 224, 59, 185, 184, 127, 190, 188, 72, 6, 64, 127, 169, 44, 62,
                    97, 168, 177, 62, 162, 52, 72, 62, 173, 170, 92, 190, 93, 200, 111, 191,
                    69, 56, 253, 191, 1, 137, 22, 63, 95, 111, 35, 191, 12, 70, 206, 63,
                    84, 85, 144, 61, 86, 38, 113, 191, 179, 146, 44, 190, 177, 29, 35, 62,
                    25, 241, 239, 62, 219, 133, 80, 63, 184, 181, 52, 63, 27, 129, 87, 189,
                    250, 38, 218, 189, 62, 241, 205, 190, 148, 9, 63, 63, 19, 133, 2, 191,
                    177, 150, 72, 191, 1, 224, 51, 63, 10, 7, 118, 191, 51, 82, 137, 63,
                    134, 162, 121, 62, 239, 81, 235, 189, 246, 203, 142, 190, 163, 80, 165, 191,
                    2, 130, 149, 63, 159, 212, 43, 191, 3, 108, 183, 190, 152, 77, 19, 63,
                    87, 37, 30, 191, 80, 131, 80, 63, 167, 89, 253, 191, 108, 164, 164, 191,
                    54, 1, 135, 189, 114, 40, 242, 190, 255, 229, 203, 190, 92, 111, 3, 63,
                    21, 5, 119, 191, 240, 251, 117, 190, 225, 102, 129, 191, 191, 168, 238, 62,
                    81, 82, 21, 191, 200, 193, 144, 63, 79, 121, 91, 61, 159, 37, 193, 62,
                    110, 172, 26, 191, 120, 49, 142, 62, 226, 136, 150, 62, 24, 213, 231, 189,
                    255, 53, 177, 191, 65, 138, 172, 63, 134, 74, 164, 191, 58, 71, 48, 63,
                    128, 173, 61, 191, 45, 216, 81, 62, 120, 92, 144, 63, 228, 197, 103, 63,
                    168, 8, 198, 62, 25, 47, 65, 63, 84, 59, 128, 63, 244, 92, 198, 63,
                    51, 43, 136, 191, 67, 56, 118, 191, 35, 31, 150, 191, 241, 239, 67, 62,
                    234, 102, 183, 62, 161, 172, 174, 62, 210, 44, 250, 190, 196, 56, 68, 63,
                    51, 59, 5, 64, 186, 67, 242, 63, 204, 49, 217, 189, 14, 153, 204, 191,
                    250, 172, 36, 63, 202, 153, 156, 63, 51, 253, 126, 191, 125, 63, 32, 63,
                    41, 195, 165, 63, 90, 110, 171, 189, 232, 242, 91, 63, 132, 247, 199, 63,
                    77, 191, 6, 64, 234, 70, 19, 190, 67, 35, 196, 60, 107, 140, 206, 191,
                    171, 224, 46, 63, 239, 149, 232, 62, 53, 230, 247, 191, 21, 252, 33, 191,
                    124, 68, 115, 62, 93, 249, 42, 191, 179, 138, 67, 191, 234, 8, 162, 191,
                    57, 244, 158, 62, 112, 35, 134, 190, 134, 20, 172, 191, 82, 189, 143, 191,
                    161, 5, 177, 189, 12, 15, 7, 191, 71, 232, 7, 63, 30, 40, 201, 62,
                    5, 212, 187, 190, 21, 248, 43, 63, 178, 215, 80, 191, 45, 210, 159, 191,
                    82, 117, 10, 62, 90, 169, 15, 62, 178, 158, 217, 190, 136, 139, 49, 62,
                    96, 221, 217, 190, 106, 189, 51, 63, 177, 205, 237, 190, 51, 235, 108, 63,
                    181, 197, 156, 190, 103, 15, 129, 189, 209, 18, 193, 191, 197, 247, 211, 63,
                    135, 92, 173, 191, 211, 212, 184, 191, 16, 231, 82, 63, 90, 240, 1, 64,
                    187, 137, 47, 191, 242, 208, 200, 190, 222, 70, 206, 62, 254, 163, 88, 61,
                    118, 127, 142, 62, 139, 81, 183, 61, 185, 193, 62, 63, 168, 102, 147, 191,
                    49, 114, 8, 189, 26, 184, 54, 63, 121, 108, 252, 191, 121, 102, 155, 190,
                    55, 135, 170, 62, 169, 103, 195, 191, 214, 226, 154, 189, 136, 248, 253, 62,
                    177, 23, 100, 61, 141, 70, 42, 61, 23, 244, 77, 63, 68, 40, 71, 63,
                    160, 138, 54, 190, 125, 70, 202, 61, 51, 67, 74, 62, 8, 250, 233, 191,
                    74, 153, 156, 63, 26, 95, 214, 63, 191, 215, 203, 189, 102, 91, 117, 191,
                    190, 19, 1, 192, 201, 4, 99, 63, 148, 152, 137, 63, 205, 195, 4, 189,
                    241, 53, 18, 63, 188, 60, 176, 62, 166, 30, 183, 63, 126, 117, 180, 63,
                    55, 98, 169, 190, 205, 255, 107, 63, 19, 243, 118, 191, 248, 248, 235, 63,
                    82, 34, 187, 191, 224, 253, 132, 63, 65, 98, 160, 191, 37, 37, 167, 63,
                    172, 202, 22, 190, 106, 97, 139, 191, 186, 80, 182, 190, 19, 118, 142, 191,
                    166, 118, 211, 190, 195, 236, 249, 188, 2, 105, 53, 59, 56, 103, 27, 190,
                    225, 70, 200, 190, 188, 30, 203, 62, 145, 228, 193, 62, 224, 74, 214, 61,
                    113, 239, 51, 191, 192, 66, 24, 191, 11, 41, 104, 63, 92, 199, 162, 190,
                    146, 182, 250, 62, 15, 207, 246, 188, 79, 83, 249, 190, 108, 130, 105, 191,
                    170, 21, 237, 190, 15, 168, 4, 190, 91, 230, 247, 61, 222, 251, 91, 190,
                    188, 67, 17, 191, 106, 45, 60, 63, 232, 169, 124, 191, 38, 86, 195, 190,
                    67, 48, 16, 190, 227, 36, 143, 189, 201, 46, 43, 191, 79, 189, 131, 63,
                    10, 19, 215, 63, 42, 206, 132, 61, 197, 180, 58, 62, 230, 227, 202, 60,
                    202, 57, 165, 62, 44, 220, 45, 191, 52, 222, 245, 189, 103, 225, 226, 62,
                    138, 157, 90, 63, 32, 252, 127, 63, 43, 232, 134, 63, 164, 206, 200, 190,
                    171, 121, 20, 64, 91, 160, 228, 188, 9, 16, 236, 190, 39, 130, 99, 191,
                    25, 147, 10, 62, 185, 251, 142, 61, 60, 121, 33, 191, 197, 83, 165, 190,
                    105, 130, 2, 191, 190, 87, 199, 190, 60, 88, 187, 60, 124, 25, 101, 191,
                    179, 147, 203, 190, 255, 93, 92, 62, 89, 104, 14, 190, 104, 191, 131, 191,
                    3, 13, 114, 190, 205, 205, 169, 191, 235, 62, 9, 63, 199, 14, 239, 189,
                    212, 48, 6, 63, 95, 248, 230, 190, 110, 226, 114, 63, 159, 139, 24, 191,
                    96, 100, 155, 189, 161, 47, 47, 191, 57, 78, 17, 62, 186, 160, 217, 61,
                    109, 22, 78, 190, 95, 149, 154, 63, 85, 62, 169, 62, 132, 113, 228, 62,
                    223, 164, 20, 63, 146, 183, 29, 62, 70, 4, 160, 191, 40, 97, 109, 63,
                    178, 33, 138, 190, 95, 217, 197, 62, 149, 126, 25, 191, 49, 131, 6, 62,
                    181, 194, 174, 190, 15, 68, 204, 62, 109, 247, 133, 61, 71, 201, 242, 190,
                    255, 200, 32, 191, 84, 125, 188, 61, 141, 189, 197, 61, 229, 183, 141, 61,
                    173, 207, 88, 191, 199, 228, 6, 190, 128, 17, 141, 62, 230, 189, 216, 61,
                    106, 192, 31, 189, 146, 243, 54, 189, 229, 138, 188, 62, 240, 33, 180, 62,
                    63, 132, 132, 189, 5, 194, 65, 62, 24, 217, 147, 187, 26, 26, 90, 63,
                    27, 178, 1, 63, 228, 156, 36, 60, 53, 119, 215, 62, 248, 241, 59, 189,
                    215, 153, 19, 61, 31, 157, 133, 62, 123, 56, 4, 63, 186, 241, 43, 189,
                    111, 164, 87, 190, 24, 107, 213, 61, 158, 115, 183, 62, 189, 65, 204, 190,
                    8, 103, 98, 190, 14, 156, 5, 191, 75, 12, 171, 62, 156, 34, 140, 61,
                    153, 113, 185, 60, 31, 78, 115, 61, 130, 157, 171, 190, 199, 26, 253, 190,
                    90, 73, 179, 62, 195, 174, 49, 190, 125, 154, 133, 62, 64, 252, 149, 61,
                    131, 159, 36, 189, 56, 10, 71, 189, 40, 217, 3, 191, 36, 24, 188, 62,
                    247, 7, 90, 190, 77, 215, 18, 63, 73, 162, 5, 62, 142, 217, 80, 189,
                    42, 110, 40, 61, 66, 223, 91, 62, 3, 127, 226, 62, 13, 25, 167, 62,
                    220, 120, 51, 190, 136, 216, 254, 62, 254, 194, 79, 63, 224, 175, 139, 189,
                    85, 148, 191, 189, 206, 161, 184, 61, 173, 246, 138, 62, 43, 121, 132, 190,
                    152, 33, 147, 61, 116, 151, 154, 189, 164, 232, 165, 189, 77, 181, 83, 63,
                    197, 34, 7, 191, 63, 46, 77, 189, 29, 9, 73, 191, 101, 62, 23, 62,
                    158, 240, 104, 190, 247, 195, 141, 188, 71, 240, 199, 62, 195, 34, 110, 61,
                    216, 15, 141, 61, 248, 19, 114, 62, 69, 113, 133, 191, 90, 106, 57, 62,
                    205, 201, 70, 63, 209, 213, 20, 191, 160, 248, 38, 191, 141, 18, 236, 190,
                    184, 171, 158, 190, 223, 58, 55, 63, 198, 148, 37, 191, 209, 167, 48, 190,
                    199, 39, 232, 62, 219, 187, 82, 63, 4, 78, 18, 191, 221, 115, 71, 191,
                    207, 125, 136, 190, 246, 200, 40, 63, 61, 42, 128, 63, 254, 186, 117, 62,
                    199, 247, 133, 191, 188, 81, 90, 190, 153, 14, 29, 63, 176, 61, 125, 190,
                    29, 215, 29, 63, 48, 160, 153, 62, 10, 85, 244, 62, 151, 109, 224, 62,
                    50, 190, 135, 190, 188, 61, 4, 190, 74, 45, 196, 63, 83, 73, 62, 63,
                    114, 31, 179, 58, 180, 81, 194, 189, 190, 127, 252, 188, 25, 101, 172, 189,
                    164, 84, 157, 62, 243, 158, 250, 189, 33, 84, 233, 189, 248, 173, 131, 188,
                    82, 166, 6, 63, 182, 50, 177, 190, 205, 70, 85, 189, 25, 111, 21, 191,
                    179, 201, 98, 62, 31, 103, 245, 62, 56, 93, 116, 61, 68, 36, 161, 62,
                    119, 78, 66, 189, 44, 115, 73, 61, 229, 57, 246, 189, 220, 205, 184, 61,
                    77, 182, 61, 190, 21, 67, 151, 188, 9, 32, 102, 62, 67, 214, 81, 189,
                    123, 206, 10, 61, 2, 210, 189, 189, 157, 157, 85, 62, 55, 240, 145, 189,
                    247, 251, 4, 61, 131, 98, 127, 62, 241, 225, 82, 190, 141, 19, 189, 188,
                    34, 124, 53, 189, 94, 66, 156, 190, 70, 2, 59, 63, 87, 191, 138, 62,
                    3, 233, 140, 190, 245, 82, 114, 190, 21, 176, 154, 62, 232, 37, 103, 190,
                    230, 187, 138, 190, 51, 204, 19, 62, 141, 183, 250, 61, 141, 120, 58, 191,
                    199, 49, 100, 189, 244, 204, 47, 189, 171, 196, 179, 62, 108, 60, 84, 61,
                    8, 2, 169, 63, 231, 38, 20, 63, 113, 68, 154, 190, 243, 43, 40, 191,
                    219, 4, 197, 62, 234, 49, 89, 62, 49, 203, 50, 63, 8, 101, 156, 61,
                    115, 118, 184, 189, 148, 240, 233, 190, 159, 243, 92, 62, 205, 49, 99, 191,
                    239, 143, 140, 63, 190, 12, 241, 62, 75, 83, 198, 190, 41, 216, 57, 63,
                    1, 12, 106, 190, 106, 255, 161, 189, 211, 1, 146, 61, 29, 60, 99, 62,
                    243, 148, 148, 62, 117, 208, 56, 189, 191, 41, 165, 62, 185, 8, 49, 62,
                    144, 69, 248, 61, 134, 53, 151, 190, 52, 45, 59, 191, 114, 6, 57, 63,
                    151, 197, 63, 191, 9, 21, 0, 63, 117, 117, 201, 62, 6, 128, 150, 62,
                    26, 208, 135, 190, 61, 156, 35, 62, 238, 138, 113, 61, 45, 107, 51, 62,
                    30, 100, 179, 60, 185, 234, 234, 61, 85, 7, 43, 62, 162, 64, 164, 189,
                    71, 126, 27, 63, 8, 247, 28, 190, 250, 198, 120, 62, 254, 226, 167, 62,
                    126, 6, 22, 191, 240, 3, 139, 62, 119, 237, 35, 191, 223, 223, 144, 62,
                    16, 218, 176, 63, 58, 94, 168, 191, 61, 151, 182, 62, 56, 36, 61, 191,
                    133, 222, 245, 63, 29, 159, 28, 191, 19, 43, 164, 61, 167, 196, 249, 188,
                    81, 116, 130, 62, 221, 87, 105, 191, 133, 254, 52, 191, 47, 26, 6, 191,
                    248, 214, 188, 62, 1, 45, 209, 190, 170, 219, 249, 62, 252, 24, 10, 63,
                    216, 121, 8, 191, 79, 142, 47, 191, 141, 86, 25, 63, 208, 105, 86, 63,
                    243, 86, 242, 190, 101, 180, 12, 190, 207, 238, 116, 62, 68, 146, 21, 63,
                    93, 72, 243, 191, 68, 0, 35, 63, 27, 201, 248, 190, 11, 200, 223, 189,
                    4, 190, 23, 190, 133, 139, 140, 63, 114, 158, 163, 190, 200, 36, 24, 190,
                    222, 156, 187, 61, 205, 154, 133, 61, 219, 9, 163, 187, 237, 16, 139, 190,
                    220, 133, 211, 189, 234, 106, 69, 61, 142, 21, 243, 61, 215, 104, 152, 189,
                    228, 181, 242, 62, 147, 147, 133, 59, 210, 31, 140, 59, 131, 200, 68, 62,
                    2, 187, 129, 190, 167, 69, 223, 190, 196, 31, 122, 62, 234, 145, 228, 61,
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
                    198, 135, 27, 191, 1, 195, 189, 189, 164, 21, 84, 190, 252, 46, 53, 62,
                    5, 1, 46, 191, 60, 78, 47, 191, 124, 35, 205, 62, 25, 41, 223, 63,
                    45, 166, 81, 62, 8, 242, 250, 61, 137, 39, 60, 191, 116, 214, 187, 190,
                    5, 76, 178, 191, 108, 15, 206, 60, 121, 198, 198, 189, 23, 255, 203, 62,
                    176, 0, 78, 62, 191, 5, 41, 63, 133, 250, 189, 62, 69, 253, 89, 62,
                    51, 167, 77, 189, 24, 177, 239, 189, 108, 63, 112, 62, 107, 209, 197, 62,
                    27, 185, 15, 190, 223, 39, 66, 63, 97, 162, 54, 190, 176, 61, 32, 63,
                    159, 25, 212, 191, 37, 108, 13, 190, 198, 235, 213, 191, 127, 63, 13, 63,
                    118, 2, 154, 190, 250, 86, 183, 191, 63, 148, 138, 191, 135, 57, 23, 191,
                    113, 61, 83, 62, 46, 175, 22, 62, 22, 54, 216, 62, 152, 105, 177, 190,
                    244, 111, 172, 190, 27, 250, 39, 191, 35, 167, 179, 190, 220, 92, 233, 191,
                    3, 92, 163, 190, 172, 227, 168, 190, 48, 3, 91, 190, 177, 134, 41, 62,
                    98, 53, 78, 188, 181, 96, 164, 191, 122, 40, 116, 191, 229, 208, 140, 63,
                    208, 204, 9, 63, 154, 61, 31, 190, 200, 104, 189, 189, 4, 193, 134, 190,
                    98, 44, 158, 61, 8, 40, 45, 63, 105, 38, 15, 187, 97, 60, 80, 188,
                    210, 113, 151, 190, 4, 119, 115, 62, 95, 151, 130, 62, 12, 192, 84, 61,
                    110, 48, 100, 190, 164, 39, 45, 190, 70, 209, 14, 189, 132, 155, 5, 191,
                    194, 145, 20, 191, 22, 214, 252, 190, 7, 82, 151, 61, 57, 118, 136, 63,
                    62, 106, 77, 191, 226, 130, 118, 191, 233, 63, 29, 191, 173, 0, 196, 190,
                    69, 57, 102, 191, 211, 141, 222, 62, 113, 50, 171, 191, 71, 19, 49, 191,
                    249, 147, 36, 191, 18, 229, 158, 63, 190, 2, 75, 63, 89, 249, 12, 191,
                    56, 169, 148, 63, 246, 220, 1, 190, 186, 121, 144, 63, 180, 86, 139, 63,
                    14, 136, 128, 191, 157, 245, 207, 63, 163, 142, 173, 191, 87, 190, 44, 192,
                    43, 210, 69, 191, 202, 242, 78, 191, 41, 222, 143, 190, 131, 84, 147, 63,
                    211, 0, 186, 191, 132, 54, 150, 63, 173, 76, 224, 191, 59, 229, 223, 186,
                    171, 244, 40, 192, 214, 153, 216, 63, 208, 116, 91, 191, 206, 19, 6, 191,
                    233, 74, 197, 63, 238, 30, 148, 191, 188, 101, 51, 63, 192, 155, 167, 62,
                    96, 120, 29, 190, 116, 118, 92, 191, 221, 79, 151, 63, 83, 173, 148, 62,
                    205, 207, 155, 62, 43, 253, 91, 191, 109, 95, 52, 191, 29, 97, 88, 191,
                    204, 117, 43, 191, 219, 56, 100, 63, 0, 178, 88, 191, 219, 224, 49, 62,
                    50, 205, 186, 63, 171, 232, 128, 63, 160, 166, 129, 63, 193, 255, 187, 190,
                    227, 161, 23, 191, 107, 246, 197, 61, 163, 112, 213, 189, 179, 30, 134, 191,
                    186, 127, 119, 190, 211, 202, 31, 191, 199, 155, 23, 62, 22, 41, 23, 191,
                    235, 25, 201, 62, 58, 193, 130, 62, 235, 25, 176, 190, 196, 62, 152, 63,
                    119, 53, 229, 190, 218, 149, 103, 190, 211, 113, 250, 62, 218, 230, 151, 189,
                    35, 2, 34, 191, 147, 185, 11, 63, 46, 196, 114, 61, 186, 176, 232, 190,
                    162, 80, 214, 190, 148, 15, 141, 190, 53, 2, 146, 191, 167, 121, 246, 190,
                    251, 254, 158, 190, 205, 26, 9, 63, 49, 181, 126, 190, 164, 155, 244, 61,
                    156, 123, 112, 63, 53, 31, 83, 191, 33, 196, 229, 60, 230, 100, 117, 63,
                    101, 246, 223, 185, 96, 15, 204, 62, 93, 185, 161, 191, 220, 158, 131, 191,
                    58, 78, 92, 63, 111, 157, 128, 59, 81, 249, 32, 62, 221, 169, 171, 63,
                    158, 2, 208, 191, 102, 12, 172, 191, 148, 64, 241, 63, 87, 231, 241, 63,
                    13, 38, 202, 191, 33, 149, 158, 190, 23, 17, 3, 191, 47, 68, 246, 191,
                    236, 214, 25, 191, 124, 208, 8, 191, 71, 107, 67, 62, 96, 225, 211, 63,
                    10, 39, 28, 189, 118, 50, 65, 190, 99, 104, 23, 191, 131, 199, 107, 191,
                    238, 187, 91, 191, 185, 76, 170, 62, 3, 175, 45, 191, 144, 247, 131, 63,
                    185, 29, 99, 63, 83, 169, 93, 62, 4, 143, 111, 62, 140, 248, 234, 63,
                    47, 19, 119, 61, 69, 95, 156, 63, 189, 31, 142, 63, 212, 177, 135, 191,
                    113, 179, 168, 62, 6, 212, 11, 191, 156, 214, 222, 190, 187, 98, 42, 63,
                    226, 83, 39, 191, 232, 12, 231, 190, 14, 49, 136, 63, 30, 47, 36, 191,
                    104, 100, 236, 61, 65, 43, 180, 62, 42, 73, 210, 190, 155, 101, 138, 191,
                    27, 30, 192, 61, 69, 18, 51, 63, 214, 245, 54, 190, 141, 93, 54, 63,
                    94, 37, 199, 189, 230, 71, 5, 63, 87, 122, 59, 63, 75, 102, 69, 63,
                    6, 96, 187, 191, 51, 118, 132, 191, 105, 215, 91, 63, 60, 130, 163, 63,
                    94, 9, 201, 63, 154, 172, 104, 63, 115, 64, 52, 191, 253, 185, 132, 63,
                    242, 164, 17, 190, 114, 44, 195, 191, 182, 87, 198, 61, 65, 133, 97, 63,
                    222, 184, 189, 189, 188, 29, 211, 190, 81, 144, 178, 191, 168, 213, 34, 191,
                    239, 201, 219, 190, 118, 138, 13, 64, 190, 8, 215, 191, 131, 217, 80, 63,
                    229, 234, 41, 190, 31, 183, 65, 191, 180, 221, 0, 63, 3, 199, 54, 191,
                    201, 6, 198, 191, 194, 133, 175, 63, 48, 239, 42, 191, 143, 196, 247, 60,
                    186, 110, 188, 190, 192, 177, 147, 62, 7, 99, 53, 190, 52, 96, 238, 62,
                    153, 197, 159, 190, 33, 27, 113, 62, 173, 146, 13, 191, 50, 179, 174, 63,
                    35, 233, 249, 63, 56, 211, 254, 190, 193, 197, 229, 62, 176, 115, 121, 62,
                    168, 208, 132, 63, 78, 190, 154, 61, 75, 169, 198, 63, 182, 186, 95, 63,
                    214, 95, 215, 190, 12, 142, 179, 190, 105, 35, 79, 192, 241, 41, 132, 191,
                    143, 227, 22, 192, 212, 7, 108, 64, 59, 183, 35, 192, 163, 202, 15, 192,
                    228, 175, 38, 64, 147, 164, 49, 63, 151, 250, 10, 64, 63, 128, 87, 191,
                    42, 46, 140, 63, 12, 59, 30, 63, 23, 37, 196, 63, 163, 207, 242, 63,
                    217, 94, 223, 189, 135, 60, 131, 191, 203, 131, 158, 190, 19, 222, 52, 190,
                    40, 63, 146, 63, 176, 98, 147, 63, 29, 16, 123, 191, 75, 243, 43, 191,
                    212, 183, 211, 190, 250, 229, 252, 61, 157, 24, 199, 62, 206, 13, 185, 63,
                    214, 15, 160, 191, 19, 226, 244, 61, 144, 78, 145, 191, 228, 166, 199, 190,
                    98, 216, 254, 191, 40, 216, 188, 63, 121, 96, 198, 190, 208, 159, 212, 63,
                    175, 151, 245, 59, 165, 77, 149, 62, 242, 164, 58, 191, 55, 20, 49, 190,
                    201, 5, 70, 191, 6, 1, 186, 63, 28, 51, 85, 63, 62, 103, 183, 61,
                    241, 29, 233, 62, 0, 66, 72, 63, 130, 95, 42, 64, 56, 22, 233, 61,
                    202, 55, 144, 190, 166, 211, 163, 61, 176, 112, 98, 190, 244, 250, 86, 61,
                    169, 214, 134, 191, 216, 82, 169, 189, 225, 195, 146, 62, 254, 132, 41, 191,
                    14, 194, 162, 191, 42, 216, 183, 63, 48, 14, 162, 63, 33, 138, 8, 192,
                    47, 29, 176, 190, 148, 69, 242, 189, 143, 216, 54, 64, 212, 164, 4, 64,
                    192, 150, 97, 191, 101, 93, 129, 188, 170, 42, 146, 186, 100, 31, 98, 190,
                    89, 41, 140, 191, 78, 100, 77, 63, 132, 218, 130, 191, 37, 38, 144, 189,
                    106, 115, 70, 63, 198, 28, 116, 190, 0, 99, 232, 63, 176, 59, 101, 61,
                    18, 209, 136, 62, 69, 174, 239, 63, 45, 72, 30, 64, 195, 20, 171, 62,
                    174, 202, 234, 191, 241, 224, 35, 64, 172, 119, 148, 63, 157, 158, 195, 191,
                    118, 5, 212, 62, 213, 142, 146, 63, 21, 44, 85, 63, 103, 227, 44, 62,
                    137, 105, 170, 191, 28, 218, 55, 190, 112, 239, 233, 62, 223, 34, 33, 192,
                    195, 106, 208, 191, 126, 190, 180, 63, 127, 25, 4, 192, 170, 190, 72, 63,
                    70, 198, 170, 62, 253, 213, 32, 192, 133, 48, 67, 61, 68, 94, 197, 191,
                    104, 80, 137, 63, 64, 212, 84, 63, 28, 193, 156, 191, 68, 238, 17, 64,
                    180, 147, 124, 191, 189, 41, 50, 192, 116, 198, 132, 63, 47, 1, 177, 191,
                    71, 15, 237, 191, 164, 174, 208, 62, 174, 180, 70, 64, 69, 158, 175, 63,
                    96, 7, 180, 61, 201, 168, 130, 190, 214, 95, 126, 191, 254, 144, 173, 63,
                    85, 230, 80, 63, 221, 130, 183, 62, 14, 152, 185, 191, 107, 157, 8, 63,
                    156, 226, 56, 63, 53, 250, 34, 62, 141, 170, 32, 64, 238, 81, 140, 63,
                    87, 128, 52, 191, 36, 10, 64, 191, 224, 81, 170, 62, 16, 63, 191, 191,
                    208, 174, 104, 190, 80, 63, 89, 191, 21, 74, 142, 60, 71, 162, 249, 191,
                    151, 142, 207, 190, 204, 91, 190, 63, 207, 174, 9, 192, 132, 115, 32, 191,
                    168, 4, 52, 191, 111, 157, 26, 190, 151, 178, 190, 63, 163, 95, 125, 62,
                    168, 85, 43, 63, 159, 208, 195, 63, 203, 158, 149, 62, 243, 163, 194, 62,
                    168, 204, 24, 192, 232, 64, 42, 63, 171, 222, 240, 191, 107, 158, 169, 190,
                    206, 77, 56, 191, 129, 141, 69, 64, 183, 35, 249, 191, 142, 32, 57, 191,
                    31, 218, 162, 62, 12, 130, 141, 63, 54, 57, 127, 63, 27, 77, 128, 61,
                    53, 8, 171, 61, 76, 150, 244, 63, 27, 161, 81, 63, 106, 13, 170, 62,
                    239, 210, 215, 191, 233, 161, 253, 191, 68, 110, 39, 63, 215, 94, 191, 189,
                    60, 191, 45, 63, 242, 104, 45, 190, 62, 63, 131, 191, 8, 48, 81, 62,
                    75, 107, 10, 62, 129, 164, 65, 191, 57, 120, 63, 62, 157, 36, 59, 63,
                    231, 98, 188, 191, 89, 79, 131, 63, 146, 203, 77, 63, 189, 93, 35, 192,
                    170, 23, 71, 63, 33, 29, 41, 190, 115, 74, 2, 63, 219, 134, 224, 62,
                    24, 85, 224, 63, 73, 49, 226, 189, 64, 109, 171, 62, 212, 224, 57, 190,
                    15, 190, 133, 63, 0, 221, 201, 63, 162, 35, 159, 62, 90, 115, 204, 63,
                    26, 1, 143, 63, 141, 253, 53, 191, 51, 209, 186, 191, 216, 32, 197, 190,
                    180, 82, 108, 191, 154, 186, 100, 61, 186, 52, 224, 189, 23, 7, 237, 63,
                    213, 101, 24, 63, 192, 220, 149, 63, 190, 164, 167, 191, 162, 136, 219, 61,
                    214, 56, 58, 63, 129, 85, 57, 191, 67, 23, 77, 64, 84, 193, 220, 191,
                    134, 194, 56, 192, 122, 252, 118, 63, 65, 183, 235, 63, 177, 110, 112, 63,
                    155, 222, 115, 191, 206, 229, 27, 64, 29, 62, 137, 62, 231, 152, 233, 190,
                    129, 34, 126, 191, 255, 160, 22, 63, 161, 30, 236, 191, 90, 64, 95, 191,
                    49, 69, 144, 191, 153, 210, 152, 191, 227, 132, 168, 63, 132, 185, 46, 192,
                    144, 96, 200, 63, 255, 243, 80, 63, 25, 61, 8, 192, 26, 145, 12, 191,
                    179, 252, 63, 63, 231, 30, 30, 191, 109, 198, 134, 190, 85, 19, 194, 62,
                    128, 124, 171, 63, 125, 66, 32, 191, 152, 111, 205, 63, 19, 68, 220, 60,
                    221, 147, 65, 191, 144, 50, 196, 62, 162, 216, 5, 192, 74, 130, 182, 61,
                    166, 87, 62, 191, 250, 81, 41, 191, 87, 205, 205, 191, 40, 176, 167, 191,
                    217, 250, 93, 191, 32, 25, 67, 63, 34, 193, 190, 63, 129, 41, 109, 191,
                    157, 56, 45, 191, 110, 21, 161, 63, 132, 193, 107, 191, 15, 90, 83, 63,
                    243, 218, 132, 191, 65, 125, 219, 61, 151, 111, 72, 64, 177, 43, 43, 63,
                    4, 192, 221, 191, 4, 134, 143, 63, 214, 163, 103, 189, 53, 48, 215, 63,
                    247, 246, 119, 190, 187, 80, 167, 63, 113, 229, 161, 62, 3, 248, 246, 62,
                    198, 156, 168, 63, 86, 171, 127, 190, 9, 32, 58, 63, 121, 124, 253, 63,
                    33, 153, 44, 190, 25, 57, 179, 63, 170, 243, 67, 62, 3, 24, 76, 63,
                    249, 19, 123, 191, 227, 2, 200, 190, 147, 223, 14, 61, 39, 49, 105, 62,
                    203, 45, 171, 190, 29, 157, 217, 190, 115, 242, 4, 190, 148, 118, 216, 190,
                    100, 238, 20, 62, 244, 225, 228, 188, 108, 208, 44, 189, 152, 48, 164, 191,
                    161, 155, 187, 61, 243, 86, 70, 191, 88, 99, 81, 62, 110, 245, 103, 191,
                    182, 212, 213, 60, 230, 55, 219, 190, 39, 244, 93, 62, 252, 252, 21, 61,
                    202, 174, 73, 190, 88, 47, 59, 63, 247, 235, 108, 191, 244, 190, 86, 190,
                    22, 23, 122, 191, 211, 123, 58, 191, 189, 174, 9, 62, 85, 207, 128, 191,
                    52, 13, 178, 63, 158, 245, 214, 190, 136, 66, 4, 191, 40, 204, 49, 191,
                    40, 244, 30, 63, 140, 105, 30, 191, 51, 192, 109, 189, 97, 59, 149, 62,
                    118, 179, 78, 190, 4, 62, 146, 60, 33, 29, 7, 63, 83, 146, 115, 190,
                    196, 64, 94, 191, 9, 93, 117, 62, 107, 93, 41, 190, 239, 53, 7, 187,
                    146, 232, 200, 61, 224, 99, 46, 61, 8, 228, 152, 62, 152, 32, 55, 62,
                    97, 4, 132, 191, 6, 91, 5, 62, 14, 99, 33, 62, 111, 28, 80, 190,
                    57, 89, 127, 190, 127, 43, 133, 190, 81, 64, 177, 62, 239, 184, 151, 62,
                    91, 191, 21, 191, 20, 51, 0, 63, 75, 35, 161, 62, 112, 59, 254, 62,
                    83, 192, 24, 64, 239, 95, 137, 190, 203, 75, 43, 62, 159, 147, 252, 62,
                    87, 121, 123, 62, 163, 104, 5, 191, 220, 50, 222, 63, 17, 57, 37, 191,
                    145, 161, 223, 191, 3, 149, 122, 63, 204, 45, 156, 189, 64, 76, 9, 191,
                    202, 246, 239, 190, 202, 251, 226, 62, 15, 83, 175, 191, 78, 97, 87, 62,
                    185, 165, 183, 62, 6, 6, 135, 191, 144, 73, 133, 63, 44, 185, 126, 63,
                    201, 73, 140, 63, 36, 15, 13, 64, 243, 21, 2, 64, 212, 91, 197, 62,
                    206, 84, 119, 190, 225, 248, 228, 190, 26, 190, 170, 62, 187, 21, 40, 63,
                    127, 10, 3, 190, 99, 170, 230, 189, 162, 70, 222, 190, 150, 212, 225, 62,
                    39, 64, 245, 63, 174, 143, 173, 62, 230, 138, 56, 63, 179, 48, 230, 190,
                    237, 249, 49, 62, 168, 153, 94, 191, 161, 75, 28, 191, 184, 78, 28, 62,
                    215, 91, 244, 187, 254, 232, 29, 63, 12, 233, 65, 62, 186, 1, 187, 190,
                    41, 74, 203, 60, 120, 223, 253, 62, 163, 71, 53, 62, 194, 102, 192, 190,
                    30, 8, 45, 191, 148, 5, 79, 189, 41, 43, 45, 63, 138, 57, 136, 62,
                    68, 139, 21, 63, 118, 223, 199, 189, 196, 110, 231, 62, 223, 195, 121, 189,
                    226, 126, 155, 60, 218, 249, 14, 63, 221, 114, 176, 190, 199, 124, 76, 63,
                    237, 55, 20, 191, 123, 182, 234, 62, 44, 123, 130, 188, 159, 254, 109, 190,
                    113, 221, 186, 61, 8, 171, 122, 62, 248, 23, 167, 62, 143, 6, 59, 62,
                    49, 148, 129, 62, 215, 15, 176, 60, 129, 51, 1, 191, 188, 57, 30, 190,
                    222, 236, 86, 190, 217, 63, 58, 189, 54, 68, 11, 63, 104, 0, 212, 189,
                    64, 137, 52, 63, 34, 98, 172, 62, 77, 184, 87, 191, 207, 8, 132, 63,
                    222, 69, 31, 190, 22, 16, 2, 191, 92, 216, 59, 63, 110, 129, 77, 62,
                    56, 150, 143, 191, 207, 184, 79, 62, 190, 230, 0, 62, 116, 108, 195, 62,
                    60, 235, 169, 60, 214, 77, 91, 192, 173, 171, 87, 191, 14, 150, 210, 191,
                    152, 194, 215, 62, 44, 113, 192, 63, 156, 252, 16, 192, 186, 225, 117, 191,
                    85, 159, 10, 61, 29, 56, 10, 62, 98, 89, 64, 191, 229, 54, 99, 63,
                    215, 203, 205, 63, 60, 72, 251, 189, 153, 53, 248, 63, 138, 140, 179, 191,
                    237, 202, 15, 63, 251, 184, 160, 62, 198, 231, 149, 63, 33, 86, 119, 191,
                    150, 158, 12, 191, 75, 82, 189, 62, 83, 92, 59, 63, 0, 13, 223, 63,
                    22, 206, 62, 63, 205, 48, 32, 62, 163, 117, 59, 63, 226, 92, 199, 191,
                    199, 17, 221, 191, 66, 218, 209, 190, 43, 150, 22, 191, 84, 221, 37, 190,
                    156, 92, 143, 62, 64, 175, 78, 191, 50, 217, 166, 62, 97, 185, 226, 188,
                    25, 181, 159, 189, 161, 35, 185, 190, 173, 206, 159, 62, 8, 238, 138, 62,
                    85, 239, 167, 191, 48, 53, 106, 63, 148, 78, 100, 189, 216, 154, 89, 62,
                    82, 187, 55, 191, 10, 150, 138, 62, 245, 126, 170, 190, 146, 227, 156, 62,
                    142, 205, 246, 190, 235, 176, 144, 62, 189, 230, 47, 190, 8, 250, 198, 63,
                    112, 18, 38, 63, 149, 120, 68, 63, 78, 245, 158, 191, 71, 197, 199, 191,
                    228, 136, 36, 63, 240, 24, 144, 189, 43, 7, 119, 63, 2, 59, 93, 62,
                    93, 64, 245, 188, 148, 94, 13, 191, 88, 246, 130, 190, 166, 169, 36, 191,
                    158, 175, 173, 62, 53, 164, 29, 63, 59, 200, 152, 190, 13, 120, 203, 189,
                    41, 151, 28, 63, 236, 183, 38, 63, 70, 175, 224, 61, 38, 8, 109, 191,
                    76, 18, 106, 191, 88, 188, 231, 62, 63, 192, 10, 63, 101, 33, 178, 63,
                    229, 15, 103, 63, 176, 11, 9, 191, 208, 249, 82, 191, 74, 221, 20, 189,
                    6, 139, 161, 61, 161, 146, 122, 62, 151, 14, 205, 62, 46, 0, 124, 191,
                    205, 102, 119, 63, 35, 216, 71, 63, 157, 227, 184, 190, 176, 99, 197, 61,
                    81, 9, 30, 61, 81, 217, 141, 62, 207, 185, 215, 63, 179, 6, 47, 191,
                    126, 102, 55, 191, 88, 74, 6, 63, 47, 228, 128, 190, 6, 253, 216, 190,
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
                    215, 200, 43, 62, 107, 21, 22, 63, 22, 174, 4, 63, 176, 172, 192, 189,
                    24, 48, 222, 62, 85, 181, 53, 63, 23, 195, 19, 191, 126, 99, 53, 62,
                    207, 145, 72, 191, 75, 118, 157, 188, 167, 218, 78, 64, 51, 211, 194, 190,
                    246, 238, 210, 188, 245, 15, 189, 63, 34, 235, 150, 191, 118, 240, 186, 190,
                    158, 181, 251, 191, 38, 118, 145, 191, 34, 76, 55, 192, 140, 117, 142, 191,
                    136, 169, 32, 192, 192, 60, 46, 191, 248, 104, 206, 191, 73, 207, 127, 192,
                    70, 55, 52, 192, 145, 1, 26, 192, 20, 44, 237, 191, 180, 68, 74, 63,
                    25, 189, 65, 192, 78, 69, 17, 192, 118, 196, 13, 64, 254, 22, 10, 192,
                    195, 122, 113, 63, 203, 209, 152, 62, 203, 210, 158, 62, 98, 218, 182, 62,
                    50, 230, 10, 62, 53, 128, 62, 192, 29, 7, 85, 64, 203, 118, 86, 189,
                    165, 87, 42, 62, 102, 117, 170, 62, 80, 158, 30, 192, 233, 151, 90, 62,
                    128, 123, 155, 61, 44, 3, 80, 191, 124, 33, 11, 63, 216, 173, 167, 61,
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
                    57, 50, 158, 62, 225, 69, 15, 63, 208, 130, 197, 61, 212, 47, 161, 60,
                    165, 161, 173, 61, 200, 215, 150, 62, 251, 97, 91, 191, 87, 251, 41, 62,
                    245, 120, 128, 191, 251, 88, 171, 190, 22, 17, 57, 64, 220, 215, 72, 190,
                    34, 150, 133, 190, 226, 131, 190, 63, 195, 42, 162, 191, 169, 167, 205, 189,
                    48, 190, 10, 192, 208, 109, 122, 191, 242, 167, 71, 192, 244, 139, 179, 191,
                    150, 249, 26, 192, 110, 11, 114, 191, 7, 212, 199, 191, 61, 37, 125, 192,
                    1, 134, 69, 192, 224, 33, 33, 192, 131, 141, 14, 192, 84, 39, 129, 63,
                    77, 173, 78, 192, 241, 247, 12, 192, 183, 106, 9, 64, 117, 154, 23, 192,
                    186, 142, 193, 62, 20, 135, 85, 190, 198, 93, 234, 62, 32, 163, 156, 61,
                    168, 246, 152, 62, 168, 141, 41, 192, 208, 100, 147, 63, 24, 195, 44, 190,
                    194, 237, 154, 61, 230, 158, 158, 188, 30, 116, 23, 192, 117, 211, 65, 190,
                    147, 133, 81, 189, 7, 93, 10, 192, 227, 233, 126, 191, 117, 246, 31, 191,
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
                    219, 156, 7, 63, 195, 125, 161, 64, 221, 183, 72, 64, 217, 99, 169, 190,
                    53, 127, 243, 59, 174, 53, 237, 191, 43, 84, 232, 63, 252, 60, 129, 62,
                    96, 203, 227, 63, 53, 251, 4, 64, 252, 252, 2, 192, 111, 38, 32, 63,
                    161, 138, 155, 64, 134, 240, 6, 192, 20, 188, 21, 192, 38, 44, 106, 64,
                    28, 203, 37, 64, 137, 27, 19, 192, 199, 183, 159, 63, 19, 26, 168, 191,
                    215, 236, 191, 64, 166, 48, 32, 192, 170, 32, 23, 64, 35, 174, 180, 192,
                    58, 118, 128, 192, 206, 38, 38, 63, 164, 121, 28, 192, 152, 191, 115, 63,
                    250, 126, 225, 63, 138, 39, 23, 192, 161, 108, 168, 61, 151, 42, 146, 63,
                    191, 34, 10, 63, 97, 241, 49, 64, 142, 62, 64, 64, 27, 60, 126, 64,
                    110, 95, 122, 64, 32, 110, 10, 192, 20, 103, 17, 64, 249, 156, 152, 63,
                    16, 106, 237, 191, 189, 157, 11, 64, 18, 1, 244, 191, 216, 156, 190, 62,
                    255, 80, 16, 192, 129, 80, 130, 191, 140, 55, 151, 191, 239, 110, 120, 64,
                    86, 206, 70, 64, 98, 106, 128, 192, 89, 144, 38, 191, 45, 140, 129, 64,
                    147, 37, 55, 190, 216, 143, 235, 191, 141, 126, 41, 64, 172, 204, 101, 192,
                    135, 33, 234, 63, 228, 109, 140, 190, 79, 106, 217, 191, 212, 118, 50, 191,
                    248, 146, 103, 64, 195, 172, 45, 192, 138, 79, 101, 191, 22, 187, 53, 64,
                    149, 92, 21, 64, 226, 82, 46, 192, 108, 48, 0, 189, 239, 206, 181, 63,
                    174, 96, 130, 64, 26, 214, 172, 191, 160, 239, 212, 63, 177, 125, 204, 191,
                    228, 244, 62, 190, 199, 228, 78, 63, 235, 55, 219, 191, 25, 145, 20, 192,
                    141, 180, 26, 192, 176, 163, 168, 191, 62, 245, 138, 62, 63, 112, 150, 192,
                    6, 125, 128, 62, 243, 138, 237, 63, 170, 75, 234, 63, 123, 104, 6, 64,
                    151, 210, 37, 192, 223, 246, 171, 191, 146, 44, 170, 63, 247, 221, 240, 63,
                    222, 34, 106, 64, 18, 244, 249, 63, 238, 220, 142, 191, 7, 214, 135, 192,
                    47, 194, 236, 190, 209, 140, 133, 191, 27, 187, 27, 192, 203, 222, 159, 62,
                    85, 218, 239, 63, 253, 149, 14, 192, 2, 93, 215, 190, 190, 0, 173, 191,
                    234, 43, 59, 191, 139, 212, 132, 191, 18, 77, 156, 63, 34, 172, 1, 192,
                    171, 214, 7, 64, 184, 38, 252, 63, 119, 41, 177, 191, 72, 83, 68, 192,
                    5, 61, 73, 64, 86, 142, 227, 191, 113, 173, 52, 191, 99, 60, 86, 192,
                    235, 41, 238, 190, 205, 28, 124, 64, 108, 31, 22, 64, 51, 231, 60, 191,
                    53, 46, 26, 64, 158, 96, 103, 191, 224, 95, 83, 63, 3, 90, 65, 63,
                    149, 17, 5, 62, 238, 21, 88, 64, 144, 164, 132, 191, 75, 240, 46, 191,
                    210, 78, 248, 191, 67, 151, 24, 191, 248, 164, 55, 191, 190, 223, 56, 192,
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
                    55, 213, 212, 63, 215, 32, 57, 64, 102, 12, 178, 63, 15, 96, 170, 63,
                    77, 115, 189, 63, 221, 236, 8, 63, 224, 209, 175, 63, 129, 32, 97, 63,
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
        117, 185, 211, 62, 158, 68, 11, 62, 137, 82, 10, 63, 115, 121, 142, 61,
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
