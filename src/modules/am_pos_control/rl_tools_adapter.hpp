/****************************************************************************
 *
 * Minimal RAPTOR-style adapter shim.
 *
 ****************************************************************************/
#pragma once

#include <cstdint>

#include <rl_tools/operations/arm.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/containers/tensor/tensor.h>

#include "blob/policy.h"

namespace rlt = rl_tools;

class RlToolsAdapter
{
public:
	using DeviceSpec = rlt::devices::DefaultARMSpecification;
	using Device = rlt::devices::arm::OPT<DeviceSpec>;
	using Index = Device::index_t;
	using Rng = Device::SPEC::RANDOM::ENGINE<>;
	using Policy = rlt::checkpoint::actor::TYPE;
	using Mode = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Evaluation<>>>;

	static constexpr int ObservationDim = static_cast<int>(Policy::INPUT_SHAPE::LAST);
	static constexpr int ActionDim = static_cast<int>(Policy::OUTPUT_SHAPE::LAST);

	using Observation = float[ObservationDim];
	using Action = float[ActionDim];

	bool init();
	void reset();
	bool infer(uint64_t now_us, const Observation &observation, Action &action);

private:
	// Mirror the RAPTOR/L2F timing model locally: run a committed/native policy step at
	// 100 Hz, allow intermediate evaluations in between, and keep the true hidden state
	// unchanged on intermediate steps.
	static constexpr uint64_t kIntermediateStepUs = 2'500;
	static constexpr uint64_t kNativeStepUs = 10'000;
	static constexpr uint8_t kForceSyncNative = 4;
	uint64_t _last_control_step_us{0};
	uint64_t _last_native_step_us{0};
	uint8_t _intermediate_steps_since_native{0};
	Action _last_action{0.f, 0.f, 0.f, 0.f};
	bool _runtime_initialized{false};

	using InputShape = rlt::tensor::Shape<Index, 1, ObservationDim>;
	using OutputShape = rlt::tensor::Shape<Index, 1, ActionDim>;
	using InputTensor = rlt::Tensor<rlt::tensor::Specification<float, Index, InputShape, false>>;
	using OutputTensor = rlt::Tensor<rlt::tensor::Specification<float, Index, OutputShape, false>>;

	Device _device{};
	Rng _rng{};
	Mode _mode{};
	Policy::template Buffer<false> _policy_buffer{};
	Policy::State<false> _policy_state{};
	Policy::State<false> _policy_state_temp{};
	InputTensor _input{};
	OutputTensor _output{};
};
