/****************************************************************************
 *
 * rl_tools AM Position policy backend.
 *
 ****************************************************************************/
#pragma once

#include <rl_tools/operations/arm.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/containers/tensor/tensor.h>

#include "blob/policy.h"
#include "policy_types.hpp"

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

	static_assert(ObservationDim == kPolicyObservationDim, "rl_tools policy observation dim mismatch");
	static_assert(ActionDim == kPolicyActionDim, "rl_tools policy action dim mismatch");

	using Observation = PolicyObservation;
	using Action = PolicyAction;

	bool init();
	void reset();
	bool infer(const Observation &observation, Action &action);

private:
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
	InputTensor _input{};
	OutputTensor _output{};
};
