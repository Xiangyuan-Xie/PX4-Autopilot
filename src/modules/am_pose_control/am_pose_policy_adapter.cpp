/****************************************************************************
 *
 * Optional AM Pose RL-Tools policy adapter.
 *
 ****************************************************************************/

#include "am_pose_policy_adapter.hpp"

#if __has_include("blob/policy.h")
#define AM_POSE_POLICY_BLOB_AVAILABLE 1

#include <rl_tools/operations/arm.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/containers/tensor/tensor.h>

#include "am_pose_policy_contract.hpp"
#include "blob/policy.h"
#endif

#ifndef AM_POSE_POLICY_BLOB_AVAILABLE
#define AM_POSE_POLICY_BLOB_AVAILABLE 0
#endif

#if AM_POSE_POLICY_BLOB_AVAILABLE
namespace pose_rlt = rl_tools;

struct AmPosePolicyAdapter::Impl {
	using DeviceSpec = pose_rlt::devices::DefaultARMSpecification;
	using Device = pose_rlt::devices::arm::OPT<DeviceSpec>;
	using Index = Device::index_t;
	using Rng = Device::SPEC::RANDOM::ENGINE<>;
	using Policy = pose_rlt::checkpoint::actor::TYPE;
	using Mode = pose_rlt::Mode<pose_rlt::nn::layers::gru::NoAutoResetMode<pose_rlt::mode::Evaluation<>>>;

	static constexpr int kPolicyObservationDim = static_cast<int>(Policy::INPUT_SHAPE::LAST);
	static constexpr uint8_t kArmObservationFeatures = am_pose_policy_manifest::kArmObservationFeatures;
	static_assert(am_pose_policy_manifest::kObservationDim == kPolicyObservationDim,
		      "AM Pose policy manifest observation_dim does not match policy input shape");
	static constexpr ObservationContract kObservationContract =
		AmPosePolicyAdapter::observationContractFor(kPolicyObservationDim, kArmObservationFeatures);
	static_assert(kObservationContract != ObservationContract::Unsupported,
		      "AM Pose manifest arm_observation must match policy input dimension: "
		      "none=130, position=134, velocity=134, position_velocity=138");
	static_assert(static_cast<int>(Policy::OUTPUT_SHAPE::LAST) == ActionDim,
		      "AM Pose policy action dim mismatch");

	using InputShape = pose_rlt::tensor::Shape<Index, 1, kPolicyObservationDim>;
	using OutputShape = pose_rlt::tensor::Shape<Index, 1, ActionDim>;
	using InputTensor = pose_rlt::Tensor<pose_rlt::tensor::Specification<float, Index, InputShape, false>>;
	using OutputTensor = pose_rlt::Tensor<pose_rlt::tensor::Specification<float, Index, OutputShape, false>>;

	Device device{};
	Rng rng{};
	Mode mode{};
	Policy::template Buffer<false> policy_buffer{};
	Policy::State<false> policy_state{};
	InputTensor input{};
	OutputTensor output{};
};
#else
struct AmPosePolicyAdapter::Impl {};
#endif

AmPosePolicyAdapter::~AmPosePolicyAdapter()
{
	delete _impl;
}

bool AmPosePolicyAdapter::init()
{
#if AM_POSE_POLICY_BLOB_AVAILABLE

	if (_impl == nullptr) {
		_impl = new Impl{};

		if (_impl == nullptr) {
			return false;
		}

		pose_rlt::init(_impl->device);
		pose_rlt::init(_impl->device, _impl->rng, 0);
	}

	reset();
	return true;
#else
	return false;
#endif
}

void AmPosePolicyAdapter::reset()
{
#if AM_POSE_POLICY_BLOB_AVAILABLE

	if (_impl == nullptr) {
		return;
	}

	pose_rlt::reset(_impl->device, pose_rlt::checkpoint::actor::module, _impl->policy_state, _impl->rng);
#endif
}

AmPosePolicyAdapter::ObservationContract AmPosePolicyAdapter::observationContract() const
{
#if AM_POSE_POLICY_BLOB_AVAILABLE
	return Impl::kObservationContract;
#else
	return ObservationContract::Unsupported;
#endif
}

bool AmPosePolicyAdapter::infer(const Observation &observation, Action &action)
{
#if AM_POSE_POLICY_BLOB_AVAILABLE

	if (_impl == nullptr) {
		return false;
	}

	for (int i = 0; i < Impl::kPolicyObservationDim; ++i) {
		pose_rlt::set(_impl->device, _impl->input, observation[i], 0, i);
	}

	pose_rlt::evaluate_step(_impl->device, pose_rlt::checkpoint::actor::module, _impl->input,
				_impl->policy_state, _impl->output, _impl->policy_buffer, _impl->rng, _impl->mode);

	for (int i = 0; i < ActionDim; ++i) {
		action[i] = pose_rlt::get(_impl->device, _impl->output, 0, i);
	}

	return true;
#else
	(void)observation;
	(void)action;
	return false;
#endif
}
