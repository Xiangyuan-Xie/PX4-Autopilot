#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/containers/tensor/tensor.h>

#include "blob/policy.h"
#include "rl_tools_adapter.hpp"

namespace
{
namespace rlt = rl_tools;

using DeviceSpec = rlt::devices::DefaultARMSpecification;
using Device = rlt::devices::arm::OPT<DeviceSpec>;
using Index = Device::index_t;
using Rng = Device::SPEC::RANDOM::ENGINE<>;
using Policy = rlt::checkpoint::actor::TYPE;
using Mode = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Evaluation<>>>;

static constexpr int kObservationDim = static_cast<int>(Policy::INPUT_SHAPE::LAST);
static constexpr int kActionDim = static_cast<int>(Policy::OUTPUT_SHAPE::LAST);

using InputShape = rlt::tensor::Shape<Index, 1, kObservationDim>;
using OutputShape = rlt::tensor::Shape<Index, 1, kActionDim>;
using InputTensor = rlt::Tensor<rlt::tensor::Specification<float, Index, InputShape, false>>;
using OutputTensor = rlt::Tensor<rlt::tensor::Specification<float, Index, OutputShape, false>>;

static_assert(kObservationDim == 30, "Exported policy observation dim mismatch");
static_assert(kActionDim == 4, "Exported policy action dim mismatch");
static_assert(RlToolsAdapter::ObservationDim == kObservationDim, "Adapter observation dim mismatch");
static_assert(RlToolsAdapter::ActionDim == kActionDim, "Adapter action dim mismatch");

using ObservationArray = std::array<float, kObservationDim>;
using ActionArray = std::array<float, kActionDim>;

template <typename T, int Size>
std::array<T, Size> exportedArray(const unsigned char (&memory)[Size * sizeof(T)])
{
	std::array<T, Size> values{};
	std::memcpy(values.data(), memory, Size * sizeof(T));
	return values;
}

ObservationArray exportedExampleObservation()
{
	return exportedArray<float, kObservationDim>(rlt::checkpoint::example::input::memory);
}

ActionArray exportedExampleOutput()
{
	return exportedArray<float, kActionDim>(rlt::checkpoint::example::output::memory);
}

ActionArray evaluateSingleStep(const ObservationArray &observation)
{
	ActionArray result{};

	Device device{};
	Rng rng{};
	Mode mode{};
	Policy::template Buffer<false> policy_buffer{};
	Policy::State<false> policy_state{};
	InputTensor input{};
	OutputTensor output{};

	rlt::init(device);
	rlt::init(device, rng, 0);
	rlt::reset(device, rlt::checkpoint::actor::module, policy_state, rng);

	for (int i = 0; i < kObservationDim; ++i) {
		rlt::set(device, input, observation[i], 0, i);
	}

	rlt::evaluate_step(
		device,
		rlt::checkpoint::actor::module,
		input,
		policy_state,
		output,
		policy_buffer,
		rng,
		mode);

	for (int i = 0; i < kActionDim; ++i) {
		result[i] = rlt::get(device, output, 0, i);
	}

	return result;
}

ActionArray inferWithAdapter(RlToolsAdapter &adapter, uint64_t now_us, const ObservationArray &observation)
{
	RlToolsAdapter::Observation adapter_observation{};
	RlToolsAdapter::Action adapter_action{};

	for (int i = 0; i < kObservationDim; ++i) {
		adapter_observation[i] = observation[i];
	}

	EXPECT_TRUE(adapter.infer(now_us, adapter_observation, adapter_action));

	ActionArray action{};
	for (int i = 0; i < kActionDim; ++i) {
		action[i] = adapter_action[i];
	}

	return action;
}

void expectActionNear(const ActionArray &actual, const ActionArray &expected)
{
	for (int i = 0; i < kActionDim; ++i) {
		EXPECT_NEAR(actual[i], expected[i], 5e-3f) << "action[" << i << "]";
	}
}
} // namespace

TEST(AmPosControlPolicyBlobTest, ExampleObservationMatchesExportedReferenceOutput)
{
	expectActionNear(evaluateSingleStep(exportedExampleObservation()), exportedExampleOutput());
}

TEST(AmPosControlPolicyBlobTest, AdapterFirstInferenceMatchesExportedReferenceOutput)
{
	RlToolsAdapter adapter{};
	ASSERT_TRUE(adapter.init());

	expectActionNear(inferWithAdapter(adapter, 1000, exportedExampleObservation()), exportedExampleOutput());
}

TEST(AmPosControlPolicyBlobTest, AdapterResetRestoresInitialHiddenState)
{
	RlToolsAdapter adapter{};
	ASSERT_TRUE(adapter.init());

	(void)inferWithAdapter(adapter, 1000, exportedExampleObservation());
	(void)inferWithAdapter(adapter, 11000, exportedExampleObservation());

	adapter.reset();

	expectActionNear(inferWithAdapter(adapter, 21000, exportedExampleObservation()), exportedExampleOutput());
}
