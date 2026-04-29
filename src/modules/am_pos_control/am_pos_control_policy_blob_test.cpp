#include <array>

#include <gtest/gtest.h>

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/containers/tensor/tensor.h>

#include "blob/policy.h"

namespace
{
namespace rlt = rl_tools;

using DeviceSpec = rlt::devices::DefaultARMSpecification;
using Device = rlt::devices::arm::OPT<DeviceSpec>;
using Index = Device::index_t;
using Rng = Device::SPEC::RANDOM::ENGINE<>;
using Policy = rlt::checkpoint::actor::TYPE;
using Mode = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Evaluation<>>>;

static_assert(static_cast<int>(Policy::INPUT_SHAPE::LAST) == 35, "Exported policy observation dim mismatch");
static_assert(static_cast<int>(Policy::OUTPUT_SHAPE::LAST) == 4, "Exported policy action dim mismatch");

using ObservationArray = std::array<float, Policy::INPUT_SHAPE::LAST>;
using ActionArray = std::array<float, Policy::OUTPUT_SHAPE::LAST>;
using InputShape = rlt::tensor::Shape<Index, 1, Policy::INPUT_SHAPE::LAST>;
using OutputShape = rlt::tensor::Shape<Index, 1, Policy::OUTPUT_SHAPE::LAST>;
using InputTensor = rlt::Tensor<rlt::tensor::Specification<float, Index, InputShape, false>>;
using OutputTensor = rlt::Tensor<rlt::tensor::Specification<float, Index, OutputShape, false>>;

constexpr ObservationArray kResetObservation{};

constexpr ObservationArray kHoverObservation{
	-0.0021454424f, -0.0088739814f, 0.1204682f, 0.9908884f, -0.0022313562f,
	-0.0016603493f, 0.0022796283f, 0.9943894f, -0.0004909930f, 0.0013912847f,
	0.0000963059f, 0.9937867f, 0.0031632942f, 0.0019838926f, -0.9852334f,
	-0.0140372198f, -0.0259357002f, 0.3152920f, -0.0012739807f, -0.0233770497f,
	0.0076213628f, -0.0033981067f, 1.8803737f, 0.0479082949f, -0.0045538712f,
	-0.0875884742f, -0.0020537272f, 0.0132233873f, -0.0091587165f, -0.0000779090f,
	-0.0036016428f, 0.91430616f, 0.83621335f, 0.90553510f, 0.83218575f
};

constexpr ObservationArray kExampleObservation{
	-1.0f, -0.94117647f, -0.88235295f, -0.82352942f, -0.76470590f,
	-0.70588237f, -0.64705884f, -0.58823532f, -0.52941179f, -0.47058824f,
	-0.41176471f, -0.35294119f, -0.29411763f, -0.23529412f, -0.17647058f,
	-0.11764705f, -0.05882353f, -0.00000000f, 0.05882353f, 0.11764705f,
	0.17647058f, 0.23529412f, 0.29411763f, 0.35294119f, 0.41176471f,
	0.47058824f, 0.52941179f, 0.58823532f, 0.64705884f, 0.70588237f,
	0.76470590f, 0.82352942f, 0.88235295f, 0.94117647f, 1.0f
};

constexpr ActionArray kExpectedResetOutput{
	1.05682886f, 0.82696760f, -0.31302094f, 0.22636014f
};

constexpr ActionArray kExpectedHoverOutput{
	1.61772561f, 1.81442213f, 1.70429087f, 0.67699170f
};

constexpr ActionArray kExpectedExampleOutput{
	0.79740399f, 0.91717404f, -0.15924889f, 0.12748112f
};

ActionArray evaluateSingleStep(const ObservationArray &observation)
{
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

	for (int i = 0; i < static_cast<int>(Policy::INPUT_SHAPE::LAST); ++i) {
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

	ActionArray result{};

	for (int i = 0; i < static_cast<int>(Policy::OUTPUT_SHAPE::LAST); ++i) {
		result[i] = rlt::get(device, output, 0, i);
	}

	return result;
}

void expectActionNear(const ActionArray &actual, const ActionArray &expected)
{
	for (int i = 0; i < static_cast<int>(Policy::OUTPUT_SHAPE::LAST); ++i) {
		EXPECT_NEAR(actual[i], expected[i], 5e-2f) << "action[" << i << "]";
	}
}
} // namespace

TEST(AmPosControlPolicyBlobTest, ResetObservationMatchesPolicyJsonCompatOutput)
{
	expectActionNear(evaluateSingleStep(kResetObservation), kExpectedResetOutput);
}

TEST(AmPosControlPolicyBlobTest, HoverObservationMatchesPolicyJsonCompatOutput)
{
	expectActionNear(evaluateSingleStep(kHoverObservation), kExpectedHoverOutput);
}

TEST(AmPosControlPolicyBlobTest, ExampleObservationMatchesPolicyJsonExampleOutput)
{
	expectActionNear(evaluateSingleStep(kExampleObservation), kExpectedExampleOutput);
}
