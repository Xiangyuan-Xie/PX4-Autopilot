#include <gtest/gtest.h>

#include "am_pose_policy_adapter.hpp"

#include <cmath>

static_assert(AmPosePolicyAdapter::ActionDim == 4, "AM Pose policy output must contain four motors");

#if __has_include("blob/policy.h")

TEST(AmPoseControlPolicyBlobTest, CompatibleBlobAdvancesOnEveryCall)
{
	AmPosePolicyAdapter adapter{};
	ASSERT_TRUE(adapter.init());
	ASSERT_TRUE(adapter.available());
	EXPECT_NE(adapter.observationContract(), AmPosePolicyAdapter::ObservationContract::Unsupported);
	EXPECT_TRUE(adapter.observationDim() == AmPosePolicyAdapter::Pose130ObservationDim
		    || adapter.observationDim() == AmPosePolicyAdapter::Pose134ObservationDim
		    || adapter.observationDim() == AmPosePolicyAdapter::Pose138ObservationDim);
	EXPECT_NE(AmPosePolicyAdapter::observationContractFor(adapter.observationDim(),
		  AmPosePolicyAdapter::armObservationFeaturesFor(adapter.observationContract())),
	  AmPosePolicyAdapter::ObservationContract::Unsupported);

	AmPosePolicyAdapter::Observation observation{};
	AmPosePolicyAdapter::Action action{};
	ASSERT_TRUE(adapter.infer(observation, action));

	for (float value : action) {
		EXPECT_TRUE(std::isfinite(value));
	}

	ASSERT_TRUE(adapter.infer(observation, action));
}

#else

TEST(AmPoseControlPolicyBlobTest, MissingBlobLeavesAdapterUnavailable)
{
	AmPosePolicyAdapter adapter{};
	EXPECT_FALSE(adapter.init());
	EXPECT_FALSE(adapter.available());
	EXPECT_EQ(adapter.observationContract(), AmPosePolicyAdapter::ObservationContract::Unsupported);

	AmPosePolicyAdapter::Observation observation{};
	AmPosePolicyAdapter::Action action{};
	EXPECT_FALSE(adapter.infer(observation, action));
}

#endif
