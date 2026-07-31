#include <gtest/gtest.h>

#include "mode_requirements.hpp"
#include <lib/modes/ui.hpp>
#include <uORB/topics/vehicle_status.h>

TEST(ModeRequirementsTest, AddsAmPoseRequirementsForDedicatedNavState)
{
	failsafe_flags_s flags{};

	mode_util::getModeRequirements(vehicle_status_s::VEHICLE_TYPE_ROTARY_WING, flags);

	const uint32_t am_pose_mask = 1u << vehicle_status_s::NAVIGATION_STATE_AM_POSE;

	EXPECT_NE(flags.mode_req_angular_velocity & am_pose_mask, 0u);
	EXPECT_NE(flags.mode_req_attitude & am_pose_mask, 0u);
	EXPECT_NE(flags.mode_req_local_position & am_pose_mask, 0u);
	EXPECT_NE(flags.mode_req_local_alt & am_pose_mask, 0u);
	EXPECT_NE(flags.mode_req_manual_control & am_pose_mask, 0u);
	EXPECT_NE(flags.mode_req_other & am_pose_mask, 0u);
}

TEST(ModeRequirementsTest, AmPoseUsesNavStateSlot11AndIsSelectable)
{
	EXPECT_EQ(vehicle_status_s::NAVIGATION_STATE_AM_POSE, 11);
	EXPECT_NE(mode_util::getValidNavStates() & (1u << vehicle_status_s::NAVIGATION_STATE_AM_POSE), 0u);
}

TEST(ModeRequirementsTest, AmPoseIsAdvertisedAsRegularMode)
{
	EXPECT_FALSE(mode_util::isAdvanced(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
}
