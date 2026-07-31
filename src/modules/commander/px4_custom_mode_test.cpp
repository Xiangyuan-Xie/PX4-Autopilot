#include <gtest/gtest.h>

#include "Commander.hpp"

#define DEFINE_GET_PX4_CUSTOM_MODE
#include "px4_custom_mode.h"

#include <lib/modes/ui.hpp>
#include <uORB/topics/vehicle_status.h>

TEST(Px4CustomModeTest, EncodesAmPoseAsDedicatedPosctlSubmode)
{
	const px4_custom_mode custom_mode = get_px4_custom_mode(vehicle_status_s::NAVIGATION_STATE_AM_POSE);

	EXPECT_EQ(custom_mode.main_mode, PX4_CUSTOM_MAIN_MODE_POSCTL);
	EXPECT_EQ(custom_mode.sub_mode, 3);
}

TEST(Px4CustomModeTest, AmPoseIsAdvertisedAsRegularMode)
{
	EXPECT_FALSE(mode_util::isAdvanced(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
}

TEST(Px4CustomModeTest, UsesPositionIntentionForArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_ALTCTL,
			  vehicle_status_s::NAVIGATION_STATE_POSCTL),
		  vehicle_status_s::NAVIGATION_STATE_POSCTL);
}

TEST(Px4CustomModeTest, UsesAmPoseIntentionForArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_DESCEND,
			  vehicle_status_s::NAVIGATION_STATE_AM_POSE),
		  vehicle_status_s::NAVIGATION_STATE_AM_POSE);
}

TEST(Px4CustomModeTest, KeepsCurrentNavStateForOtherArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_ALTCTL,
			  vehicle_status_s::NAVIGATION_STATE_MANUAL),
		  vehicle_status_s::NAVIGATION_STATE_ALTCTL);
}
