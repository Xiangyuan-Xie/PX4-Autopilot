#include <gtest/gtest.h>

#include "Commander.hpp"

#define DEFINE_GET_PX4_CUSTOM_MODE
#include "px4_custom_mode.h"

#include <lib/modes/ui.hpp>
#include <uORB/topics/vehicle_status.h>

TEST(Px4CustomModeTest, EncodesAmPositionAsDedicatedPosctlSubmode)
{
	const px4_custom_mode custom_mode = get_px4_custom_mode(vehicle_status_s::NAVIGATION_STATE_AM_POSITION);

	EXPECT_EQ(custom_mode.main_mode, PX4_CUSTOM_MAIN_MODE_POSCTL);
	EXPECT_EQ(custom_mode.sub_mode, 3);
}

TEST(Px4CustomModeTest, EncodesAmOffboardAsDedicatedOffboardSubmode)
{
	const px4_custom_mode custom_mode = get_px4_custom_mode(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_EQ(custom_mode.main_mode, PX4_CUSTOM_MAIN_MODE_OFFBOARD);
	EXPECT_EQ(custom_mode.sub_mode, 1);
}

TEST(Px4CustomModeTest, EncodesAmTestAsDedicatedPosctlSubmode)
{
	const px4_custom_mode custom_mode = get_px4_custom_mode(vehicle_status_s::NAVIGATION_STATE_AM_TEST);

	EXPECT_EQ(custom_mode.main_mode, PX4_CUSTOM_MAIN_MODE_POSCTL);
	EXPECT_EQ(custom_mode.sub_mode, PX4_CUSTOM_SUB_MODE_POSCTL_AM_TEST);
	EXPECT_NE(custom_mode.sub_mode, PX4_CUSTOM_SUB_MODE_POSCTL_ORBIT);
	EXPECT_NE(custom_mode.sub_mode, PX4_CUSTOM_SUB_MODE_POSCTL_SLOW);
	EXPECT_NE(custom_mode.sub_mode, PX4_CUSTOM_SUB_MODE_POSCTL_AM_POSITION);
}

TEST(Px4CustomModeTest, MarksAmOffboardNotSelectableWhenItCannotRunYet)
{
	const uint32_t am_position_bit = 1u << vehicle_status_s::NAVIGATION_STATE_AM_POSITION;
	const uint32_t am_offboard_bit = 1u << vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;
	const uint32_t can_set_nav_states_mask = am_position_bit | am_offboard_bit;

	const uint32_t filtered_mask = Commander::filterCanSetNavStatesForAmVisibility(can_set_nav_states_mask, false);

	EXPECT_EQ(filtered_mask & am_position_bit, am_position_bit);
	EXPECT_EQ(filtered_mask & am_offboard_bit, 0u);
}

TEST(Px4CustomModeTest, KeepsAmOffboardSelectableWhenItCanRun)
{
	const uint32_t am_offboard_bit = 1u << vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;

	const uint32_t filtered_mask = Commander::filterCanSetNavStatesForAmVisibility(am_offboard_bit, true);

	EXPECT_EQ(filtered_mask & am_offboard_bit, am_offboard_bit);
}

TEST(Px4CustomModeTest, AmOffboardIsAdvertisedAsRegularMode)
{
	EXPECT_FALSE(mode_util::isAdvanced(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(Px4CustomModeTest, UsesPositionIntentionForArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_ALTCTL,
			  vehicle_status_s::NAVIGATION_STATE_POSCTL),
		  vehicle_status_s::NAVIGATION_STATE_POSCTL);
}

TEST(Px4CustomModeTest, UsesAmPositionIntentionForArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_DESCEND,
			  vehicle_status_s::NAVIGATION_STATE_AM_POSITION),
		  vehicle_status_s::NAVIGATION_STATE_AM_POSITION);
}

TEST(Px4CustomModeTest, UsesAmOffboardIntentionForArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_DESCEND,
			  vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD),
		  vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);
}

TEST(Px4CustomModeTest, KeepsCurrentNavStateForOtherArmingChecks)
{
	EXPECT_EQ(Commander::getNavStateForArmingCheck(
			  vehicle_status_s::NAVIGATION_STATE_ALTCTL,
			  vehicle_status_s::NAVIGATION_STATE_MANUAL),
		  vehicle_status_s::NAVIGATION_STATE_ALTCTL);
}
