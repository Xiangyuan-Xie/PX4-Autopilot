#include <gtest/gtest.h>

#define DEFINE_GET_PX4_CUSTOM_MODE
#include "px4_custom_mode.h"

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
