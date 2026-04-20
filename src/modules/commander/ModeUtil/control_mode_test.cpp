#include <gtest/gtest.h>

#include "control_mode.hpp"

TEST(ControlModeTest, DetectsManualAmPositionControlMode)
{
	vehicle_control_mode_s control_mode{};
	control_mode.flag_control_manual_enabled = true;
	control_mode.flag_control_position_enabled = true;
	control_mode.flag_control_velocity_enabled = true;
	control_mode.flag_control_altitude_enabled = true;
	control_mode.flag_control_climb_rate_enabled = true;
	control_mode.flag_control_termination_enabled = true;

	EXPECT_TRUE(mode_util::isAmPositionControlMode(control_mode));
}

TEST(ControlModeTest, DetectsOffboardAmPositionControlMode)
{
	vehicle_control_mode_s control_mode{};
	control_mode.flag_control_offboard_enabled = true;
	control_mode.flag_control_position_enabled = true;
	control_mode.flag_control_velocity_enabled = true;
	control_mode.flag_control_altitude_enabled = true;
	control_mode.flag_control_climb_rate_enabled = true;
	control_mode.flag_control_termination_enabled = true;

	EXPECT_FALSE(mode_util::isAmPositionControlMode(control_mode));
	EXPECT_TRUE(mode_util::isAnyAmPositionControlMode(control_mode));
}

TEST(ControlModeTest, RejectsNativeOffboardPositionControlMode)
{
	vehicle_control_mode_s control_mode{};
	control_mode.flag_control_offboard_enabled = true;
	control_mode.flag_control_position_enabled = true;
	control_mode.flag_control_velocity_enabled = true;
	control_mode.flag_control_altitude_enabled = true;
	control_mode.flag_control_climb_rate_enabled = true;
	control_mode.flag_control_acceleration_enabled = true;
	control_mode.flag_control_attitude_enabled = true;
	control_mode.flag_control_rates_enabled = true;
	control_mode.flag_control_allocation_enabled = true;

	EXPECT_FALSE(mode_util::isAmPositionControlMode(control_mode));
	EXPECT_FALSE(mode_util::isAnyAmPositionControlMode(control_mode));
}
