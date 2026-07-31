#include <gtest/gtest.h>

#include "control_mode.hpp"
#include <uORB/topics/vehicle_status.h>

TEST(ControlModeTest, UsesManualControlShapeForAmPose)
{
	vehicle_control_mode_s control_mode{};
	offboard_control_mode_s offboard_control_mode{};

	mode_util::getVehicleControlMode(vehicle_status_s::NAVIGATION_STATE_AM_POSE,
					 vehicle_status_s::VEHICLE_TYPE_ROTARY_WING,
					 offboard_control_mode, control_mode);

	EXPECT_TRUE(control_mode.flag_control_manual_enabled);
	EXPECT_FALSE(control_mode.flag_control_offboard_enabled);
	EXPECT_TRUE(control_mode.flag_control_position_enabled);
	EXPECT_TRUE(control_mode.flag_control_velocity_enabled);
	EXPECT_TRUE(control_mode.flag_control_altitude_enabled);
	EXPECT_TRUE(control_mode.flag_control_climb_rate_enabled);
	EXPECT_TRUE(control_mode.flag_control_termination_enabled);
	EXPECT_FALSE(control_mode.flag_control_acceleration_enabled);
	EXPECT_FALSE(control_mode.flag_control_attitude_enabled);
	EXPECT_FALSE(control_mode.flag_control_rates_enabled);
	EXPECT_FALSE(control_mode.flag_control_allocation_enabled);
	EXPECT_TRUE(mode_util::isAmPoseControlMode(control_mode));
	EXPECT_TRUE(mode_util::isAnyAmPoseControlMode(control_mode));
}

TEST(ControlModeTest, UsesAmPoseControlShapeForStandardOffboard)
{
	vehicle_control_mode_s control_mode{};
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.position = true;
	offboard_control_mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;

	mode_util::getVehicleControlMode(vehicle_status_s::NAVIGATION_STATE_OFFBOARD,
					 vehicle_status_s::VEHICLE_TYPE_ROTARY_WING,
					 offboard_control_mode, control_mode);

	EXPECT_FALSE(control_mode.flag_control_manual_enabled);
	EXPECT_TRUE(control_mode.flag_control_offboard_enabled);
	EXPECT_TRUE(control_mode.flag_control_position_enabled);
	EXPECT_TRUE(control_mode.flag_control_velocity_enabled);
	EXPECT_TRUE(control_mode.flag_control_altitude_enabled);
	EXPECT_TRUE(control_mode.flag_control_climb_rate_enabled);
	EXPECT_TRUE(control_mode.flag_control_termination_enabled);
	EXPECT_FALSE(control_mode.flag_control_acceleration_enabled);
	EXPECT_FALSE(control_mode.flag_control_attitude_enabled);
	EXPECT_FALSE(control_mode.flag_control_rates_enabled);
	EXPECT_FALSE(control_mode.flag_control_allocation_enabled);
	EXPECT_FALSE(mode_util::isAmPoseControlMode(control_mode));
	EXPECT_TRUE(mode_util::isAmPoseOffboardControlMode(control_mode));
	EXPECT_TRUE(mode_util::isAnyAmPoseControlMode(control_mode));
}

TEST(ControlModeTest, KeepsNativeOffboardPositionControlShape)
{
	vehicle_control_mode_s control_mode{};
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.position = true;
	offboard_control_mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_NATIVE;

	mode_util::getVehicleControlMode(vehicle_status_s::NAVIGATION_STATE_OFFBOARD,
					 vehicle_status_s::VEHICLE_TYPE_ROTARY_WING,
					 offboard_control_mode, control_mode);

	EXPECT_TRUE(control_mode.flag_control_offboard_enabled);
	EXPECT_TRUE(control_mode.flag_control_acceleration_enabled);
	EXPECT_TRUE(control_mode.flag_control_attitude_enabled);
	EXPECT_TRUE(control_mode.flag_control_rates_enabled);
	EXPECT_TRUE(control_mode.flag_control_allocation_enabled);
	EXPECT_FALSE(mode_util::isAnyAmPoseControlMode(control_mode));
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

	EXPECT_FALSE(mode_util::isAmPoseControlMode(control_mode));
	EXPECT_FALSE(mode_util::isAnyAmPoseControlMode(control_mode));
}
