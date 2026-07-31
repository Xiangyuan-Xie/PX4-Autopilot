#include <gtest/gtest.h>

#define private public
#include "Commander.hpp"
#undef private

#include "ModeUtil/control_mode.hpp"

namespace
{

vehicle_status_s rotaryWingStatus(uint8_t nav_state)
{
	vehicle_status_s status{};
	status.vehicle_type = vehicle_status_s::VEHICLE_TYPE_ROTARY_WING;
	status.nav_state = nav_state;
	return status;
}

vehicle_control_mode_s controlModeFor(uint8_t nav_state, bool am_pose_offboard = false)
{
	vehicle_control_mode_s control_mode{};
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.position = true;
	offboard_control_mode.controller_type = am_pose_offboard
						? offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE
						: offboard_control_mode_s::CONTROLLER_TYPE_NATIVE;

	mode_util::getVehicleControlMode(nav_state, vehicle_status_s::VEHICLE_TYPE_ROTARY_WING, offboard_control_mode,
					 control_mode);

	return control_mode;
}

bool canDisarmInAir(uint8_t nav_state, arm_disarm_reason_t reason, bool com_disarm_man,
		    bool am_pose_offboard = false)
{
	return Commander::manualDisarmInAirAllowed(rotaryWingStatus(nav_state), controlModeFor(nav_state, am_pose_offboard), reason,
			com_disarm_man);
}

} // namespace

TEST(CommanderDisarmTest, AllowsRcDisarmInAirForAmPoseWhenManualDisarmEnabled)
{
	EXPECT_TRUE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_POSE, arm_disarm_reason_t::rc_switch, true));
}

TEST(CommanderDisarmTest, AllowsRcDisarmInAirForAmPoseOffboardWhenManualDisarmEnabled)
{
	EXPECT_TRUE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_OFFBOARD, arm_disarm_reason_t::rc_button, true, true));
}

TEST(CommanderDisarmTest, DeniesCommandDisarmInAirForAmPoseWithoutForce)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_POSE,
				    arm_disarm_reason_t::command_external, true));
}

TEST(CommanderDisarmTest, DeniesCommandDisarmInAirForAmPoseOffboardWithoutForce)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_OFFBOARD,
				    arm_disarm_reason_t::command_external, true, true));
}

TEST(CommanderDisarmTest, DeniesRcDisarmInAirForAmPoseWhenManualDisarmDisabled)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_POSE, arm_disarm_reason_t::rc_switch, false));
}

TEST(CommanderDisarmTest, DeniesRcDisarmInAirForAmPoseOffboardWhenManualDisarmDisabled)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_OFFBOARD, arm_disarm_reason_t::rc_button, false, true));
}

TEST(CommanderDisarmTest, DeniesRcDisarmInAirForPositionControlWhenManualDisarmEnabled)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_POSCTL, arm_disarm_reason_t::rc_switch, true));
}
