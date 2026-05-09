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

vehicle_control_mode_s controlModeFor(uint8_t nav_state)
{
	vehicle_control_mode_s control_mode{};
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.position = true;

	mode_util::getVehicleControlMode(nav_state, vehicle_status_s::VEHICLE_TYPE_ROTARY_WING, offboard_control_mode,
					 control_mode);

	return control_mode;
}

bool canDisarmInAir(uint8_t nav_state, arm_disarm_reason_t reason, bool com_disarm_man)
{
	return Commander::manualDisarmInAirAllowed(rotaryWingStatus(nav_state), controlModeFor(nav_state), reason,
			com_disarm_man);
}

} // namespace

TEST(CommanderDisarmTest, AllowsRcDisarmInAirForAmPositionWhenManualDisarmEnabled)
{
	EXPECT_TRUE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_POSITION, arm_disarm_reason_t::rc_switch, true));
}

TEST(CommanderDisarmTest, AllowsRcDisarmInAirForAmOffboardWhenManualDisarmEnabled)
{
	EXPECT_TRUE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD, arm_disarm_reason_t::rc_button, true));
}

TEST(CommanderDisarmTest, DeniesCommandDisarmInAirForAmPositionWithoutForce)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_POSITION, arm_disarm_reason_t::command_external, true));
}

TEST(CommanderDisarmTest, DeniesRcDisarmInAirForAmPositionWhenManualDisarmDisabled)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_POSITION, arm_disarm_reason_t::rc_switch, false));
}

TEST(CommanderDisarmTest, DeniesRcDisarmInAirForPositionControlWhenManualDisarmEnabled)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_POSCTL, arm_disarm_reason_t::rc_switch, true));
}

TEST(CommanderDisarmTest, DeniesRcDisarmInAirForAmTestWhenManualDisarmEnabled)
{
	EXPECT_FALSE(canDisarmInAir(vehicle_status_s::NAVIGATION_STATE_AM_TEST, arm_disarm_reason_t::rc_switch, true));
}
