#include <gtest/gtest.h>

#include "mavlink_main.h"
#include "streams/AVAILABLE_MODES.hpp"

namespace
{

vehicle_status_s baseVehicleStatus()
{
	vehicle_status_s vehicle_status{};
	vehicle_status.vehicle_type = vehicle_status_s::VEHICLE_TYPE_ROTARY_WING;
	vehicle_status.is_vtol = false;
	vehicle_status.valid_nav_states_mask = mode_util::getValidNavStates();
	vehicle_status.can_set_nav_states_mask = vehicle_status.valid_nav_states_mask;
	return vehicle_status;
}

} // namespace

TEST(AvailableModesTest, AmOffboardAdvertisesNamedNonStandardMode)
{
	const vehicle_status_s vehicle_status = baseVehicleStatus();

	const mavlink_available_modes_t mode = available_modes_stream::buildAvailableModeMessage(vehicle_status, 1, 1,
			vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD, nullptr);
	const px4_custom_mode custom_mode = get_px4_custom_mode(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_EQ(mode.standard_mode, MAV_STANDARD_MODE_NON_STANDARD);
	EXPECT_STREQ(mode.mode_name, "AM Offboard");
	EXPECT_EQ(mode.custom_mode, custom_mode.data);
	EXPECT_EQ(mode.properties & MAV_MODE_PROPERTY_ADVANCED, 0u);
	EXPECT_EQ(mode.properties & MAV_MODE_PROPERTY_NOT_USER_SELECTABLE, 0u);
}

TEST(AvailableModesTest, AmOffboardRemainsNamedWhenNotSelectable)
{
	vehicle_status_s vehicle_status = baseVehicleStatus();
	vehicle_status.can_set_nav_states_mask &= ~(1u << vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	const mavlink_available_modes_t mode = available_modes_stream::buildAvailableModeMessage(vehicle_status, 1, 1,
			vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD, nullptr);

	EXPECT_STREQ(mode.mode_name, "AM Offboard");
	EXPECT_NE(mode.properties & MAV_MODE_PROPERTY_NOT_USER_SELECTABLE, 0u);
}

TEST(AvailableModesTest, FirstVehicleStatusUpdateRequestsMonitorRefreshOnlyOnce)
{
	vehicle_status_s vehicle_status{};
	bool received_vehicle_status = false;
	uint32_t last_valid_nav_states_mask = 0;
	uint32_t last_can_set_nav_states_mask = 0;

	EXPECT_TRUE(available_modes_stream::updateVehicleStatusMasks(received_vehicle_status, last_valid_nav_states_mask,
			last_can_set_nav_states_mask, vehicle_status));
	EXPECT_TRUE(received_vehicle_status);
	EXPECT_EQ(last_valid_nav_states_mask, vehicle_status.valid_nav_states_mask);
	EXPECT_EQ(last_can_set_nav_states_mask, vehicle_status.can_set_nav_states_mask);

	EXPECT_FALSE(available_modes_stream::updateVehicleStatusMasks(received_vehicle_status, last_valid_nav_states_mask,
			last_can_set_nav_states_mask, vehicle_status));

	vehicle_status.can_set_nav_states_mask = 1u << vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;
	EXPECT_TRUE(available_modes_stream::updateVehicleStatusMasks(received_vehicle_status, last_valid_nav_states_mask,
			last_can_set_nav_states_mask, vehicle_status));
	EXPECT_EQ(last_can_set_nav_states_mask, vehicle_status.can_set_nav_states_mask);
}
