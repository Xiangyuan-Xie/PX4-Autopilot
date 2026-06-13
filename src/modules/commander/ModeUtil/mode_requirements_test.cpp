#include <gtest/gtest.h>

#include "mode_requirements.hpp"
#include <lib/modes/ui.hpp>
#include <uORB/topics/vehicle_status.h>

TEST(ModeRequirementsTest, AddsAmPositionRequirementsForDedicatedInternalNavState)
{
	failsafe_flags_s flags{};

	mode_util::getModeRequirements(vehicle_status_s::VEHICLE_TYPE_ROTARY_WING, flags);

	const uint32_t am_position_mask = 1u << vehicle_status_s::NAVIGATION_STATE_AM_POSITION;

	EXPECT_NE(flags.mode_req_angular_velocity & am_position_mask, 0u);
	EXPECT_NE(flags.mode_req_attitude & am_position_mask, 0u);
	EXPECT_NE(flags.mode_req_local_position & am_position_mask, 0u);
	EXPECT_NE(flags.mode_req_local_alt & am_position_mask, 0u);
	EXPECT_NE(flags.mode_req_manual_control & am_position_mask, 0u);
	EXPECT_NE(flags.mode_req_other & am_position_mask, 0u);
}

TEST(ModeRequirementsTest, AddsOnlyAmPosControlRequirementForDedicatedAmOffboardNavState)
{
	failsafe_flags_s flags{};

	mode_util::getModeRequirements(vehicle_status_s::VEHICLE_TYPE_ROTARY_WING, flags);

	const uint32_t am_offboard_mask = 1u << vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;

	EXPECT_EQ(flags.mode_req_angular_velocity & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_attitude & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_local_position & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_local_position_relaxed & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_global_position & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_global_position_relaxed & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_local_alt & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_offboard_signal & am_offboard_mask, 0u);
	EXPECT_EQ(flags.mode_req_manual_control & am_offboard_mask, 0u);
	EXPECT_NE(flags.mode_req_other & am_offboard_mask, 0u);
}

TEST(ModeRequirementsTest, ReservedNavStateSlot11IsNotSelectable)
{
	EXPECT_EQ(vehicle_status_s::NAVIGATION_STATE_FREE5, 11);
	EXPECT_EQ(mode_util::getValidNavStates() & (1u << vehicle_status_s::NAVIGATION_STATE_FREE5), 0u);
}

TEST(ModeRequirementsTest, ReservedNavStateSlot11IsAdvancedByDefault)
{
	EXPECT_TRUE(mode_util::isAdvanced(vehicle_status_s::NAVIGATION_STATE_FREE5));
}
