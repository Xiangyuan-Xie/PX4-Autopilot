#include <gtest/gtest.h>

#define private public
#include "Common.hpp"
#undef private
#include "checks/amPosControlCheck.hpp"
#include "checks/modeCheck.hpp"

#include <uORB/Publication.hpp>
#include <uORB/topics/am_pos_control_status.h>

#include <cstring>

namespace
{

Report runChecks(const am_pos_control_status_s &am_status, uint8_t nav_state, bool offboard_signal_lost = false)
{
	uORB::Publication<am_pos_control_status_s> am_status_pub{ORB_ID(am_pos_control_status)};
	am_status_pub.publish(am_status);

	vehicle_status_s vehicle_status{};
	vehicle_status.vehicle_type = vehicle_status_s::VEHICLE_TYPE_ROTARY_WING;
	vehicle_status.nav_state_user_intention = nav_state;

	Context context{vehicle_status};
	failsafe_flags_s failsafe_flags{};
	Report reporter{failsafe_flags, 0_s};
	reporter.prepare(vehicle_status.vehicle_type);
	reporter.failsafeFlags().offboard_control_signal_lost = offboard_signal_lost;

	AmPosControlChecks am_checks;
	ModeChecks mode_checks;
	am_checks.checkAndReport(context, reporter);
	mode_checks.checkAndReport(context, reporter);
	reporter.finalize();

	return reporter;
}

am_pos_control_status_s baseStatus()
{
	am_pos_control_status_s status{};
	status.timestamp = hrt_absolute_time();
	status.module_running = true;
	status.manual_control_available = true;
	status.vehicle_state_valid = true;
	status.attitude_valid = true;
	status.angular_velocity_valid = true;
	status.arm_state_valid = true;
	status.offboard_control_mode_supported = true;
	status.offboard_control_mode_fresh = true;
	status.offboard_control_mode_valid = true;
	status.trajectory_setpoint_valid = true;
	status.am_position_available = true;
	status.am_offboard_available = true;
	return status;
}

bool hasEventId(const Report &report, uint32_t expected_event_id)
{
	int offset = 0;

	while (offset < report._next_buffer_idx) {
		const auto *header = reinterpret_cast<const Report::EventBufferHeader *>(report._event_buffer + offset);
		uint32_t event_id = 0;
		std::memcpy(&event_id, &header->id, sizeof(event_id));

		if (event_id == expected_event_id) {
			return true;
		}

		offset += sizeof(Report::EventBufferHeader) + header->size;
	}

	return false;
}

} // namespace

TEST(AmPosControlCheckTest, ModuleOfflineBlocksBothModes)
{
	am_pos_control_status_s status = baseStatus();
	status.module_running = false;

	Report am_position_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSITION);
	Report am_offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(am_position_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
	EXPECT_FALSE(am_offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(am_position_report, events::ID("check_am_pos_module_stopped")));
	EXPECT_TRUE(hasEventId(am_offboard_report, events::ID("check_am_offboard_module_stopped")));
}

TEST(AmPosControlCheckTest, StaleStatusBlocksAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.timestamp = 1;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(reporter, events::ID("check_am_offboard_status_stale")));
}

TEST(AmPosControlCheckTest, ManualReadyAllowsAmPosition)
{
	const am_pos_control_status_s status = baseStatus();
	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSITION);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
}

TEST(AmPosControlCheckTest, InvalidVehicleStateBlocksBothModes)
{
	am_pos_control_status_s status = baseStatus();
	status.vehicle_state_valid = false;

	Report am_position_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSITION);
	Report am_offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(am_position_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
	EXPECT_FALSE(am_offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(am_position_report, events::ID("check_am_pos_vehicle_state")));
	EXPECT_TRUE(hasEventId(am_offboard_report, events::ID("check_am_offboard_vehicle_state")));
}

TEST(AmPosControlCheckTest, InvalidAttitudeBlocksBothModes)
{
	am_pos_control_status_s status = baseStatus();
	status.attitude_valid = false;

	Report am_position_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSITION);
	Report am_offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(am_position_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
	EXPECT_FALSE(am_offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(am_position_report, events::ID("check_am_pos_attitude")));
	EXPECT_TRUE(hasEventId(am_offboard_report, events::ID("check_am_offboard_attitude")));
}

TEST(AmPosControlCheckTest, InvalidAngularVelocityBlocksBothModes)
{
	am_pos_control_status_s status = baseStatus();
	status.angular_velocity_valid = false;

	Report am_position_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSITION);
	Report am_offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(am_position_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
	EXPECT_FALSE(am_offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(am_position_report, events::ID("check_am_pos_angular_velocity")));
	EXPECT_TRUE(hasEventId(am_offboard_report, events::ID("check_am_offboard_angular_velocity")));
}

TEST(AmPosControlCheckTest, AmSpecificOffboardReadinessBlocksAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.am_offboard_available = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(reporter, events::ID("check_am_offboard_mode_unavailable")));
}

TEST(AmPosControlCheckTest, OffboardSignalLossDoesNotBlockAmOffboard)
{
	const am_pos_control_status_s status = baseStatus();
	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD, true);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(AmPosControlCheckTest, StaleAmOffboardControlModeDoesNotBlockAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.offboard_control_mode_fresh = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(AmPosControlCheckTest, UnsupportedAmOffboardControlModeDoesNotBlockAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.offboard_control_mode_supported = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(AmPosControlCheckTest, MissingTrajectorySetpointDoesNotBlockAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.trajectory_setpoint_valid = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(AmPosControlCheckTest, YawOnlyTrajectorySetpointDoesNotBlockAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.trajectory_setpoint_valid = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(AmPosControlCheckTest, InvalidArmStateBlocksAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.arm_state_valid = false;
	status.am_offboard_available = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
	EXPECT_TRUE(hasEventId(reporter, events::ID("check_am_offboard_arm_state")));
}
