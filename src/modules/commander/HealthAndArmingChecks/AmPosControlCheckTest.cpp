#include <gtest/gtest.h>

#define private public
#include "Common.hpp"
#undef private
#include "checks/amPosControlCheck.hpp"
#include "checks/modeCheck.hpp"

#include <uORB/Publication.hpp>
#include <uORB/topics/am_pos_control_status.h>

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
	status.arm_state_valid = true;
	status.offboard_control_mode_supported = true;
	status.offboard_control_mode_fresh = true;
	status.trajectory_setpoint_valid = true;
	status.am_position_available = true;
	status.am_offboard_available = true;
	return status;
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
}

TEST(AmPosControlCheckTest, StaleStatusBlocksAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.timestamp = 1;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}

TEST(AmPosControlCheckTest, ManualReadyAllowsAmPosition)
{
	const am_pos_control_status_s status = baseStatus();
	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSITION);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
}

TEST(AmPosControlCheckTest, AmSpecificOffboardReadinessBlocksAmOffboard)
{
	am_pos_control_status_s status = baseStatus();
	status.am_offboard_available = false;

	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
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
}
