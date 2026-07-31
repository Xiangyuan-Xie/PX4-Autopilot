#include <gtest/gtest.h>

#define private public
#include "Common.hpp"
#undef private
#include "checks/amPoseControlCheck.hpp"
#include "checks/modeCheck.hpp"

#include <uORB/Publication.hpp>
#include <uORB/topics/am_pose_control_status.h>
#include <uORB/topics/offboard_control_mode.h>

#include <cstring>

namespace
{

Report runChecks(const am_pose_control_status_s &am_status, uint8_t nav_state,
		 uint8_t controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE,
		 bool offboard_signal_lost = false)
{
	uORB::Publication<am_pose_control_status_s> am_status_pub{ORB_ID(am_pose_control_status)};
	am_status_pub.publish(am_status);
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.timestamp = hrt_absolute_time();
	offboard_control_mode.position = true;
	offboard_control_mode.controller_type = controller_type;
	uORB::Publication<offboard_control_mode_s> offboard_control_mode_pub{ORB_ID(offboard_control_mode)};
	offboard_control_mode_pub.publish(offboard_control_mode);

	vehicle_status_s vehicle_status{};
	vehicle_status.vehicle_type = vehicle_status_s::VEHICLE_TYPE_ROTARY_WING;
	vehicle_status.nav_state = nav_state;
	vehicle_status.nav_state_user_intention = nav_state;

	Context context{vehicle_status};
	failsafe_flags_s failsafe_flags{};
	Report reporter{failsafe_flags, 0_s};
	reporter.prepare(vehicle_status.vehicle_type);
	reporter.failsafeFlags().offboard_control_signal_lost = offboard_signal_lost;

	AmPoseControlChecks am_checks;
	ModeChecks mode_checks;
	am_checks.checkAndReport(context, reporter);
	mode_checks.checkAndReport(context, reporter);
	reporter.finalize();
	return reporter;
}

am_pose_control_status_s baseStatus()
{
	am_pose_control_status_s status{};
	status.timestamp = hrt_absolute_time();
	status.module_running = true;
	status.manual_control_available = true;
	status.vehicle_state_valid = true;
	status.attitude_valid = true;
	status.angular_velocity_valid = true;
	status.arm_state_valid = true;
	status.trajectory_setpoint_valid = true;
	status.offboard_control_mode_fresh = true;
	status.offboard_control_mode_supported = true;
	status.offboard_control_mode_valid = true;
	status.am_pose_available = true;
	status.offboard_am_pose_available = true;
	status.policy_loaded = true;
	status.handover_action_ready = true;
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

TEST(AmPoseControlCheckTest, ModuleOfflineBlocksPoseAndOffboard)
{
	am_pose_control_status_s status = baseStatus();
	status.module_running = false;

	Report pose_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE);
	Report offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD);

	EXPECT_FALSE(pose_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
	EXPECT_FALSE(offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
	EXPECT_TRUE(hasEventId(pose_report, events::ID("check_am_pose_module_stopped")));
	EXPECT_TRUE(hasEventId(offboard_report, events::ID("check_offboard_am_pose_module_stopped")));
}

TEST(AmPoseControlCheckTest, ReadyInputsAllowPoseAndOffboard)
{
	const am_pose_control_status_s status = baseStatus();
	EXPECT_TRUE(runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE)
		    .canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
	EXPECT_TRUE(runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD)
		    .canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
}

TEST(AmPoseControlCheckTest, MissingPolicyBlocksPose)
{
	am_pose_control_status_s status = baseStatus();
	status.policy_loaded = false;
	status.am_pose_available = false;
	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
	EXPECT_TRUE(hasEventId(reporter, events::ID("check_am_pose_mode_unavailable")));
}

TEST(AmPoseControlCheckTest, MissingManualControlBlocksPoseOnly)
{
	am_pose_control_status_s status = baseStatus();
	status.manual_control_available = false;
	status.am_pose_available = false;

	Report pose_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE);
	Report offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD);

	EXPECT_FALSE(pose_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
	EXPECT_TRUE(hasEventId(pose_report, events::ID("check_am_pose_manual_control")));
	EXPECT_TRUE(offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
}

TEST(AmPoseControlCheckTest, MissingAllocatorHandoverBlocksPoseAndOffboard)
{
	am_pose_control_status_s status = baseStatus();
	status.handover_action_ready = false;
	status.am_pose_available = false;
	status.offboard_am_pose_available = false;

	Report pose_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE);
	Report offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD);

	EXPECT_FALSE(pose_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
	EXPECT_FALSE(offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
	EXPECT_TRUE(hasEventId(pose_report, events::ID("check_am_pose_handover")));
	EXPECT_TRUE(hasEventId(offboard_report, events::ID("check_offboard_am_pose_handover")));
}

TEST(AmPoseControlCheckTest, InvalidVehicleStateBlocksPoseAndOffboard)
{
	am_pose_control_status_s status = baseStatus();
	status.vehicle_state_valid = false;

	Report pose_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE);
	Report offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD);

	EXPECT_FALSE(pose_report.canRun(vehicle_status_s::NAVIGATION_STATE_AM_POSE));
	EXPECT_FALSE(offboard_report.canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
	EXPECT_TRUE(hasEventId(pose_report, events::ID("check_am_pose_vehicle_state")));
	EXPECT_TRUE(hasEventId(offboard_report, events::ID("check_offboard_am_pose_vehicle_state")));
}

TEST(AmPoseControlCheckTest, InvalidArmStateBlocksPoseAndOffboard)
{
	am_pose_control_status_s status = baseStatus();
	status.arm_state_valid = false;
	status.am_pose_available = false;
	status.offboard_am_pose_available = false;

	Report pose_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_AM_POSE);
	Report offboard_report = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD);

	EXPECT_TRUE(hasEventId(pose_report, events::ID("check_am_pose_arm_state")));
	EXPECT_TRUE(hasEventId(offboard_report, events::ID("check_offboard_am_pose_arm_state")));
}

TEST(AmPoseControlCheckTest, NativeOffboardDoesNotRequireAmPoseController)
{
	am_pose_control_status_s status = baseStatus();
	status.module_running = false;
	status.offboard_am_pose_available = false;
	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD,
				    offboard_control_mode_s::CONTROLLER_TYPE_NATIVE);

	EXPECT_TRUE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
}

TEST(AmPoseControlCheckTest, GenericOffboardSignalLossBlocksAmPoseOffboard)
{
	const am_pose_control_status_s status = baseStatus();
	Report reporter = runChecks(status, vehicle_status_s::NAVIGATION_STATE_OFFBOARD,
				    offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE, true);

	EXPECT_FALSE(reporter.canRun(vehicle_status_s::NAVIGATION_STATE_OFFBOARD));
}
