#include "amPoseControlCheck.hpp"

namespace
{

uint32_t modeBit(uint8_t nav_state)
{
	return 1u << nav_state;
}

NavModes modeGroup(uint8_t nav_state)
{
	return static_cast<NavModes>(modeBit(nav_state));
}

void reportAmPoseUnavailable(const am_pose_control_status_s &status, bool status_recent, bool module_running,
			     Report &reporter)
{
	const NavModes mode = modeGroup(vehicle_status_s::NAVIGATION_STATE_AM_POSE);

	if (!status_recent) {
		/* EVENT
		 * @description
		 * Start the AM Pose controller and wait for a recent status update.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_am_pose_status_stale"), events::Log::Error,
					    "AM Pose status missing or stale");
		return;
	}

	if (!module_running) {
		/* EVENT
		 * @description
		 * Start the AM Pose controller module.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_am_pose_module_stopped"), events::Log::Error,
					    "AM Pose controller not running");
		return;
	}

	if (!status.manual_control_available) {
		/* EVENT
		 * @description
		 * Connect and enable manual control input.
		 */
		reporter.armingCheckFailure(mode, health_component_t::remote_control,
					    events::ID("check_am_pose_manual_control"), events::Log::Error,
					    "AM Pose manual control unavailable");
	}

	if (!status.vehicle_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid local position and velocity data.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_am_pose_vehicle_state"), events::Log::Error,
					    "AM Pose vehicle state invalid");
	}

	if (!status.attitude_valid) {
		/* EVENT
		 * @description
		 * Wait for a valid attitude estimate.
		 */
		reporter.armingCheckFailure(mode, health_component_t::attitude_estimate,
					    events::ID("check_am_pose_attitude"), events::Log::Error,
					    "AM Pose attitude invalid");
	}

	if (!status.angular_velocity_valid) {
		/* EVENT
		 * @description
		 * Wait for valid filtered angular velocity data.
		 */
		reporter.armingCheckFailure(mode, health_component_t::system,
					    events::ID("check_am_pose_angular_velocity"), events::Log::Error,
					    "AM Pose angular velocity invalid");
	}

	if (!status.arm_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid manipulator joint state data.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_am_pose_arm_state"), events::Log::Error,
					    "AM Pose arm state invalid");
	}

	if (!status.handover_action_ready) {
		/* EVENT
		 * @description
		 * Wait for a recent motor output from the standard control allocator before switching in flight.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_am_pose_handover"), events::Log::Error,
					    "AM Pose allocator handover unavailable");
	}

	if (status.manual_control_available && status.vehicle_state_valid && status.attitude_valid
	    && status.angular_velocity_valid && status.arm_state_valid && status.handover_action_ready
	    && !status.am_pose_available) {
		/* EVENT
		 * @description
		 * Install a compatible AM Pose policy blob and restart the controller.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_am_pose_mode_unavailable"), events::Log::Error,
					    "AM Pose policy unavailable");
	}
}

void reportOffboardAmPoseUnavailable(const am_pose_control_status_s &status, bool status_recent, bool module_running,
				     Report &reporter)
{
	const NavModes mode = modeGroup(vehicle_status_s::NAVIGATION_STATE_OFFBOARD);

	if (!status_recent) {
		/* EVENT
		 * @description
		 * Start the AM Pose controller and wait for a recent status update.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_status_stale"), events::Log::Error,
					    "Offboard AM Pose status missing or stale");
		return;
	}

	if (!module_running) {
		/* EVENT
		 * @description
		 * Start the AM Pose controller module.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_module_stopped"), events::Log::Error,
					    "Offboard AM Pose controller not running");
		return;
	}

	if (!status.vehicle_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid local position and velocity data.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_vehicle_state"), events::Log::Error,
					    "Offboard AM Pose vehicle state invalid");
	}

	if (!status.attitude_valid) {
		/* EVENT
		 * @description
		 * Wait for a valid attitude estimate.
		 */
		reporter.armingCheckFailure(mode, health_component_t::attitude_estimate,
					    events::ID("check_offboard_am_pose_attitude"), events::Log::Error,
					    "Offboard AM Pose attitude invalid");
	}

	if (!status.angular_velocity_valid) {
		/* EVENT
		 * @description
		 * Wait for valid filtered angular velocity data.
		 */
		reporter.armingCheckFailure(mode, health_component_t::system,
					    events::ID("check_offboard_am_pose_angular_velocity"), events::Log::Error,
					    "Offboard AM Pose angular velocity invalid");
	}

	if (!status.arm_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid manipulator joint state data.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_arm_state"), events::Log::Error,
					    "Offboard AM Pose arm state invalid");
	}

	if (!status.handover_action_ready) {
		/* EVENT
		 * @description
		 * Wait for a recent motor output from the standard control allocator before switching in flight.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_handover"), events::Log::Error,
					    "Offboard AM Pose allocator handover unavailable");
	}

	if (status.vehicle_state_valid && status.attitude_valid && status.angular_velocity_valid
		    && status.arm_state_valid && status.handover_action_ready && !status.offboard_am_pose_available) {
		/* EVENT
		 * @description
		 * Install a compatible AM Pose policy and restart the controller.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_unavailable"), events::Log::Error,
					    "Offboard AM Pose controller unavailable");
	}

	if (status.offboard_selector_mismatch) {
		/* EVENT
		 * @description
		 * Keep publishing the controller type selected when Offboard was entered, or exit Offboard before changing it.
		 */
		reporter.armingCheckFailure(mode, health_component_t::position_controller,
					    events::ID("check_offboard_am_pose_selector_changed"), events::Log::Error,
					    "Offboard controller changed; exit and re-enter Offboard");
	}
}

} // namespace

void AmPoseControlChecks::checkAndReport(const Context &context, Report &reporter)
{
	const hrt_abstime now = hrt_absolute_time();
	am_pose_control_status_s status{};
	const bool status_received = _am_pose_control_status_sub.copy(&status);
	const bool status_recent = status_received && now < status.timestamp + kStatusTimeout;
	const bool module_running = status_recent && status.module_running;
	const bool policy_state_valid = status.vehicle_state_valid && status.attitude_valid
					&& status.angular_velocity_valid && status.arm_state_valid;

	const uint8_t am_pose_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_POSE;
	const bool am_pose_available = module_running && policy_state_valid
				       && status.manual_control_available && status.am_pose_available;

	if (am_pose_available) {
		reporter.failsafeFlags().mode_req_other &= ~modeBit(am_pose_nav_state);

	} else {
		reporter.clearArmingBits(modeGroup(am_pose_nav_state));
		reporter.failsafeFlags().mode_req_other |= modeBit(am_pose_nav_state);

		if (context.status().nav_state_user_intention == am_pose_nav_state) {
			reportAmPoseUnavailable(status, status_recent, module_running, reporter);
		}
	}

	offboard_control_mode_s offboard_control_mode{};
	const bool offboard_control_mode_received = _offboard_control_mode_sub.copy(&offboard_control_mode);
	const bool am_pose_requested = offboard_control_mode_received
				       && offboard_control_mode.controller_type
				       == offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;
	const uint8_t offboard_nav_state = vehicle_status_s::NAVIGATION_STATE_OFFBOARD;
	const bool offboard_am_pose_available = module_running && policy_state_valid
						&& status.offboard_am_pose_available;

	if (!am_pose_requested || offboard_am_pose_available) {
		reporter.failsafeFlags().mode_req_other &= ~modeBit(offboard_nav_state);

	} else {
		reporter.clearArmingBits(modeGroup(offboard_nav_state));
		reporter.failsafeFlags().mode_req_other |= modeBit(offboard_nav_state);

		if (context.status().nav_state_user_intention == offboard_nav_state) {
			reportOffboardAmPoseUnavailable(status, status_recent, module_running, reporter);
		}
	}
}
