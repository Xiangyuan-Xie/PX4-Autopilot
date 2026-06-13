#include "amPosControlCheck.hpp"

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

void reportAmPositionUnavailable(const am_pos_control_status_s &status, bool status_recent, bool module_running,
				  Report &reporter)
{
	const uint8_t am_position_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_POSITION;
	const NavModes am_position_mode = modeGroup(am_position_nav_state);

	if (!status_recent) {
		/* EVENT
		 * @description
		 * Start the AM position controller and wait for a recent status update.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_pos_status_stale"),
					    events::Log::Error, "AM Position status missing or stale");
		return;
	}

	if (!module_running) {
		/* EVENT
		 * @description
		 * Start the AM position controller module.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_pos_module_stopped"),
					    events::Log::Error, "AM Position controller not running");
		return;
	}

	if (!status.manual_control_available) {
		/* EVENT
		 * @description
		 * Connect and enable manual control input.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::remote_control,
					    events::ID("check_am_pos_manual_control"),
					    events::Log::Error, "AM Position manual control unavailable");
	}

	if (!status.vehicle_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid local position, velocity, heading, attitude, and angular velocity data.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_pos_vehicle_state"),
					    events::Log::Error, "AM Position vehicle state invalid");
	}

	if (!status.attitude_valid) {
		/* EVENT
		 * @description
		 * Wait for a valid attitude estimate.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::attitude_estimate,
					    events::ID("check_am_pos_attitude"),
					    events::Log::Error, "AM Position attitude invalid");
	}

	if (!status.angular_velocity_valid) {
		/* EVENT
		 * @description
		 * Wait for valid filtered angular velocity data.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::system,
					    events::ID("check_am_pos_angular_velocity"),
					    events::Log::Error, "AM Position angular velocity invalid");
	}

	if (!status.arm_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid manipulator joint state data.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_pos_arm_state"),
					    events::Log::Error, "AM Position arm state invalid");
	}

	if (status.manual_control_available && status.vehicle_state_valid && status.attitude_valid
	    && status.angular_velocity_valid && status.arm_state_valid && !status.am_position_available) {
		/* EVENT
		 * @description
		 * Start the AM position controller and ensure manual control input, vehicle state, and arm state are valid.
		 */
		reporter.armingCheckFailure(am_position_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_pos_mode_unavailable"),
					    events::Log::Error, "AM Position unavailable");
	}
}

void reportAmOffboardUnavailable(const am_pos_control_status_s &status, bool status_recent, bool module_running,
				 Report &reporter)
{
	const uint8_t am_offboard_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;
	const NavModes am_offboard_mode = modeGroup(am_offboard_nav_state);

	if (!status_recent) {
		/* EVENT
		 * @description
		 * Start the AM position controller and wait for a recent status update.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_offboard_status_stale"),
					    events::Log::Error, "AM Offboard status missing or stale");
		return;
	}

	if (!module_running) {
		/* EVENT
		 * @description
		 * Start the AM position controller module.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_offboard_module_stopped"),
					    events::Log::Error, "AM Offboard controller not running");
		return;
	}

	if (!status.vehicle_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid local position, velocity, heading, attitude, and angular velocity data.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_offboard_vehicle_state"),
					    events::Log::Error, "AM Offboard vehicle state invalid");
	}

	if (!status.attitude_valid) {
		/* EVENT
		 * @description
		 * Wait for a valid attitude estimate.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::attitude_estimate,
					    events::ID("check_am_offboard_attitude"),
					    events::Log::Error, "AM Offboard attitude invalid");
	}

	if (!status.angular_velocity_valid) {
		/* EVENT
		 * @description
		 * Wait for valid filtered angular velocity data.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::system,
					    events::ID("check_am_offboard_angular_velocity"),
					    events::Log::Error, "AM Offboard angular velocity invalid");
	}

	if (!status.arm_state_valid) {
		/* EVENT
		 * @description
		 * Wait for valid manipulator joint state data.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_offboard_arm_state"),
					    events::Log::Error, "AM Offboard arm state invalid");
	}

	if (status.vehicle_state_valid && status.attitude_valid && status.angular_velocity_valid && status.arm_state_valid
	    && !status.am_offboard_available) {
		/* EVENT
		 * @description
		 * Start the AM position controller and ensure the vehicle state and supported AM offboard inputs are available.
		 */
		reporter.armingCheckFailure(am_offboard_mode,
					    health_component_t::position_controller,
					    events::ID("check_am_offboard_mode_unavailable"),
					    events::Log::Error, "AM Offboard unavailable");
	}
}

} // namespace

void AmPosControlChecks::checkAndReport(const Context &context, Report &reporter)
{
	const hrt_abstime now = hrt_absolute_time();
	am_pos_control_status_s status{};

	const bool status_received = _am_pos_control_status_sub.copy(&status);
	const bool status_recent = status_received && now < status.timestamp + kStatusTimeout;
	const bool module_running = status_recent && status.module_running;
	const bool policy_state_valid = status.vehicle_state_valid && status.attitude_valid
					&& status.angular_velocity_valid;

	const bool am_position_available = module_running && status.am_position_available && policy_state_valid;
	const bool am_offboard_available = module_running && status.am_offboard_available && status.arm_state_valid
					   && policy_state_valid;

	const uint8_t am_position_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_POSITION;
	const uint8_t am_offboard_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;

	if (am_position_available) {
		reporter.failsafeFlags().mode_req_other &= ~modeBit(am_position_nav_state);

	} else {
		reporter.clearArmingBits(modeGroup(am_position_nav_state));
		reporter.failsafeFlags().mode_req_other |= modeBit(am_position_nav_state);

		if (context.status().nav_state_user_intention == am_position_nav_state) {
			reportAmPositionUnavailable(status, status_recent, module_running, reporter);
		}
	}

	if (am_offboard_available) {
		reporter.failsafeFlags().mode_req_other &= ~modeBit(am_offboard_nav_state);

	} else {
		reporter.clearArmingBits(modeGroup(am_offboard_nav_state));
		reporter.failsafeFlags().mode_req_other |= modeBit(am_offboard_nav_state);

		if (context.status().nav_state_user_intention == am_offboard_nav_state) {
			reportAmOffboardUnavailable(status, status_recent, module_running, reporter);
		}
	}
}
