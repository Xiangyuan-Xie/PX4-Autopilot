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

} // namespace

void AmPosControlChecks::checkAndReport(const Context &context, Report &reporter)
{
	const hrt_abstime now = hrt_absolute_time();
	am_pos_control_status_s status{};

	const bool status_received = _am_pos_control_status_sub.copy(&status);
	const bool status_recent = status_received && now < status.timestamp + kStatusTimeout;
	const bool module_running = status_recent && status.module_running;

	const bool am_position_available = module_running && status.am_position_available;
	const bool am_offboard_available = module_running && status.am_offboard_available && status.arm_state_valid
					   && status.offboard_control_mode_fresh && status.offboard_control_mode_supported
					   && status.trajectory_setpoint_valid;

	const uint8_t am_position_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_POSITION;
	const uint8_t am_offboard_nav_state = vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;

	if (am_position_available) {
		reporter.failsafeFlags().mode_req_other &= ~modeBit(am_position_nav_state);

	} else {
		reporter.clearArmingBits(modeGroup(am_position_nav_state));
		reporter.failsafeFlags().mode_req_other |= modeBit(am_position_nav_state);

		if (context.status().nav_state_user_intention == am_position_nav_state) {
			/* EVENT
			 * @description
			 * Start the AM position controller and ensure manual control input plus arm state are valid.
			 */
			reporter.armingCheckFailure(modeGroup(am_position_nav_state),
						    health_component_t::position_controller,
						    events::ID("check_am_pos_mode_unavailable"),
						    events::Log::Error, "AM Position unavailable");
		}
	}

	if (am_offboard_available) {
		reporter.failsafeFlags().mode_req_other &= ~modeBit(am_offboard_nav_state);

	} else {
		reporter.clearArmingBits(modeGroup(am_offboard_nav_state));
		reporter.failsafeFlags().mode_req_other |= modeBit(am_offboard_nav_state);

		if (context.status().nav_state_user_intention == am_offboard_nav_state) {
			/* EVENT
			 * @description
			 * Start the AM position controller and ensure the supported AM offboard inputs are available.
			 */
			reporter.armingCheckFailure(modeGroup(am_offboard_nav_state),
						    health_component_t::position_controller,
						    events::ID("check_am_offboard_mode_unavailable"),
						    events::Log::Error, "AM Offboard unavailable");
		}
	}
}
