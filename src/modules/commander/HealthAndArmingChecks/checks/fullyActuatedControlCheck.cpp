/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
 *
 ****************************************************************************/

#include "fullyActuatedControlCheck.hpp"

#include <px4_platform_common/events.h>

void FullyActuatedControlChecks::checkAndReport(const Context &context, Report &reporter)
{
	if (_param_ca_airframe.get() != CA_AIRFRAME_FULLY_ACTUATED_MULTIROTOR) {
		return;
	}

	fully_actuated_control_status_s status{};
	const hrt_abstime now = hrt_absolute_time();
	const bool received = _status_sub.copy(&status);
	const bool recent = received && now < status.timestamp + STATUS_TIMEOUT;
	const bool valid = recent && status.enabled && status.config_valid && status.matrix_full_rank
			   && status.hover_feasible;

	if (!valid) {
		/* EVENT
		 * @description
		 * Verify the fully actuated rotor count, axes, reversible mask and hover actuator margin.
		 */
		reporter.armingCheckFailure(NavModes::All, health_component_t::motors_escs,
					    events::ID("check_fully_actuated_allocation"), events::Log::Error,
					    "Fully actuated allocation unavailable");

		if (reporter.mavlink_log_pub()) {
			mavlink_log_critical(reporter.mavlink_log_pub(),
					     "Preflight Fail: Fully actuated allocation unavailable");
		}
	}
}
