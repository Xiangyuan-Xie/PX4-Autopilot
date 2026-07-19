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

#pragma once

#include "../Common.hpp"

#include <control_allocation/actuator_effectiveness/ActuatorEffectiveness.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/topics/fully_actuated_control_status.h>

class FullyActuatedControlChecks : public HealthAndArmingCheckBase
{
public:
	FullyActuatedControlChecks() = default;
	~FullyActuatedControlChecks() = default;

	void checkAndReport(const Context &context, Report &reporter) override;

private:
	static constexpr hrt_abstime STATUS_TIMEOUT = 1_s;
	uORB::Subscription _status_sub{ORB_ID(fully_actuated_control_status)};

	DEFINE_PARAMETERS_CUSTOM_PARENT(HealthAndArmingCheckBase,
					(ParamInt<px4::params::CA_AIRFRAME>) _param_ca_airframe
				       )
};
