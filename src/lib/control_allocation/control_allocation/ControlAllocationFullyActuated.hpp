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

#include "ControlAllocationPseudoInverse.hpp"

class ControlAllocationFullyActuated : public ControlAllocationPseudoInverse
{
public:
	struct EffectivenessMetrics {
		uint8_t rank{0};
		float condition_number{NAN};
		bool hover_feasible{false};
		float hover_residual{NAN};
		float hover_actuator_min{NAN};
		float hover_actuator_max{NAN};
	};

	ControlAllocationFullyActuated() = default;
	~ControlAllocationFullyActuated() override = default;

	void allocate() override;
	void setEffectivenessMatrix(const matrix::Matrix<float, NUM_AXES, NUM_ACTUATORS> &effectiveness,
				    const ActuatorVector &actuator_trim, const ActuatorVector &linearization_point,
				    int num_actuators, bool update_normalization_scale) override;
	EffectivenessMetrics effectivenessMetrics(float hover_thrust, float actuator_margin) const;

private:
	void addControlIncrement(ActuatorVector &actuator_sp, const ActuatorVector &increment) const;
	float feasibleScale(const ActuatorVector &actuator_sp, const ActuatorVector &increment) const;
};
