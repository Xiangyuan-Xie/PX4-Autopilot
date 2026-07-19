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

#include "ControlAllocationFullyActuated.hpp"

#include <mathlib/mathlib.h>

#include <float.h>
#include <string.h>

void ControlAllocationFullyActuated::setEffectivenessMatrix(
	const matrix::Matrix<float, NUM_AXES, NUM_ACTUATORS> &effectiveness,
	const ActuatorVector &actuator_trim, const ActuatorVector &linearization_point,
	int num_actuators, bool update_normalization_scale)
{
	ControlAllocationPseudoInverse::setEffectivenessMatrix(effectiveness, actuator_trim, linearization_point,
			num_actuators, update_normalization_scale);
	updatePseudoInverse();
}

float ControlAllocationFullyActuated::feasibleScale(const ActuatorVector &actuator_sp,
		const ActuatorVector &increment) const
{
	float scale = 1.f;

	for (int i = 0; i < _num_actuators; ++i) {
		if (increment(i) > FLT_EPSILON) {
			scale = math::min(scale, (_actuator_max(i) - actuator_sp(i)) / increment(i));

		} else if (increment(i) < -FLT_EPSILON) {
			scale = math::min(scale, (_actuator_min(i) - actuator_sp(i)) / increment(i));
		}
	}

	return math::constrain(scale, 0.f, 1.f);
}

void ControlAllocationFullyActuated::addControlIncrement(ActuatorVector &actuator_sp,
		const ActuatorVector &increment) const
{
	actuator_sp += increment * feasibleScale(actuator_sp, increment);
}

void ControlAllocationFullyActuated::allocate()
{
	updatePseudoInverse();
	_prev_actuator_sp = _actuator_sp;

	const matrix::Vector<float, NUM_AXES> control = _control_sp - _control_trim;
	_actuator_sp = _actuator_trim;

	// Preserve attitude and vertical authority first, then yaw and lateral force.
	addControlIncrement(_actuator_sp, _mix.col(ControlAxis::ROLL) * control(ControlAxis::ROLL)
			    + _mix.col(ControlAxis::PITCH) * control(ControlAxis::PITCH)
			    + _mix.col(ControlAxis::THRUST_Z) * control(ControlAxis::THRUST_Z));
	addControlIncrement(_actuator_sp, _mix.col(ControlAxis::YAW) * control(ControlAxis::YAW));
	addControlIncrement(_actuator_sp, _mix.col(ControlAxis::THRUST_X) * control(ControlAxis::THRUST_X)
			    + _mix.col(ControlAxis::THRUST_Y) * control(ControlAxis::THRUST_Y));
}

ControlAllocationFullyActuated::EffectivenessMetrics
ControlAllocationFullyActuated::effectivenessMetrics(float hover_thrust, float actuator_margin) const
{
	EffectivenessMetrics metrics{};
	float normalized[NUM_AXES][NUM_ACTUATORS] {};

	for (int row = 0; row < NUM_AXES; ++row) {
		float norm_squared = 0.f;

		for (int column = 0; column < _num_actuators; ++column) {
			norm_squared += _effectiveness(row, column) * _effectiveness(row, column);
		}

		const float norm = sqrtf(norm_squared);

		if (norm > FLT_EPSILON) {
			for (int column = 0; column < _num_actuators; ++column) {
				normalized[row][column] = _effectiveness(row, column) / norm;
			}
		}
	}

	float echelon[NUM_AXES][NUM_ACTUATORS];
	memcpy(echelon, normalized, sizeof(echelon));
	int pivot_row = 0;

	for (int column = 0; column < _num_actuators && pivot_row < NUM_AXES; ++column) {
		int best_row = pivot_row;

		for (int row = pivot_row + 1; row < NUM_AXES; ++row) {
			if (fabsf(echelon[row][column]) > fabsf(echelon[best_row][column])) {
				best_row = row;
			}
		}

		if (fabsf(echelon[best_row][column]) < 1e-3f) {
			continue;
		}

		for (int remaining_column = column; remaining_column < _num_actuators; ++remaining_column) {
			const float tmp = echelon[pivot_row][remaining_column];
			echelon[pivot_row][remaining_column] = echelon[best_row][remaining_column];
			echelon[best_row][remaining_column] = tmp;
		}

		const float pivot = echelon[pivot_row][column];

		for (int row = pivot_row + 1; row < NUM_AXES; ++row) {
			const float factor = echelon[row][column] / pivot;

			for (int remaining_column = column; remaining_column < _num_actuators; ++remaining_column) {
				echelon[row][remaining_column] -= factor * echelon[pivot_row][remaining_column];
			}
		}

		++pivot_row;
	}

	metrics.rank = pivot_row;

	// Jacobi eigenvalue iteration on B*B^T gives the 2-norm condition number.
	float gram[NUM_AXES][NUM_AXES] {};

	for (int row = 0; row < NUM_AXES; ++row) {
		for (int column = 0; column < NUM_AXES; ++column) {
			for (int actuator = 0; actuator < _num_actuators; ++actuator) {
				gram[row][column] += _effectiveness(row, actuator) * _effectiveness(column, actuator);
			}
		}
	}

	for (int sweep = 0; sweep < 64; ++sweep) {
		int p = 0;
		int q = 1;
		float largest = fabsf(gram[p][q]);

		for (int row = 0; row < NUM_AXES; ++row) {
			for (int column = row + 1; column < NUM_AXES; ++column) {
				if (fabsf(gram[row][column]) > largest) {
					largest = fabsf(gram[row][column]);
					p = row;
					q = column;
				}
			}
		}

		if (largest < 1e-7f) {
			break;
		}

		const float angle = 0.5f * atan2f(2.f * gram[p][q], gram[q][q] - gram[p][p]);
		const float cosine = cosf(angle);
		const float sine = sinf(angle);
		const float app = gram[p][p];
		const float aqq = gram[q][q];
		const float apq = gram[p][q];
		gram[p][p] = cosine * cosine * app - 2.f * sine * cosine * apq + sine * sine * aqq;
		gram[q][q] = sine * sine * app + 2.f * sine * cosine * apq + cosine * cosine * aqq;
		gram[p][q] = 0.f;
		gram[q][p] = 0.f;

		for (int index = 0; index < NUM_AXES; ++index) {
			if (index != p && index != q) {
				const float aip = gram[index][p];
				const float aiq = gram[index][q];
				gram[index][p] = gram[p][index] = cosine * aip - sine * aiq;
				gram[index][q] = gram[q][index] = sine * aip + cosine * aiq;
			}
		}
	}

	float minimum_eigenvalue = FLT_MAX;
	float maximum_eigenvalue = 0.f;

	for (int index = 0; index < NUM_AXES; ++index) {
		minimum_eigenvalue = math::min(minimum_eigenvalue, gram[index][index]);
		maximum_eigenvalue = math::max(maximum_eigenvalue, gram[index][index]);
	}

	if (metrics.rank == NUM_AXES && minimum_eigenvalue > FLT_EPSILON) {
		metrics.condition_number = sqrtf(maximum_eigenvalue / minimum_eigenvalue);
	}

	matrix::Vector<float, NUM_AXES> hover_control;
	hover_control.setZero();
	hover_control(ControlAxis::THRUST_Z) = -math::constrain(hover_thrust, 0.f, 1.f);
	const ActuatorVector hover_actuator = _actuator_trim + _mix * (hover_control - _control_trim);
	metrics.hover_actuator_min = FLT_MAX;
	metrics.hover_actuator_max = -FLT_MAX;
	bool inside_margin = true;

	for (int actuator = 0; actuator < _num_actuators; ++actuator) {
		metrics.hover_actuator_min = math::min(metrics.hover_actuator_min, hover_actuator(actuator));
		metrics.hover_actuator_max = math::max(metrics.hover_actuator_max, hover_actuator(actuator));
		inside_margin = inside_margin
				&& hover_actuator(actuator) >= _actuator_min(actuator) + actuator_margin
				&& hover_actuator(actuator) <= _actuator_max(actuator) - actuator_margin;
	}

	const matrix::Vector<float, NUM_AXES> allocated =
		(_effectiveness * (hover_actuator - _actuator_trim)).emult(_control_allocation_scale);
	metrics.hover_residual = 0.f;

	for (int axis = 0; axis < NUM_AXES; ++axis) {
		metrics.hover_residual = math::max(metrics.hover_residual, fabsf(hover_control(axis) - allocated(axis)));
	}

	metrics.hover_feasible = metrics.rank == NUM_AXES && inside_margin && metrics.hover_residual < 1e-3f;
	return metrics;
}
