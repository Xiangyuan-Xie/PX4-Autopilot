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

#include <gtest/gtest.h>

#include "ControlAllocationFullyActuated.hpp"

using matrix::Matrix;
using matrix::Vector;

namespace
{

ControlAllocationFullyActuated makeTarotAllocator(int actuator_count = 6, bool remove_yaw_authority = false)
{
	ControlAllocationFullyActuated allocator;
	Matrix<float, 6, 16> effectiveness;
	effectiveness.setZero();
	const float tarot_effectiveness[6][6] {
		{-3.804684456f, -6.659014665f, -2.860308939f,  2.871289968f,  6.646942577f,  3.769568522f},
		{ 4.485667563f, -0.511253609f, -7.037704046f, -7.024337302f, -0.484687541f,  4.499028938f},
		{ 3.740112772f, -4.243556892f,  4.743499997f, -4.739986635f,  4.236587106f, -3.736656339f},
		{-5.368979259f, 10.737945954f, -5.368979259f, -5.368979259f, 10.737945954f, -5.368979259f},
		{ 9.299286834f,  0.f,         -9.299286834f,  9.299286834f,  0.f,         -9.299286834f},
		{-18.598788428f, -18.598766668f, -18.598788428f, -18.598788428f, -18.598766668f, -18.598788428f},
	};

	for (int row = 0; row < 6; ++row) {
		for (int column = 0; column < actuator_count; ++column) {
			effectiveness(row, column) = remove_yaw_authority && row == ControlAllocation::YAW ? 0.f :
						     tarot_effectiveness[row][column % 6];
		}
	}

	ControlAllocation::ActuatorVector minimum;
	ControlAllocation::ActuatorVector maximum;
	ControlAllocation::ActuatorVector trim;
	ControlAllocation::ActuatorVector linearization_point;
	minimum.setAll(0.f);
	maximum.setAll(1.f);
	trim.setZero();
	linearization_point.setZero();
	allocator.setActuatorMin(minimum);
	allocator.setActuatorMax(maximum);
	allocator.setNormalizeRPY(true);
	allocator.setEffectivenessMatrix(effectiveness, trim, linearization_point, actuator_count, true);
	return allocator;
}

} // namespace

TEST(ControlAllocationFullyActuated, AllocatesFeasibleSixDimensionalWrench)
{
	auto allocator = makeTarotAllocator();
	const float setpoint_data[6] {0.05f, -0.04f, 0.03f, 0.02f, -0.02f, -0.4f};
	Vector<float, 6> setpoint{setpoint_data};
	allocator.setControlSetpoint(setpoint);
	allocator.allocate();

	const Vector<float, 6> allocated = allocator.getAllocatedControl();

	for (int axis = 0; axis < 6; ++axis) {
		EXPECT_NEAR(allocated(axis), setpoint(axis), 1e-5f);
	}

	for (int actuator = 0; actuator < 6; ++actuator) {
		EXPECT_GE(allocator.getActuatorSetpoint()(actuator), 0.f);
		EXPECT_LE(allocator.getActuatorSetpoint()(actuator), 1.f);
	}
}

TEST(ControlAllocationFullyActuated, ReportsTarotGeometryAndHoverMargin)
{
	auto allocator = makeTarotAllocator();
	const auto metrics = allocator.effectivenessMetrics(0.5f, 0.05f);

	EXPECT_EQ(metrics.rank, 6);
	EXPECT_NEAR(metrics.condition_number, 4.4028f, 1e-3f);
	EXPECT_TRUE(metrics.hover_feasible);
	EXPECT_NEAR(metrics.hover_residual, 0.f, 1e-5f);
	EXPECT_NEAR(metrics.hover_actuator_min, 0.36764f, 1e-4f);
	EXPECT_NEAR(metrics.hover_actuator_max, 0.63236f, 1e-4f);
}

TEST(ControlAllocationFullyActuated, SupportsFullRankSixToEightActuatorMatrices)
{
	for (const int actuator_count : {6, 8}) {
		auto allocator = makeTarotAllocator(actuator_count);
		const auto metrics = allocator.effectivenessMetrics(0.5f, 0.05f);

		EXPECT_EQ(metrics.rank, 6) << "actuator_count=" << actuator_count;
		EXPECT_TRUE(metrics.hover_feasible) << "actuator_count=" << actuator_count;
	}
}

TEST(ControlAllocationFullyActuated, RejectsRankDeficientEffectiveness)
{
	auto allocator = makeTarotAllocator(6, true);
	const auto metrics = allocator.effectivenessMetrics(0.5f, 0.05f);

	EXPECT_EQ(metrics.rank, 5);
	EXPECT_FALSE(metrics.hover_feasible);
	EXPECT_FALSE(PX4_ISFINITE(metrics.condition_number));
}

TEST(ControlAllocationFullyActuated, PreservesPriorityWhenSaturated)
{
	auto allocator = makeTarotAllocator();
	const float setpoint_data[6] {0.15f, 0.15f, 0.8f, 0.8f, 0.8f, -0.5f};
	Vector<float, 6> setpoint{setpoint_data};
	allocator.setControlSetpoint(setpoint);
	allocator.allocate();

	const Vector<float, 6> allocated = allocator.getAllocatedControl();
	EXPECT_NEAR(allocated(0), setpoint(0), 1e-5f);
	EXPECT_NEAR(allocated(1), setpoint(1), 1e-5f);
	EXPECT_NEAR(allocated(5), setpoint(5), 1e-5f);
	EXPECT_GT(fabsf(allocated(2)), 0.1f);
	EXPECT_LT(fabsf(allocated(3)), 1e-4f);
	EXPECT_LT(fabsf(allocated(4)), 1e-4f);

	for (int actuator = 0; actuator < 6; ++actuator) {
		EXPECT_GE(allocator.getActuatorSetpoint()(actuator), 0.f);
		EXPECT_LE(allocator.getActuatorSetpoint()(actuator), 1.f);
	}
}
