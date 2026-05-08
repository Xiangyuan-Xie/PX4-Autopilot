#include <gtest/gtest.h>

#include "am_pos_control.hpp"

namespace
{
vehicle_local_position_s validLocalPosition()
{
	vehicle_local_position_s position{};
	position.timestamp = 1000;
	position.xy_valid = true;
	position.z_valid = true;
	position.v_xy_valid = true;
	position.v_z_valid = true;
	position.x = 1.0f;
	position.y = -2.0f;
	position.z = -3.0f;
	position.vx = 0.1f;
	position.vy = -0.2f;
	position.vz = 0.3f;
	position.heading = 0.4f;
	return position;
}
}

TEST(AmPosControlTest, FillAmTestResultFromActionMapsMotorControls)
{
	const hrt_abstime now = 123456;
	const hrt_abstime sample = 123000;
	const hrt_abstime setpoint = 120000;
	const RlToolsAdapter::Action action{0.f, 1.f, -1.f, 2.f};

	am_test_result_s result{};
	AmPosControl::fillAmTestResultFromAction(result, now, sample, setpoint, action);

	EXPECT_EQ(result.timestamp, now);
	EXPECT_EQ(result.timestamp_sample, sample);
	EXPECT_EQ(result.am_setpoint_timestamp, setpoint);
	EXPECT_TRUE(result.am_valid);
	EXPECT_EQ(result.failure_flags, am_test_result_s::FAILURE_NONE);
	EXPECT_EQ(result.degraded_flags, 0u);

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(result.am_raw_action[i], action[i]);
		EXPECT_GE(result.am_mapped_action[i], 0.f);
		EXPECT_LE(result.am_mapped_action[i], 1.f);
		EXPECT_FLOAT_EQ(result.am_motor_control[i], result.am_mapped_action[i]);
	}

	EXPECT_FLOAT_EQ(result.am_mapped_action[0], 0.5f);

	for (int i = 4; i < 12; ++i) {
		EXPECT_TRUE(std::isnan(result.am_motor_control[i]));
	}
}

TEST(AmPosControlTest, FillInvalidAmTestResultUsesNanControlsAndFailureFlag)
{
	am_test_result_s result{};
	AmPosControl::fillInvalidAmTestResult(result, 10, 20, 30, am_test_result_s::FAILURE_AM_SETPOINT_INVALID);

	EXPECT_EQ(result.timestamp, 10);
	EXPECT_EQ(result.timestamp_sample, 20);
	EXPECT_EQ(result.am_setpoint_timestamp, 30);
	EXPECT_FALSE(result.am_valid);
	EXPECT_EQ(result.failure_flags, am_test_result_s::FAILURE_AM_SETPOINT_INVALID);
	EXPECT_EQ(result.degraded_flags, 0u);

	for (float value : result.am_raw_action) {
		EXPECT_TRUE(std::isnan(value));
	}

	for (float value : result.am_mapped_action) {
		EXPECT_TRUE(std::isnan(value));
	}

	for (float value : result.am_motor_control) {
		EXPECT_TRUE(std::isnan(value));
	}
}

TEST(AmPosControlTest, AmTestLocalPositionAllowsInvalidFiniteXyForDegradedLogging)
{
	vehicle_local_position_s position{};
	position.timestamp = 1;
	position.x = 1.f;
	position.y = -2.f;
	position.z = 3.f;
	position.vx = 0.1f;
	position.vy = -0.2f;
	position.vz = 0.3f;
	position.heading = 0.4f;
	position.xy_valid = false;
	position.z_valid = true;
	position.v_xy_valid = false;
	position.v_z_valid = true;

	uint32_t degraded_flags = am_test_result_s::DEGRADED_NONE;
	EXPECT_TRUE(AmPosControl::vehicleStateValidForAmTest(position, true, true, 1, degraded_flags));
	EXPECT_EQ(degraded_flags, am_test_result_s::DEGRADED_LOCAL_XY_INVALID
		  | am_test_result_s::DEGRADED_LOCAL_VXY_INVALID);
}

TEST(AmPosControlTest, StrictLocalPositionRejectsInvalidXy)
{
	vehicle_local_position_s position{};
	position.timestamp = 1;
	position.x = 1.f;
	position.y = -2.f;
	position.z = 3.f;
	position.vx = 0.1f;
	position.vy = -0.2f;
	position.vz = 0.3f;
	position.heading = 0.4f;
	position.xy_valid = false;
	position.z_valid = true;
	position.v_xy_valid = false;
	position.v_z_valid = true;

	EXPECT_FALSE(AmPosControl::vehicleStateValidStrict(position, true, true, 1));
}

TEST(AmPosControlTest, StrictVehicleStateRejectsInvalidLocalXyFlags)
{
	vehicle_local_position_s position = validLocalPosition();
	position.xy_valid = false;
	position.v_xy_valid = false;

	EXPECT_FALSE(AmPosControl::vehicleStateValidStrict(position, true, true, 1500));
}

TEST(AmPosControlTest, AmTestVehicleStateAllowsFiniteInvalidLocalXyFlagsAsDegraded)
{
	vehicle_local_position_s position = validLocalPosition();
	position.xy_valid = false;
	position.v_xy_valid = false;

	uint32_t degraded_flags = 0;

	EXPECT_TRUE(AmPosControl::vehicleStateValidForAmTest(position, true, true, 1500, degraded_flags));
	EXPECT_EQ(degraded_flags, am_test_result_s::DEGRADED_LOCAL_XY_INVALID
		  | am_test_result_s::DEGRADED_LOCAL_VXY_INVALID);
}

TEST(AmPosControlTest, AmTestVehicleStateRejectsNonFiniteInvalidLocalXyValues)
{
	vehicle_local_position_s position = validLocalPosition();
	position.xy_valid = false;
	position.x = NAN;

	uint32_t degraded_flags = 0;

	EXPECT_FALSE(AmPosControl::vehicleStateValidForAmTest(position, true, true, 1500, degraded_flags));
}

TEST(AmPosControlTest, FillDefaultAmTestSetpointUsesCurrentPoseAndMarksDegraded)
{
	const vehicle_local_position_s position = validLocalPosition();

	trajectory_setpoint_s setpoint{};
	uint32_t degraded_flags = 0;

	AmPosControl::fillDefaultAmTestSetpoint(setpoint, 2000, position, degraded_flags);

	EXPECT_EQ(setpoint.timestamp, 2000);
	EXPECT_FLOAT_EQ(setpoint.position[0], position.x);
	EXPECT_FLOAT_EQ(setpoint.position[1], position.y);
	EXPECT_FLOAT_EQ(setpoint.position[2], position.z);
	EXPECT_FLOAT_EQ(setpoint.velocity[0], 0.0f);
	EXPECT_FLOAT_EQ(setpoint.velocity[1], 0.0f);
	EXPECT_FLOAT_EQ(setpoint.velocity[2], 0.0f);
	EXPECT_FLOAT_EQ(setpoint.yaw, position.heading);
	EXPECT_FLOAT_EQ(setpoint.yawspeed, 0.0f);
	EXPECT_EQ(degraded_flags, am_test_result_s::DEGRADED_SETPOINT_DEFAULTED);
}
