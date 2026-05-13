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
	const RlToolsAdapter::Action action{0.f, 0.25f, 0.75f, 1.25f};

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

	EXPECT_FLOAT_EQ(result.am_mapped_action[0], 0.0f);
	EXPECT_FLOAT_EQ(result.am_mapped_action[1], 0.25f);
	EXPECT_FLOAT_EQ(result.am_mapped_action[2], 0.75f);
	EXPECT_FLOAT_EQ(result.am_mapped_action[3], 1.0f);

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

TEST(AmPosControlTest, FillThrustSetpointFromMotorsAveragesFourMotorCommands)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.control[0] = 0.8f;
	actuator_motors.control[1] = 0.6f;
	actuator_motors.control[2] = 0.4f;
	actuator_motors.control[3] = 0.2f;

	vehicle_thrust_setpoint_s thrust_setpoint{};
	AmPosControl::fillThrustSetpointFromMotors(thrust_setpoint, 2000, 1500, actuator_motors);

	EXPECT_EQ(thrust_setpoint.timestamp, 2000);
	EXPECT_EQ(thrust_setpoint.timestamp_sample, 1500);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[0], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[1], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[2], -0.5f);
}

TEST(AmPosControlTest, FillThrustSetpointFromMotorsIgnoresNanCommands)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.control[0] = 0.8f;
	actuator_motors.control[1] = NAN;
	actuator_motors.control[2] = 0.4f;
	actuator_motors.control[3] = NAN;

	vehicle_thrust_setpoint_s thrust_setpoint{};
	AmPosControl::fillThrustSetpointFromMotors(thrust_setpoint, 2000, 1500, actuator_motors);

	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[0], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[1], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[2], -0.6f);
}

TEST(AmPosControlTest, FillThrustSetpointFromMotorsPublishesZeroWhenAllCommandsAreNan)
{
	actuator_motors_s actuator_motors{};

	for (float &control : actuator_motors.control) {
		control = NAN;
	}

	vehicle_thrust_setpoint_s thrust_setpoint{};
	AmPosControl::fillThrustSetpointFromMotors(thrust_setpoint, 2000, 1500, actuator_motors);

	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[0], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[1], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[2], 0.0f);
}

TEST(AmPosControlTest, FillIdleMotorSetpointCommandsIdleForPolicyMotors)
{
	actuator_motors_s actuator_motors{};
	AmPosControl::fillIdleMotorSetpoint(actuator_motors, 2000, 1500);

	EXPECT_EQ(actuator_motors.timestamp, 2000);
	EXPECT_EQ(actuator_motors.timestamp_sample, 1500);
	EXPECT_EQ(actuator_motors.reversible_flags, 0u);

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(actuator_motors.control[i], 0.0f);
	}

	for (int i = 4; i < 12; ++i) {
		EXPECT_TRUE(std::isnan(actuator_motors.control[i]));
	}
}

TEST(AmPosControlTest, IdleMotorSetpointPublishesZeroThrustIntent)
{
	actuator_motors_s actuator_motors{};
	AmPosControl::fillIdleMotorSetpoint(actuator_motors, 2000, 1500);

	vehicle_thrust_setpoint_s thrust_setpoint{};
	AmPosControl::fillThrustSetpointFromMotors(thrust_setpoint, 2000, 1500, actuator_motors);

	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[0], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[1], 0.0f);
	EXPECT_FLOAT_EQ(thrust_setpoint.xyz[2], 0.0f);
}

TEST(AmPosControlTest, ManualTakeoffReleaseFollowsThrottleFromMinimumToCenter)
{
	const float deadzone = 0.1f;

	EXPECT_FLOAT_EQ(AmPosControl::manualTakeoffReleaseFromThrottle(-1.0f, deadzone), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::manualTakeoffReleaseFromThrottle(-0.9f, deadzone), 0.0f);
	EXPECT_NEAR(AmPosControl::manualTakeoffReleaseFromThrottle(-0.45f, deadzone), 0.5f, 1e-6f);
	EXPECT_FLOAT_EQ(AmPosControl::manualTakeoffReleaseFromThrottle(0.0f, deadzone), 1.0f);
	EXPECT_FLOAT_EQ(AmPosControl::manualTakeoffReleaseFromThrottle(0.5f, deadzone), 1.0f);
}

TEST(AmPosControlTest, ApplyMotorReleaseScalesPolicyMotorsAndPreservesNanChannels)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.control[0] = 0.8f;
	actuator_motors.control[1] = 0.6f;
	actuator_motors.control[2] = 0.4f;
	actuator_motors.control[3] = 0.2f;

	for (int i = 4; i < 12; ++i) {
		actuator_motors.control[i] = NAN;
	}

	AmPosControl::applyMotorRelease(actuator_motors, 0.5f);

	EXPECT_FLOAT_EQ(actuator_motors.control[0], 0.4f);
	EXPECT_FLOAT_EQ(actuator_motors.control[1], 0.3f);
	EXPECT_FLOAT_EQ(actuator_motors.control[2], 0.2f);
	EXPECT_FLOAT_EQ(actuator_motors.control[3], 0.1f);

	for (int i = 4; i < 12; ++i) {
		EXPECT_TRUE(std::isnan(actuator_motors.control[i]));
	}
}

TEST(AmPosControlTest, GatedPositionErrorKeepsFullBodyErrorWhenLinearCommandInactive)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Vector3f pos_error_w = root_quat.rotateVector(matrix::Vector3f{1.f, 2.f, 3.f});
	const bool active_w[3] {false, false, false};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, active_w);

	EXPECT_NEAR(gated_pos_error_b(0), 1.f, 1e-5f);
	EXPECT_NEAR(gated_pos_error_b(1), 2.f, 1e-5f);
	EXPECT_NEAR(gated_pos_error_b(2), 3.f, 1e-5f);
}

TEST(AmPosControlTest, GatedPositionErrorZeroesWorldAxesBeforeBodyProjection)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_w[3] {false, false, true};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, active_w);

	EXPECT_NEAR(gated_pos_error_b(0), 4.f * cosf(pitch), 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), -5.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 4.f * sinf(pitch), 1e-6f);
}

TEST(AmPosControlTest, GatedPositionErrorUsesWorldAxesWhenVehicleYawed)
{
	const float yaw = M_PI_F / 4.f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_w[3] {true, false, false};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, active_w);

	EXPECT_NEAR(gated_pos_error_b(0), -5.f * sinf(yaw), 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), -5.f * cosf(yaw), 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 6.f, 1e-6f);
}

TEST(AmPosControlTest, GatedPositionErrorPreservesInactiveHorizontalAxesDuringVerticalCommand)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_w[3] {false, false, true};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, active_w);

	EXPECT_NEAR(gated_pos_error_b(0), 4.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), -5.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 0.f, 1e-6f);
}

TEST(AmPosControlTest, GatedPositionErrorPreservesInactiveAltitudeAxisDuringHorizontalCommand)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_w[3] {true, true, false};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, active_w);

	EXPECT_NEAR(gated_pos_error_b(0), 0.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), 0.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 6.f, 1e-6f);
}

TEST(AmPosControlTest, LinearVelocityErrorDampsInactiveWorldAxesWhenVehicleYawed)
{
	const float yaw = M_PI_F / 2.f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Vector3f desired_vel_w{0.5f, 0.f, 0.f};
	const matrix::Vector3f actual_vel_w{0.5f, 3.f, 0.f};
	const bool active_w[3] {true, false, false};
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(matrix::Vector3f{0.f, -3.f, 0.f});

	const matrix::Vector3f vel_error_b =
		AmPosControl::linearVelocityErrorForPolicy(desired_vel_w, actual_vel_w, root_quat, active_w);

	EXPECT_NEAR(vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, LinearVelocityErrorDampsZeroCommandVelocity)
{
	const float yaw = M_PI_F / 2.f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Vector3f desired_vel_w{0.f, 0.f, 0.f};
	const matrix::Vector3f actual_vel_w{0.4f, -0.2f, 0.1f};
	const bool active_w[3] {false, false, false};
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(matrix::Vector3f{-0.4f, 0.2f, -0.1f});

	const matrix::Vector3f vel_error_b =
		AmPosControl::linearVelocityErrorForPolicy(desired_vel_w, actual_vel_w, root_quat, active_w);

	EXPECT_NEAR(vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, AngularVelocityErrorTracksYawAndDampsRollPitchBeforeBodyProjection)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Vector3f desired_ang_vel_w{0.f, 0.f, 0.4f};
	const matrix::Vector3f actual_ang_vel_w{2.f, -1.f, 0.1f};
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(matrix::Vector3f{-2.f, 1.f, 0.3f});

	const matrix::Vector3f ang_vel_error_b =
		AmPosControl::angularVelocityErrorForPolicy(desired_ang_vel_w, actual_ang_vel_w, root_quat, true);

	EXPECT_NEAR(ang_vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, AngularVelocityErrorDampsZeroCommandRate)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Vector3f desired_ang_vel_w{0.f, 0.f, 0.f};
	const matrix::Vector3f actual_ang_vel_w{0.2f, -0.3f, 0.4f};
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(matrix::Vector3f{-0.2f, 0.3f, -0.4f});

	const matrix::Vector3f ang_vel_error_b =
		AmPosControl::angularVelocityErrorForPolicy(desired_ang_vel_w, actual_ang_vel_w, root_quat, false);

	EXPECT_NEAR(ang_vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, GatedAttitudeErrorKeepsFullErrorWhenAngularCommandInactive)
{
	const matrix::Vector3f att_error_b{0.1f, -0.2f, 0.3f};

	const matrix::Vector3f gated_att_error_b =
		AmPosControl::gateAttitudeErrorForPolicy(att_error_b, false);

	EXPECT_NEAR(gated_att_error_b(0), 0.1f, 1e-6f);
	EXPECT_NEAR(gated_att_error_b(1), -0.2f, 1e-6f);
	EXPECT_NEAR(gated_att_error_b(2), 0.3f, 1e-6f);
}

TEST(AmPosControlTest, GatedAttitudeErrorZeroesOnlyYawWhenYawRateActive)
{
	const matrix::Vector3f att_error_b{0.1f, -0.2f, 0.3f};

	const matrix::Vector3f gated_att_error_b =
		AmPosControl::gateAttitudeErrorForPolicy(att_error_b, true);

	EXPECT_NEAR(gated_att_error_b(0), 0.1f, 1e-6f);
	EXPECT_NEAR(gated_att_error_b(1), -0.2f, 1e-6f);
	EXPECT_NEAR(gated_att_error_b(2), 0.f, 1e-6f);
}

TEST(AmPosControlTest, DesiredYawRateBodyIsPureBodyZWhenLevel)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f desired_ang_vel_b =
		AmPosControl::desiredYawRateBodyFromNed(root_quat, 0.4f);

	EXPECT_NEAR(desired_ang_vel_b(0), 0.f, 1e-6f);
	EXPECT_NEAR(desired_ang_vel_b(1), 0.f, 1e-6f);
	EXPECT_NEAR(desired_ang_vel_b(2), -0.4f, 1e-6f);
}

TEST(AmPosControlTest, DesiredYawRateBodyRotatesWorldYawRateIntoTiltedBodyFrame)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(matrix::Vector3f{0.f, 0.f, -0.4f});
	const matrix::Vector3f desired_ang_vel_b =
		AmPosControl::desiredYawRateBodyFromNed(root_quat, 0.4f);

	EXPECT_NEAR(desired_ang_vel_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(desired_ang_vel_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(desired_ang_vel_b(2), expected(2), 1e-6f);
	EXPECT_GT(fabsf(desired_ang_vel_b(0)), 0.1f);
}

TEST(AmPosControlTest, WorldYawRateOnlyActivatesYaw)
{
	EXPECT_TRUE(AmPosControl::commandActive(0.4f));
	EXPECT_FALSE(AmPosControl::commandActive(0.0f));
}

TEST(AmPosControlTest, InverseMappedMotorToActionClampsNormalizedCommands)
{
	EXPECT_TRUE(std::isfinite(AmPosControl::inverseMappedMotorToAction(0.0f)));
	EXPECT_TRUE(std::isfinite(AmPosControl::inverseMappedMotorToAction(1.0f)));
	EXPECT_FLOAT_EQ(AmPosControl::inverseMappedMotorToAction(-0.25f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::inverseMappedMotorToAction(0.5f), 0.5f);
	EXPECT_FLOAT_EQ(AmPosControl::inverseMappedMotorToAction(1.25f), 1.0f);
}

TEST(AmPosControlTest, FillActionHistoryFromMotorsUsesExecutedMotorCommands)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.control[0] = 0.4f;
	actuator_motors.control[1] = 0.3f;
	actuator_motors.control[2] = 0.2f;
	actuator_motors.control[3] = 0.1f;

	RlToolsAdapter::Action action{};
	AmPosControl::fillActionHistoryFromMotors(action, actuator_motors);

	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(action[i], actuator_motors.control[i], 1e-5f);
	}
}

TEST(AmPosControlTest, RampMotorOutputsForTakeoffScalesCollectiveAndDifferential)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.control[0] = 0.8f;
	actuator_motors.control[1] = 0.6f;
	actuator_motors.control[2] = 0.4f;
	actuator_motors.control[3] = 0.2f;

	for (int i = 4; i < 12; ++i) {
		actuator_motors.control[i] = NAN;
	}

	AmPosControl::rampMotorOutputsForTakeoff(actuator_motors, 0.5f);

	EXPECT_FLOAT_EQ(actuator_motors.control[0], 0.4f);
	EXPECT_FLOAT_EQ(actuator_motors.control[1], 0.3f);
	EXPECT_FLOAT_EQ(actuator_motors.control[2], 0.2f);
	EXPECT_FLOAT_EQ(actuator_motors.control[3], 0.1f);

	for (int i = 4; i < 12; ++i) {
		EXPECT_TRUE(std::isnan(actuator_motors.control[i]));
	}
}

TEST(AmPosControlTest, RampMotorOutputsForTakeoffZeroAndFullProgress)
{
	actuator_motors_s zero_progress{};
	actuator_motors_s full_progress{};

	for (int i = 0; i < 4; ++i) {
		zero_progress.control[i] = 0.7f - 0.1f * i;
		full_progress.control[i] = zero_progress.control[i];
	}

	for (int i = 4; i < 12; ++i) {
		zero_progress.control[i] = NAN;
		full_progress.control[i] = NAN;
	}

	AmPosControl::rampMotorOutputsForTakeoff(zero_progress, 0.0f);
	AmPosControl::rampMotorOutputsForTakeoff(full_progress, 1.0f);

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(zero_progress.control[i], 0.0f);
		EXPECT_FLOAT_EQ(full_progress.control[i], 0.7f - 0.1f * i);
	}
}

TEST(AmPosControlTest, AdvanceAmTakeoffRampProgressFollowsTakeoffState)
{
	EXPECT_FLOAT_EQ(AmPosControl::advanceAmTakeoffRampProgress(0.8f, 0.1f, 1.0f,
			takeoff_status_s::TAKEOFF_STATE_READY_FOR_TAKEOFF, false), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::advanceAmTakeoffRampProgress(0.2f, 0.1f, 1.0f,
			takeoff_status_s::TAKEOFF_STATE_RAMPUP, false), 0.3f);
	EXPECT_FLOAT_EQ(AmPosControl::advanceAmTakeoffRampProgress(0.95f, 0.1f, 1.0f,
			takeoff_status_s::TAKEOFF_STATE_RAMPUP, false), 1.0f);
	EXPECT_FLOAT_EQ(AmPosControl::advanceAmTakeoffRampProgress(0.2f, 0.1f, 1.0f,
			takeoff_status_s::TAKEOFF_STATE_FLIGHT, false), 1.0f);
	EXPECT_FLOAT_EQ(AmPosControl::advanceAmTakeoffRampProgress(0.2f, 0.1f, 1.0f,
			takeoff_status_s::TAKEOFF_STATE_RAMPUP, true), 1.0f);
}

TEST(AmPosControlTest, GroundedReadyStateRequiresOutputGate)
{
	EXPECT_TRUE(AmPosControl::takeoffStateRequiresOutputGate(takeoff_status_s::TAKEOFF_STATE_SPOOLUP, false));
	EXPECT_TRUE(AmPosControl::takeoffStateRequiresOutputGate(takeoff_status_s::TAKEOFF_STATE_READY_FOR_TAKEOFF, false));
}

TEST(AmPosControlTest, RampupAndFlightWithoutGroundContactAllowOutput)
{
	EXPECT_FALSE(AmPosControl::takeoffStateRequiresOutputGate(takeoff_status_s::TAKEOFF_STATE_RAMPUP, false));
	EXPECT_FALSE(AmPosControl::takeoffStateRequiresOutputGate(takeoff_status_s::TAKEOFF_STATE_FLIGHT, false));
}

TEST(AmPosControlTest, FlightWithGroundContactRequiresOutputGate)
{
	EXPECT_TRUE(AmPosControl::takeoffStateRequiresOutputGate(takeoff_status_s::TAKEOFF_STATE_FLIGHT, true));
}

TEST(AmPosControlTest, OffboardSetpointWantsTakeoffForUpwardPositionVelocityOrAcceleration)
{
	vehicle_local_position_s position = validLocalPosition();
	position.timestamp_sample = 1000;
	position.z = -3.0f;

	trajectory_setpoint_s setpoint{};
	setpoint.timestamp = 1000;
	setpoint.position[2] = -4.0f;
	EXPECT_TRUE(AmPosControl::offboardSetpointWantsTakeoff(setpoint, position, position.timestamp_sample));

	setpoint = {};
	setpoint.timestamp = 1000;
	setpoint.velocity[2] = -0.1f;
	EXPECT_TRUE(AmPosControl::offboardSetpointWantsTakeoff(setpoint, position, position.timestamp_sample));

	setpoint = {};
	setpoint.timestamp = 1000;
	setpoint.acceleration[2] = -0.1f;
	EXPECT_TRUE(AmPosControl::offboardSetpointWantsTakeoff(setpoint, position, position.timestamp_sample));
}

TEST(AmPosControlTest, OffboardSetpointDoesNotWantTakeoffForStaleOrNonUpwardSetpoint)
{
	vehicle_local_position_s position = validLocalPosition();
	position.timestamp_sample = 3_s;
	position.z = -3.0f;

	trajectory_setpoint_s setpoint{};
	setpoint.timestamp = 1000;
	setpoint.position[2] = -4.0f;
	EXPECT_FALSE(AmPosControl::offboardSetpointWantsTakeoff(setpoint, position, position.timestamp_sample));

	setpoint = {};
	setpoint.timestamp = 3_s;
	setpoint.position[2] = -2.0f;
	setpoint.velocity[2] = 0.1f;
	setpoint.acceleration[2] = 0.1f;
	EXPECT_FALSE(AmPosControl::offboardSetpointWantsTakeoff(setpoint, position, position.timestamp_sample));
}

TEST(AmPosControlTest, EnteringAmModeInAirSkipsTakeoffRamp)
{
	EXPECT_TRUE(AmPosControl::shouldSkipTakeoffRampOnAmModeEntry(true, false, false));
	EXPECT_FALSE(AmPosControl::shouldSkipTakeoffRampOnAmModeEntry(true, false, true));
	EXPECT_FALSE(AmPosControl::shouldSkipTakeoffRampOnAmModeEntry(true, true, false));
	EXPECT_FALSE(AmPosControl::shouldSkipTakeoffRampOnAmModeEntry(false, true, false));
}
