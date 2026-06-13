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

TEST(AmPosControlTest, FillPolicyObservationIncludesSourceTiming)
{
	RlToolsAdapter::Observation observation{};

	for (int i = 0; i < RlToolsAdapter::ObservationDim; ++i) {
		observation[i] = static_cast<float>(i) * 0.5f;
	}

	const RlToolsAdapter::Action action{0.1f, 0.2f, 0.3f, 0.4f};

	actuator_motors_s actuator_motors{};
	actuator_motors.timestamp = 9000;
	actuator_motors.control[0] = 0.11f;
	actuator_motors.control[1] = 0.22f;
	actuator_motors.control[2] = 0.33f;
	actuator_motors.control[3] = 0.44f;

	for (int i = 4; i < 12; ++i) {
		actuator_motors.control[i] = NAN;
	}

	AmPosControl::PolicyObservationTiming timing{};
	timing.observation_build_timestamp = 8000;
	timing.vehicle_local_position_timestamp = 7010;
	timing.vehicle_local_position_timestamp_sample = 7000;
	timing.vehicle_attitude_timestamp = 7110;
	timing.vehicle_attitude_timestamp_sample = 7100;
	timing.vehicle_angular_velocity_timestamp = 7210;
	timing.vehicle_angular_velocity_timestamp_sample = 7200;
	timing.arm_joint_state_timestamp = 7310;
	timing.arm_joint_state_timestamp_sample = 7300;
	timing.arm_joint_state_sequence = 77;
	timing.arm_joint_state_position_wrap_mask = 0b10101;
	timing.trajectory_setpoint_timestamp = 7400;
	timing.offboard_control_mode_timestamp = 7500;
	timing.policy_inference_start_timestamp = 8100;
	timing.policy_inference_finish_timestamp = 8125;
	timing.policy_sequence = 42;
	timing.takeoff_state = takeoff_status_s::TAKEOFF_STATE_RAMPUP;
	timing.takeoff_ramp_scale = 0.25f;
	timing.takeoff_ramped_speed_up = 0.25f;

	am_policy_observation_s policy_observation{};
	AmPosControl::fillPolicyObservation(policy_observation, observation, action, actuator_motors,
					    am_policy_observation_s::DEGRADED_SETPOINT_DEFAULTED, timing);

	EXPECT_EQ(policy_observation.timestamp, actuator_motors.timestamp);
	EXPECT_EQ(policy_observation.timestamp_sample, timing.vehicle_angular_velocity_timestamp_sample);
	EXPECT_EQ(policy_observation.degraded_flags, am_policy_observation_s::DEGRADED_SETPOINT_DEFAULTED);
	EXPECT_EQ(policy_observation.observation_build_timestamp, timing.observation_build_timestamp);
	EXPECT_EQ(policy_observation.vehicle_local_position_timestamp, timing.vehicle_local_position_timestamp);
	EXPECT_EQ(policy_observation.vehicle_local_position_timestamp_sample, timing.vehicle_local_position_timestamp_sample);
	EXPECT_EQ(policy_observation.vehicle_attitude_timestamp, timing.vehicle_attitude_timestamp);
	EXPECT_EQ(policy_observation.vehicle_attitude_timestamp_sample, timing.vehicle_attitude_timestamp_sample);
	EXPECT_EQ(policy_observation.vehicle_angular_velocity_timestamp, timing.vehicle_angular_velocity_timestamp);
	EXPECT_EQ(policy_observation.vehicle_angular_velocity_timestamp_sample, timing.vehicle_angular_velocity_timestamp_sample);
	EXPECT_EQ(policy_observation.arm_joint_state_timestamp, timing.arm_joint_state_timestamp);
	EXPECT_EQ(policy_observation.arm_joint_state_timestamp_sample, timing.arm_joint_state_timestamp_sample);
	EXPECT_EQ(policy_observation.arm_joint_state_sequence, timing.arm_joint_state_sequence);
	EXPECT_EQ(policy_observation.arm_joint_state_position_wrap_mask, timing.arm_joint_state_position_wrap_mask);
	EXPECT_EQ(policy_observation.trajectory_setpoint_timestamp, timing.trajectory_setpoint_timestamp);
	EXPECT_EQ(policy_observation.offboard_control_mode_timestamp, timing.offboard_control_mode_timestamp);
	EXPECT_EQ(policy_observation.policy_inference_start_timestamp, timing.policy_inference_start_timestamp);
	EXPECT_EQ(policy_observation.policy_inference_finish_timestamp, timing.policy_inference_finish_timestamp);
	EXPECT_EQ(policy_observation.policy_sequence, timing.policy_sequence);
	EXPECT_EQ(policy_observation.takeoff_state, timing.takeoff_state);
	EXPECT_FLOAT_EQ(policy_observation.takeoff_ramp_scale, timing.takeoff_ramp_scale);
	EXPECT_FLOAT_EQ(policy_observation.takeoff_ramped_speed_up, timing.takeoff_ramped_speed_up);

	for (int i = 0; i < RlToolsAdapter::ObservationDim; ++i) {
		EXPECT_FLOAT_EQ(policy_observation.observation[i], observation[i]);
	}

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(policy_observation.raw_action[i], action[i]);
		EXPECT_FLOAT_EQ(policy_observation.mapped_action[i], actuator_motors.control[i]);
	}
}

TEST(AmPosControlTest, WrapArmJointPositionKeepsPiBoundaries)
{
	uint32_t wrap_mask = 0;

	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(0.0f, wrap_mask, 0), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(M_PI_F, wrap_mask, 1), M_PI_F);
	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(-M_PI_F, wrap_mask, 2), -M_PI_F);
	EXPECT_EQ(wrap_mask, 0u);
}

TEST(AmPosControlTest, WrapArmJointPositionNormalizesFiniteOutOfRangeAngles)
{
	uint32_t wrap_mask = 0;

	EXPECT_NEAR(AmPosControl::wrapArmJointPositionForPolicy(M_PI_F + 0.2f, wrap_mask, 0),
		    -M_PI_F + 0.2f, 1e-6f);
	EXPECT_NEAR(AmPosControl::wrapArmJointPositionForPolicy(-M_PI_F - 0.3f, wrap_mask, 2),
		    M_PI_F - 0.3f, 1e-6f);
	const float wrapped_multi_turn = AmPosControl::wrapArmJointPositionForPolicy(5.0f * M_PI_F, wrap_mask, 3);
	EXPECT_LE(wrapped_multi_turn, M_PI_F);
	EXPECT_GE(wrapped_multi_turn, -M_PI_F);
	EXPECT_EQ(wrap_mask, (1u << 0) | (1u << 2) | (1u << 3));
}

TEST(AmPosControlTest, WrapArmJointPositionConvertsJoint5PhysicalAngleToOpening)
{
	uint32_t wrap_mask = 0;

	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(0.0f, wrap_mask, 4), 0.0f);
	EXPECT_NEAR(AmPosControl::wrapArmJointPositionForPolicy(-0.8615f, wrap_mask, 4), 0.5f, 1e-6f);
	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(-1.723f, wrap_mask, 4), 1.0f);
	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(1.0f, wrap_mask, 4), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::wrapArmJointPositionForPolicy(-2.0f, wrap_mask, 4), 1.0f);
	EXPECT_EQ(wrap_mask, 0u);
}

TEST(AmPosControlTest, WrapArmJointPositionHandlesLargeFiniteInput)
{
	uint32_t wrap_mask = 0;
	const float wrapped = AmPosControl::wrapArmJointPositionForPolicy(1.0e20f, wrap_mask, 3);

	EXPECT_TRUE(PX4_ISFINITE(wrapped));
	EXPECT_LE(wrapped, M_PI_F);
	EXPECT_GE(wrapped, -M_PI_F);
	EXPECT_EQ(wrap_mask, 1u << 3);
}

TEST(AmPosControlTest, WrapArmJointPositionDoesNotHideNonFiniteInput)
{
	uint32_t wrap_mask = 0;

	EXPECT_FALSE(PX4_ISFINITE(AmPosControl::wrapArmJointPositionForPolicy(NAN, wrap_mask, 0)));
	EXPECT_FALSE(PX4_ISFINITE(AmPosControl::wrapArmJointPositionForPolicy(INFINITY, wrap_mask, 1)));
	EXPECT_FALSE(PX4_ISFINITE(AmPosControl::wrapArmJointPositionForPolicy(-INFINITY, wrap_mask, 2)));
	EXPECT_EQ(wrap_mask, 0u);
}

TEST(AmPosControlTest, ArmJointStateCarriesSampleTimestampAndSequence)
{
	arm_joint_state_s arm_joint_state{};
	arm_joint_state.timestamp = 2000;
	arm_joint_state.timestamp_sample = 1900;
	arm_joint_state.sequence = 7;

	EXPECT_EQ(arm_joint_state.timestamp, 2000u);
	EXPECT_EQ(arm_joint_state.timestamp_sample, 1900u);
	EXPECT_EQ(arm_joint_state.sequence, 7u);
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

TEST(AmPosControlTest, FillAmOffboardHoldSetpointKeepsReferenceAndMarksDegraded)
{
	trajectory_setpoint_s setpoint{};
	uint32_t degraded_flags = 0;

	AmPosControl::fillAmOffboardHoldSetpoint(setpoint, 2000, degraded_flags);

	EXPECT_EQ(setpoint.timestamp, 2000);
	EXPECT_TRUE(std::isnan(setpoint.position[0]));
	EXPECT_TRUE(std::isnan(setpoint.position[1]));
	EXPECT_TRUE(std::isnan(setpoint.position[2]));
	EXPECT_FLOAT_EQ(setpoint.velocity[0], 0.0f);
	EXPECT_FLOAT_EQ(setpoint.velocity[1], 0.0f);
	EXPECT_FLOAT_EQ(setpoint.velocity[2], 0.0f);
	EXPECT_TRUE(std::isnan(setpoint.yaw));
	EXPECT_FLOAT_EQ(setpoint.yawspeed, 0.0f);
	EXPECT_EQ(degraded_flags, am_policy_observation_s::DEGRADED_SETPOINT_DEFAULTED);
}

TEST(AmPosControlTest, AmOffboardUsesExternalSetpointOnlyForFreshVelocityCommands)
{
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.velocity = true;

	trajectory_setpoint_s setpoint{};
	setpoint.timestamp = 1000;
	setpoint.velocity[0] = 1.0f;
	setpoint.velocity[1] = NAN;
	setpoint.velocity[2] = NAN;

	EXPECT_TRUE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, true, true));
	EXPECT_FALSE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, false, true));
	EXPECT_FALSE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, true, false));

	setpoint.velocity[0] = NAN;
	setpoint.velocity[1] = 2.0f;
	EXPECT_TRUE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, true, true));

	offboard_control_mode.velocity = false;
	offboard_control_mode.attitude = true;
	EXPECT_FALSE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, true, true));
}

TEST(AmPosControlTest, AmOffboardControlModeSupportsVelocityOnlyCommands)
{
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.velocity = true;

	EXPECT_TRUE(AmPosControl::offboardControlModeSupported(offboard_control_mode));

	offboard_control_mode.velocity = false;
	offboard_control_mode.position = true;
	EXPECT_FALSE(AmPosControl::offboardControlModeSupported(offboard_control_mode));

	offboard_control_mode.position = false;
	offboard_control_mode.velocity = true;
	offboard_control_mode.attitude = true;
	EXPECT_FALSE(AmPosControl::offboardControlModeSupported(offboard_control_mode));
}

TEST(AmPosControlTest, AmOffboardDoesNotTreatPositionOnlySetpointAsExternalCommand)
{
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.position = true;

	trajectory_setpoint_s setpoint{};
	setpoint.timestamp = 1000;
	setpoint.position[0] = 1.0f;
	setpoint.position[1] = NAN;
	setpoint.position[2] = NAN;
	setpoint.velocity[0] = NAN;
	setpoint.velocity[1] = NAN;
	setpoint.velocity[2] = NAN;

	EXPECT_FALSE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, true, true));
}

TEST(AmPosControlTest, AmOffboardDoesNotTreatYawOnlySetpointAsExternalCommand)
{
	offboard_control_mode_s offboard_control_mode{};
	offboard_control_mode.velocity = true;

	trajectory_setpoint_s setpoint{};
	setpoint.timestamp = 1000;
	setpoint.position[0] = NAN;
	setpoint.position[1] = NAN;
	setpoint.position[2] = NAN;
	setpoint.velocity[0] = NAN;
	setpoint.velocity[1] = NAN;
	setpoint.velocity[2] = NAN;
	setpoint.yaw = 1.0f;
	setpoint.yawspeed = 0.2f;

	EXPECT_FALSE(AmPosControl::amOffboardExternalSetpointUsable(offboard_control_mode, setpoint, true, true));
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

TEST(AmPosControlTest, ConstrainUpwardVelocityNedLimitsOnlyUpwardDemand)
{
	EXPECT_FLOAT_EQ(AmPosControl::constrainUpwardVelocityNed(-2.0f, 0.5f), -0.5f);
	EXPECT_FLOAT_EQ(AmPosControl::constrainUpwardVelocityNed(-0.25f, 0.5f), -0.25f);
	EXPECT_FLOAT_EQ(AmPosControl::constrainUpwardVelocityNed(0.4f, 0.5f), 0.4f);
	EXPECT_FLOAT_EQ(AmPosControl::constrainUpwardVelocityNed(-1.0f, 0.0f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::constrainUpwardVelocityNed(NAN, 0.5f), 0.0f);
}

TEST(AmPosControlTest, GatedPositionErrorKeepsFullBodyErrorWhenLinearCommandInactive)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f pos_error_w = root_quat.rotateVector(matrix::Vector3f{1.f, 2.f, 3.f});
	const bool active_h[3] {false, false, false};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(gated_pos_error_b(0), 1.f, 1e-5f);
	EXPECT_NEAR(gated_pos_error_b(1), 2.f, 1e-5f);
	EXPECT_NEAR(gated_pos_error_b(2), 3.f, 1e-5f);
}

TEST(AmPosControlTest, GatedPositionErrorZeroesHeadingAxesBeforeBodyProjection)
{
	const float pitch = 0.35f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, pitch, 0.f));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_h[3] {false, false, true};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(gated_pos_error_b(0), 4.f * cosf(pitch), 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), -5.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 4.f * sinf(pitch), 1e-6f);
}

TEST(AmPosControlTest, GatedPositionErrorUsesHeadingAxesWhenVehicleYawed)
{
	const float yaw = M_PI_F / 4.f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_h[3] {true, false, false};
	const matrix::Vector3f pos_error_h = heading_quat.inversed().rotateVector(pos_error_w);
	const matrix::Vector3f expected_error_w = heading_quat.rotateVector(matrix::Vector3f{0.f, pos_error_h(1), pos_error_h(2)});
	const matrix::Vector3f expected_error_b = root_quat.inversed().rotateVector(expected_error_w);

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(gated_pos_error_b(0), expected_error_b(0), 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), expected_error_b(1), 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), expected_error_b(2), 1e-6f);
	EXPECT_GT(fabsf(gated_pos_error_b(1)), 1.f);
}

TEST(AmPosControlTest, GatedPositionErrorPreservesCrossTrackDuringForwardHeadingCommand)
{
	const float yaw = M_PI_F / 4.f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Vector3f pos_error_w = heading_quat.rotateVector(matrix::Vector3f{3.f, -2.f, 1.f});
	const bool active_h[3] {true, false, false};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(gated_pos_error_b(0), 0.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), -2.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 1.f, 1e-6f);
}

TEST(AmPosControlTest, GatedPositionErrorPreservesInactiveHorizontalAxesDuringVerticalCommand)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_h[3] {false, false, true};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(gated_pos_error_b(0), 4.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), -5.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 0.f, 1e-6f);
}

TEST(AmPosControlTest, GatedPositionErrorPreservesInactiveAltitudeAxisDuringHorizontalCommand)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f pos_error_w{4.f, -5.f, 6.f};
	const bool active_h[3] {true, true, false};

	const matrix::Vector3f gated_pos_error_b =
		AmPosControl::gatePositionErrorForPolicy(pos_error_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(gated_pos_error_b(0), 0.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(1), 0.f, 1e-6f);
	EXPECT_NEAR(gated_pos_error_b(2), 6.f, 1e-6f);
}

TEST(AmPosControlTest, LinearVelocityErrorDampsInactiveHeadingAxesWhenVehicleYawed)
{
	const float yaw = M_PI_F / 2.f;
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, yaw));
	const matrix::Vector3f desired_vel_w = heading_quat.rotateVector(matrix::Vector3f{0.5f, 0.f, 0.f});
	const matrix::Vector3f actual_vel_w = heading_quat.rotateVector(matrix::Vector3f{0.5f, 3.f, 0.f});
	const bool active_h[3] {true, false, false};
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(heading_quat.rotateVector(matrix::Vector3f{0.f, -3.f, 0.f}));

	const matrix::Vector3f vel_error_b =
		AmPosControl::linearVelocityErrorForPolicy(desired_vel_w, actual_vel_w, root_quat, heading_quat, active_h);

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
	const matrix::Quatf heading_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const bool active_h[3] {false, false, false};
	const matrix::Vector3f expected = root_quat.inversed().rotateVector(matrix::Vector3f{-0.4f, 0.2f, -0.1f});

	const matrix::Vector3f vel_error_b =
		AmPosControl::linearVelocityErrorForPolicy(desired_vel_w, actual_vel_w, root_quat, heading_quat, active_h);

	EXPECT_NEAR(vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, AngularVelocityErrorDampsRollPitchAndTracksYawBeforeBodyProjection)
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

TEST(AmPosControlTest, AngularVelocityErrorDampsAllAxesWhenYawCommandInactive)
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

TEST(AmPosControlTest, AngularVelocityErrorIgnoresDesiredYawRateWhenYawInactive)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f desired_ang_vel_w{0.f, 0.f, 0.3f};
	const matrix::Vector3f actual_ang_vel_w{0.f, 0.f, 0.1f};
	const matrix::Vector3f expected{0.f, 0.f, -0.1f};

	const matrix::Vector3f ang_vel_error_b =
		AmPosControl::angularVelocityErrorForPolicy(desired_ang_vel_w, actual_ang_vel_w, root_quat, false);

	EXPECT_NEAR(ang_vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, AngularVelocityErrorKeepsRollPitchDampingDuringYawRateCommand)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Vector3f desired_ang_vel_w{0.f, 0.f, 0.3f};
	const matrix::Vector3f actual_ang_vel_w{0.4f, -0.2f, 0.1f};
	const matrix::Vector3f expected{-0.4f, 0.2f, 0.2f};

	const matrix::Vector3f ang_vel_error_b =
		AmPosControl::angularVelocityErrorForPolicy(desired_ang_vel_w, actual_ang_vel_w, root_quat, true);

	EXPECT_NEAR(ang_vel_error_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(ang_vel_error_b(2), expected(2), 1e-6f);
}

TEST(AmPosControlTest, GatedAttitudeErrorKeepsTiltedYawProjectionWhenYawCommandInactive)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.15f, -0.25f, 0.0f));
	const matrix::Quatf desired_quat(matrix::Eulerf(0.f, 0.f, 0.6f));
	const matrix::Quatf expected = root_quat.inversed() * desired_quat;

	const matrix::Quatf att_error_quat_b =
		AmPosControl::attitudeErrorQuatForPolicy(root_quat, desired_quat, false);

	EXPECT_NEAR(att_error_quat_b(0), expected(0), 1e-6f);
	EXPECT_NEAR(att_error_quat_b(1), expected(1), 1e-6f);
	EXPECT_NEAR(att_error_quat_b(2), expected(2), 1e-6f);
	EXPECT_NEAR(att_error_quat_b(3), expected(3), 1e-6f);
}

TEST(AmPosControlTest, AttitudeErrorIsIdentityWhenLevelAndYawRateActive)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.f, 0.f, 1.2f));
	const matrix::Quatf desired_quat(matrix::Eulerf(0.f, 0.f, -0.7f));

	const matrix::Dcmf att_error_dcm(AmPosControl::attitudeErrorQuatForPolicy(root_quat, desired_quat, true));

	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			EXPECT_NEAR(att_error_dcm(r, c), r == c ? 1.f : 0.f, 1e-6f);
		}
	}
}

TEST(AmPosControlTest, AttitudeErrorKeepsTiltCorrectionWhenYawRateActive)
{
	const matrix::Quatf root_quat(matrix::Eulerf(0.15f, -0.25f, 0.0f));
	const matrix::Quatf desired_quat(matrix::Eulerf(0.f, 0.f, 0.6f));

	const matrix::Dcmf att_error_dcm(AmPosControl::attitudeErrorQuatForPolicy(root_quat, desired_quat, true));

	EXPECT_GT(fabsf(att_error_dcm(0, 2)), 0.1f);
	EXPECT_GT(fabsf(att_error_dcm(1, 2)), 0.05f);
}

TEST(AmPosControlTest, AttitudeErrorIsContinuousNearPitchSingularityWhenYawRateActive)
{
	const matrix::Quatf desired_quat(matrix::Eulerf(0.f, 0.f, 0.f));
	const matrix::Quatf before(matrix::Eulerf(0.f, math::radians(89.9f), 0.f));
	const matrix::Quatf after(matrix::Eulerf(0.f, math::radians(90.1f), 0.f));
	const matrix::Dcmf before_dcm(AmPosControl::attitudeErrorQuatForPolicy(before, desired_quat, true));
	const matrix::Dcmf after_dcm(AmPosControl::attitudeErrorQuatForPolicy(after, desired_quat, true));
	float jump = 0.f;

	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			const float diff = after_dcm(r, c) - before_dcm(r, c);
			jump += diff * diff;
		}
	}

	EXPECT_LT(sqrtf(jump), 0.02f);
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

TEST(AmPosControlTest, DesiredYawIgnoresFiniteTrajectoryYawWhenManualYawInactive)
{
	EXPECT_FLOAT_EQ(AmPosControl::desiredYawForCommandReference(0.7f, 1.2f, -2.0f, false, false, false),
			0.7f);
}

TEST(AmPosControlTest, DesiredYawCapturesHeadingWhileYawRateActive)
{
	EXPECT_FLOAT_EQ(AmPosControl::desiredYawForCommandReference(0.7f, 1.2f, -2.0f, true, false, false),
			1.2f);
}

TEST(AmPosControlTest, DesiredYawCapturesHeadingOnYawRateRelease)
{
	EXPECT_FLOAT_EQ(AmPosControl::desiredYawForCommandReference(0.7f, 1.2f, -2.0f, false, true, false),
			1.2f);
}

TEST(AmPosControlTest, ManualYawRateActiveUsesDeadbandAndReleaseDelay)
{
	hrt_abstime release_start = 0;

	EXPECT_FALSE(AmPosControl::manualYawRateActive(0.03f, false, release_start, 1_s));
	EXPECT_EQ(release_start, 0u);

	EXPECT_TRUE(AmPosControl::manualYawRateActive(0.05f, false, release_start, 1_s));
	EXPECT_EQ(release_start, 0u);

	EXPECT_TRUE(AmPosControl::manualYawRateActive(0.03f, true, release_start, 1100_ms));
	EXPECT_EQ(release_start, 0u);

	EXPECT_TRUE(AmPosControl::manualYawRateActive(0.02f, true, release_start, 1200_ms));
	EXPECT_EQ(release_start, 1200_ms);
	EXPECT_TRUE(AmPosControl::manualYawRateActive(0.02f, true, release_start, 1300_ms));
	EXPECT_FALSE(AmPosControl::manualYawRateActive(0.02f, true, release_start, 1350_ms));
	EXPECT_EQ(release_start, 0u);
}

TEST(AmPosControlTest, OffboardYawRateActiveKeepsCommandZeroEpsSemantics)
{
	hrt_abstime release_start = 0;

	EXPECT_TRUE(AmPosControl::yawRateActiveForMode(0.03f, false, AmPosControl::ActiveMode::Offboard,
			release_start, 1_s));
	EXPECT_FALSE(AmPosControl::yawRateActiveForMode(0.03f, false, AmPosControl::ActiveMode::Manual,
			release_start, 1_s));
}

TEST(AmPosControlTest, DesiredYawRespectsFiniteTrajectoryYawWhenRequested)
{
	EXPECT_FLOAT_EQ(AmPosControl::desiredYawForCommandReference(0.7f, 1.2f, -2.0f, false, false, true),
			-2.0f);
}

TEST(AmPosControlTest, DesiredYawFallsBackToHeldYawWhenTrajectoryYawInvalid)
{
	EXPECT_FLOAT_EQ(AmPosControl::desiredYawForCommandReference(0.7f, 1.2f, NAN, false, false, true),
			0.7f);
}

TEST(AmPosControlTest, WorldYawRateOnlyActivatesYaw)
{
	EXPECT_TRUE(AmPosControl::commandActive(0.4f));
	EXPECT_FALSE(AmPosControl::commandActive(0.0f));
}

TEST(AmPosControlTest, ClampNormalizedMotorControlIsIdentityInsideActionRange)
{
	EXPECT_TRUE(std::isfinite(AmPosControl::clampNormalizedMotorControl(0.0f)));
	EXPECT_TRUE(std::isfinite(AmPosControl::clampNormalizedMotorControl(1.0f)));
	EXPECT_FLOAT_EQ(AmPosControl::clampNormalizedMotorControl(-0.25f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::clampNormalizedMotorControl(0.5f), 0.5f);
	EXPECT_FLOAT_EQ(AmPosControl::clampNormalizedMotorControl(1.25f), 1.0f);
}

TEST(AmPosControlTest, TakeoffRampScaleIsDiagnosticOnly)
{
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_SPOOLUP, 0.5f, 1.0f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_READY_FOR_TAKEOFF, 0.5f,
			1.0f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_RAMPUP, 0.0f, 1.0f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_RAMPUP, 0.25f, 1.0f),
			0.25f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_RAMPUP, 1.5f, 1.0f), 1.0f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_RAMPUP, NAN, 1.0f), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_RAMPUP, 0.5f, NAN), 0.0f);
	EXPECT_FLOAT_EQ(AmPosControl::takeoffRampOutputScale(takeoff_status_s::TAKEOFF_STATE_FLIGHT, 0.0f, 1.0f), 1.0f);
}

TEST(AmPosControlTest, MotorSetpointFromActionClampsRawActionWithoutRampScaling)
{
	const RlToolsAdapter::Action action{-0.5f, 0.5f, 1.5f, 0.25f};
	RlToolsAdapter::Action executed_action{};
	actuator_motors_s actuator_motors{};

	AmPosControl::fillMotorSetpointFromAction(actuator_motors, executed_action, action, 2000, 1500);

	EXPECT_EQ(actuator_motors.timestamp, 2000);
	EXPECT_EQ(actuator_motors.timestamp_sample, 1500);
	EXPECT_EQ(actuator_motors.reversible_flags, 0u);
	EXPECT_FLOAT_EQ(actuator_motors.control[0], 0.0f);
	EXPECT_FLOAT_EQ(actuator_motors.control[1], 0.5f);
	EXPECT_FLOAT_EQ(actuator_motors.control[2], 1.0f);
	EXPECT_FLOAT_EQ(actuator_motors.control[3], 0.25f);

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(executed_action[i], actuator_motors.control[i]);
	}

	for (int i = 4; i < 12; ++i) {
		EXPECT_TRUE(std::isnan(actuator_motors.control[i]));
	}
}

TEST(AmPosControlTest, PolicyStateCommitsOncePolicyOutputCanDriveMotors)
{
	EXPECT_FALSE(AmPosControl::takeoffStateAllowsPolicyStateCommit(takeoff_status_s::TAKEOFF_STATE_SPOOLUP));
	EXPECT_FALSE(AmPosControl::takeoffStateAllowsPolicyStateCommit(takeoff_status_s::TAKEOFF_STATE_READY_FOR_TAKEOFF));
	EXPECT_TRUE(AmPosControl::takeoffStateAllowsPolicyStateCommit(takeoff_status_s::TAKEOFF_STATE_RAMPUP));
	EXPECT_TRUE(AmPosControl::takeoffStateAllowsPolicyStateCommit(takeoff_status_s::TAKEOFF_STATE_FLIGHT));
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

TEST(AmPosControlTest, RampupWithGroundContactStillAllowsPolicyOutput)
{
	EXPECT_FALSE(AmPosControl::takeoffStateRequiresOutputGate(takeoff_status_s::TAKEOFF_STATE_RAMPUP, true));
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
