#include <gtest/gtest.h>

#include "am_pose_control.hpp"

namespace
{

trajectory_setpoint_s emptyTrajectorySetpoint()
{
	trajectory_setpoint_s setpoint{};

	for (float &value : setpoint.position) {
		value = NAN;
	}

	for (float &value : setpoint.velocity) {
		value = NAN;
	}

	for (float &value : setpoint.acceleration) {
		value = NAN;
	}

	setpoint.yaw = NAN;
	setpoint.yawspeed = NAN;
	return setpoint;
}

} // namespace

TEST(AmPoseControlTest, PolicyScheduleIsHundredHertz)
{
	EXPECT_EQ(AmPoseControl::PolicyIntervalUs, 10'000);
	EXPECT_EQ(AmPoseControl::WatchdogIntervalUs, 100'000);
}

TEST(AmPoseControlTest, PolicyDeadlineUsesFreshSamplesWithoutCatchUp)
{
	hrt_abstime next_deadline = 0;

	EXPECT_FALSE(AmPoseControl::policyStepDue(0, next_deadline));
	EXPECT_TRUE(AmPoseControl::policyStepDue(1, next_deadline));
	EXPECT_EQ(next_deadline, 10'001);
	EXPECT_FALSE(AmPoseControl::policyStepDue(5'001, next_deadline));
	EXPECT_TRUE(AmPoseControl::policyStepDue(10'001, next_deadline));
	EXPECT_EQ(next_deadline, 20'001);

	// One fresh sample can cross several deadlines, but it still requests only one policy step.
	EXPECT_TRUE(AmPoseControl::policyStepDue(45'001, next_deadline));
	EXPECT_EQ(next_deadline, 50'001);
	EXPECT_FALSE(AmPoseControl::policyStepDue(45'002, next_deadline));
}

TEST(AmPoseControlTest, FillPolicyObservationUsesDeclaredContractAndSourceTiming)
{
	static_assert(AmPosePolicyAdapter::MaxObservationDim == 138, "AM Pose observation storage must be 138D");
	static_assert(sizeof(am_pose_policy_observation_s::observation) / sizeof(float) == 138,
		      "AM Pose uORB observation must retain both supported contracts");
	static_assert(am_pose_trajectory_s::MAX_POINTS == AmPoseTrajectory::kMaximumWaypointCount,
		      "trajectory message and controller capacities must match");
	static_assert(sizeof(am_pose_trajectory_s::points) / sizeof(am_pose_trajectory_s::points[0])
		      == AmPoseTrajectory::kMaximumWaypointCount,
		      "uORB trajectory point storage must have fixed AM Pose capacity");

	AmPosePolicyAdapter::Observation observation{};
	AmPosePolicyAdapter::Action action{0.1f, 0.2f, 0.3f, 0.4f};

	for (int i = 0; i < AmPosePolicyAdapter::MaxObservationDim; ++i) {
		observation[i] = static_cast<float>(i) * 0.5f;
	}

	actuator_motors_s actuator_motors{};
	actuator_motors.timestamp = 9000;
	actuator_motors.control[0] = 0.11f;
	actuator_motors.control[1] = 0.22f;
	actuator_motors.control[2] = 0.33f;
	actuator_motors.control[3] = 0.44f;

	AmPoseControl::PolicyObservationTiming timing{};
	timing.observation_build_timestamp = 8000;
	timing.vehicle_local_position_timestamp = 7000;
	timing.vehicle_local_position_timestamp_sample = 6900;
	timing.vehicle_attitude_timestamp = 7100;
	timing.vehicle_attitude_timestamp_sample = 6950;
	timing.vehicle_angular_velocity_timestamp = 7200;
	timing.vehicle_angular_velocity_timestamp_sample = 7000;
	timing.arm_joint_state_timestamp = 7300;
	timing.arm_joint_state_timestamp_sample = 7050;
	timing.arm_joint_state_sequence = 42;
	timing.arm_joint_state_position_wrap_mask = 3;
	timing.trajectory_setpoint_timestamp = 7400;
	timing.am_pose_trajectory_timestamp = 7450;
	timing.offboard_control_mode_timestamp = 7500;
	timing.policy_inference_start_timestamp = 8100;
	timing.policy_inference_finish_timestamp = 8200;
	timing.policy_sequence = 9;
	timing.takeoff_state = takeoff_status_s::TAKEOFF_STATE_FLIGHT;

	am_pose_policy_observation_s message{};
	AmPoseControl::fillPolicyObservation(message, observation, action, actuator_motors, 8u, timing,
				      AmPosePolicyAdapter::ObservationContract::Position);

	EXPECT_EQ(message.timestamp, actuator_motors.timestamp);
	EXPECT_EQ(message.timestamp_sample, timing.vehicle_angular_velocity_timestamp_sample);
	EXPECT_EQ(message.policy_sequence, timing.policy_sequence);
	EXPECT_EQ(message.arm_joint_state_sequence, timing.arm_joint_state_sequence);
	EXPECT_EQ(message.arm_joint_state_position_wrap_mask, 3u);
	EXPECT_EQ(message.am_pose_trajectory_timestamp, timing.am_pose_trajectory_timestamp);
	EXPECT_EQ(message.degraded_flags, 8u);
	EXPECT_EQ(message.takeoff_state, takeoff_status_s::TAKEOFF_STATE_FLIGHT);
	EXPECT_EQ(message.observation_dim, 134u);
	EXPECT_EQ(message.arm_observation_features, am_pose_policy_observation_s::ARM_OBSERVATION_POSITION);

	for (int i = 0; i < AmPosePolicyAdapter::Pose134ObservationDim; ++i) {
		EXPECT_FLOAT_EQ(message.observation[i], observation[i]);
	}

	for (int i = AmPosePolicyAdapter::Pose134ObservationDim; i < AmPosePolicyAdapter::MaxObservationDim; ++i) {
		EXPECT_FLOAT_EQ(message.observation[i], 0.f);
	}

	for (int i = 0; i < AmPosePolicyAdapter::ActionDim; ++i) {
		EXPECT_FLOAT_EQ(message.raw_action[i], action[i]);
		EXPECT_FLOAT_EQ(message.mapped_action[i], actuator_motors.control[i]);
	}
}

TEST(AmPoseControlTest, FillPolicyObservationRetainsTheFull138DContract)
{
	AmPosePolicyAdapter::Observation observation{};

	for (int i = 0; i < AmPosePolicyAdapter::MaxObservationDim; ++i) {
		observation[i] = static_cast<float>(i);
	}

	AmPosePolicyAdapter::Action action{};
	actuator_motors_s actuator_motors{};
	AmPoseControl::PolicyObservationTiming timing{};
	am_pose_policy_observation_s message{};
	AmPoseControl::fillPolicyObservation(message, observation, action, actuator_motors, 0u, timing,
				      AmPosePolicyAdapter::ObservationContract::PositionVelocity);

	EXPECT_EQ(message.observation_dim, 138u);
	EXPECT_EQ(message.arm_observation_features,
		  am_pose_policy_observation_s::ARM_OBSERVATION_POSITION
		  | am_pose_policy_observation_s::ARM_OBSERVATION_VELOCITY);
	EXPECT_FLOAT_EQ(message.observation[130], 130.f);
	EXPECT_FLOAT_EQ(message.observation[137], 137.f);
}

TEST(AmPoseControlTest, FillPolicyInputUsesPositionLayout)
{
	constexpr hrt_abstime now = 1_s;
	AmPoseTrajectory trajectory{};
	trajectory.reset({1.f, 2.f, -3.f}, 0.f, now);

	AmPoseControl::PoseObservationState state{};
	state.position_w = {1.f, 2.f, -3.f};
	state.attitude_w = matrix::Quatf{};
	state.linear_velocity_b = {0.1f, 0.2f, 0.3f};
	state.angular_velocity_b = {-0.1f, -0.2f, -0.3f};
	state.arm_position[0] = 1.f;
	state.arm_position[1] = 2.f;
	state.arm_position[2] = 3.f;
	state.arm_position[3] = 4.f;
	state.arm_velocity[0] = -1.f;
	state.arm_velocity[1] = -2.f;
	state.arm_velocity[2] = -3.f;
	state.arm_velocity[3] = -4.f;
	state.last_action[0] = 0.1f;
	state.last_action[1] = 0.2f;
	state.last_action[2] = 0.3f;
	state.last_action[3] = 0.4f;

	AmPosePolicyAdapter::Observation observation{};
	AmPoseControl::fillPolicyInput(observation, trajectory, state, now,
				      AmPosePolicyAdapter::ObservationContract::Position);

	for (int preview = 0; preview < AmPoseTrajectory::kPreviewCount; ++preview) {
		const int offset = preview * 12;
		EXPECT_FLOAT_EQ(observation[offset], 0.f);
		EXPECT_FLOAT_EQ(observation[offset + 1], 0.f);
		EXPECT_FLOAT_EQ(observation[offset + 2], 0.f);

		for (int row = 0; row < 3; ++row) {
			for (int column = 0; column < 3; ++column) {
				EXPECT_FLOAT_EQ(observation[offset + 3 + row * 3 + column], row == column ? 1.f : 0.f);
			}
		}
	}

	EXPECT_FLOAT_EQ(observation[120], 0.1f);
	EXPECT_FLOAT_EQ(observation[123], -0.1f);
	EXPECT_FLOAT_EQ(observation[126], 1.f);
	EXPECT_FLOAT_EQ(observation[129], 4.f);
	EXPECT_FLOAT_EQ(observation[130], 0.1f);
	EXPECT_FLOAT_EQ(observation[133], 0.4f);

	for (int index = AmPosePolicyAdapter::Pose134ObservationDim;
	     index < AmPosePolicyAdapter::MaxObservationDim; ++index) {
		EXPECT_FLOAT_EQ(observation[index], 0.f);
	}
}

TEST(AmPoseControlTest, FillPolicyInputUsesVelocityLayout)
{
	constexpr hrt_abstime now = 1_s;
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, now);

	AmPoseControl::PoseObservationState state{};
	state.attitude_w = matrix::Quatf{};
	state.arm_position[0] = 1.f;
	state.arm_position[1] = 2.f;
	state.arm_position[2] = 3.f;
	state.arm_position[3] = 4.f;
	state.arm_velocity[0] = -1.f;
	state.arm_velocity[1] = -2.f;
	state.arm_velocity[2] = -3.f;
	state.arm_velocity[3] = -4.f;
	state.last_action[0] = 0.1f;
	state.last_action[1] = 0.2f;
	state.last_action[2] = 0.3f;
	state.last_action[3] = 0.4f;

	AmPosePolicyAdapter::Observation observation{};
	AmPoseControl::fillPolicyInput(observation, trajectory, state, now,
				      AmPosePolicyAdapter::ObservationContract::Velocity);

	EXPECT_FLOAT_EQ(observation[126], -1.f);
	EXPECT_FLOAT_EQ(observation[129], -4.f);
	EXPECT_FLOAT_EQ(observation[130], 0.1f);
	EXPECT_FLOAT_EQ(observation[133], 0.4f);

	for (int index = AmPosePolicyAdapter::Pose134ObservationDim;
	     index < AmPosePolicyAdapter::MaxObservationDim; ++index) {
		EXPECT_FLOAT_EQ(observation[index], 0.f);
	}
}

TEST(AmPoseControlTest, FillPolicyInputUsesNoArmLayout)
{
	constexpr hrt_abstime now = 1_s;
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, now);

	AmPoseControl::PoseObservationState state{};
	state.attitude_w = matrix::Quatf{};
	state.arm_position[0] = 1.f;
	state.arm_velocity[0] = -1.f;
	state.last_action[0] = 0.1f;
	state.last_action[1] = 0.2f;
	state.last_action[2] = 0.3f;
	state.last_action[3] = 0.4f;

	AmPosePolicyAdapter::Observation observation{};
	AmPoseControl::fillPolicyInput(observation, trajectory, state, now,
				      AmPosePolicyAdapter::ObservationContract::NoArm);

	EXPECT_FLOAT_EQ(observation[126], 0.1f);
	EXPECT_FLOAT_EQ(observation[129], 0.4f);

	for (int index = AmPosePolicyAdapter::Pose130ObservationDim;
	     index < AmPosePolicyAdapter::MaxObservationDim; ++index) {
		EXPECT_FLOAT_EQ(observation[index], 0.f);
	}
}

TEST(AmPoseControlTest, FillPolicyInputUsesPositionVelocityLayout)
{
	constexpr hrt_abstime now = 1_s;
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, now);

	AmPoseControl::PoseObservationState state{};
	state.attitude_w = matrix::Quatf{};
	state.arm_position[0] = 1.f;
	state.arm_position[1] = 2.f;
	state.arm_position[2] = 3.f;
	state.arm_position[3] = 4.f;
	state.arm_velocity[0] = -1.f;
	state.arm_velocity[1] = -2.f;
	state.arm_velocity[2] = -3.f;
	state.arm_velocity[3] = -4.f;
	state.last_action[0] = 0.1f;
	state.last_action[1] = 0.2f;
	state.last_action[2] = 0.3f;
	state.last_action[3] = 0.4f;

	AmPosePolicyAdapter::Observation observation{};
	AmPoseControl::fillPolicyInput(observation, trajectory, state, now,
				      AmPosePolicyAdapter::ObservationContract::PositionVelocity);

	EXPECT_FLOAT_EQ(observation[126], 1.f);
	EXPECT_FLOAT_EQ(observation[129], 4.f);
	EXPECT_FLOAT_EQ(observation[130], -1.f);
	EXPECT_FLOAT_EQ(observation[133], -4.f);
	EXPECT_FLOAT_EQ(observation[134], 0.1f);
	EXPECT_FLOAT_EQ(observation[137], 0.4f);
}

TEST(AmPoseControlTest, OffboardAmPoseAcceptsPositionWhenIgnoredFieldsArePresent)
{
	offboard_control_mode_s mode{};
	mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;
	mode.position = true;
	trajectory_setpoint_s setpoint = emptyTrajectorySetpoint();
	setpoint.position[0] = 1.f;

	EXPECT_TRUE(AmPoseControl::offboardControlModeSupported(mode));
	EXPECT_TRUE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));

	mode.velocity = true;
	mode.acceleration = true;
	setpoint.velocity[0] = 0.5f;
	setpoint.acceleration[0] = 0.1f;
	EXPECT_TRUE(AmPoseControl::offboardControlModeSupported(mode));
	EXPECT_TRUE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));
}

TEST(AmPoseControlTest, OffboardAmPoseAcceptsVelocityOnlyAndKeepsPositionPriority)
{
	offboard_control_mode_s mode{};
	mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;
	mode.velocity = true;
	trajectory_setpoint_s setpoint = emptyTrajectorySetpoint();
	setpoint.velocity[0] = 0.5f;

	EXPECT_TRUE(AmPoseControl::offboardControlModeSupported(mode));
	EXPECT_TRUE(AmPoseControl::offboardVelocityModeSelected(mode));
	EXPECT_TRUE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));

	setpoint.velocity[0] = NAN;
	setpoint.yawspeed = NAN;
	EXPECT_TRUE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));

	mode.position = true;
	EXPECT_TRUE(AmPoseControl::offboardPositionModeSelected(mode));
	EXPECT_FALSE(AmPoseControl::offboardVelocityModeSelected(mode));
	EXPECT_FALSE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));

	setpoint.position[0] = 1.f;
	EXPECT_TRUE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));
}

TEST(AmPoseControlTest, OffboardAmPoseRejectsAccelerationOnlyAndBypassModes)
{
	offboard_control_mode_s mode{};
	mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;
	mode.acceleration = true;
	trajectory_setpoint_s setpoint = emptyTrajectorySetpoint();
	setpoint.acceleration[0] = 0.5f;

	EXPECT_FALSE(AmPoseControl::offboardControlModeSupported(mode));
	EXPECT_TRUE(AmPoseControl::offboardAmPoseInputRejected(mode, setpoint, true, true));

	mode.position = true;
	mode.attitude = true;
	setpoint.position[0] = 1.f;
	EXPECT_FALSE(AmPoseControl::offboardControlModeSupported(mode));

	mode.attitude = false;
	setpoint.position[0] = NAN;
	EXPECT_FALSE(AmPoseControl::offboardAmPoseExternalSetpointUsable(mode, setpoint, true, true));
}

TEST(AmPoseControlTest, OffboardAmPoseAllowsHoldWithoutACommand)
{
	offboard_control_mode_s mode{};
	mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;
	trajectory_setpoint_s setpoint = emptyTrajectorySetpoint();

	EXPECT_FALSE(AmPoseControl::offboardAmPoseInputRejected(mode, setpoint, false, false));
	EXPECT_FALSE(AmPoseControl::offboardAmPoseInputRejected(mode, setpoint, true, true));

	mode.acceleration = true;
	setpoint.acceleration[0] = 0.5f;
	EXPECT_TRUE(AmPoseControl::offboardAmPoseInputRejected(mode, setpoint, true, true));
}

TEST(AmPoseControlTest, OffboardVelocityLimitPreservesDirection)
{
	const matrix::Vector3f constrained = AmPoseControl::constrainVelocityCommand({3.f, 4.f, 0.f}, 1.f);
	EXPECT_NEAR(constrained.norm(), 1.f, 1e-6f);
	EXPECT_NEAR(constrained(0), 0.6f, 1e-6f);
	EXPECT_NEAR(constrained(1), 0.8f, 1e-6f);
	EXPECT_EQ(AmPoseControl::constrainVelocityCommand({NAN, 0.f, 0.f}, 1.f), matrix::Vector3f{});
}

TEST(AmPoseControlTest, OffboardVelocityCanRequestTakeoff)
{
	offboard_control_mode_s mode{};
	mode.controller_type = offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE;
	mode.velocity = true;
	trajectory_setpoint_s setpoint = emptyTrajectorySetpoint();
	setpoint.timestamp = 1_s;
	setpoint.velocity[2] = -0.5f;
	vehicle_local_position_s position{};
	position.z = 0.f;

	EXPECT_TRUE(AmPoseControl::offboardSetpointWantsTakeoff(mode, setpoint, position, 1100_ms));
	setpoint.velocity[2] = 0.5f;
	EXPECT_FALSE(AmPoseControl::offboardSetpointWantsTakeoff(mode, setpoint, position, 1100_ms));
}

TEST(AmPoseControlTest, ArmPositionsUsePerFramePiWrap)
{
	uint32_t wrap_mask = 0;
	EXPECT_FLOAT_EQ(AmPoseControl::prepareArmJointPositionForPolicy(3.12f, wrap_mask, 1), 3.12f);
	EXPECT_EQ(wrap_mask, 0u);
	EXPECT_NEAR(AmPoseControl::prepareArmJointPositionForPolicy(2.f * M_PI_F + 0.2f, wrap_mask, 2), 0.2f, 1e-6f);
	EXPECT_EQ(wrap_mask, 1u << 2);
}

TEST(AmPoseControlTest, ArmJointStateValidatesOnlyInputsRequiredByThePolicyContract)
{
	static_assert(arm_joint_state_s::MAX_JOINTS == 14, "ArmJointState must carry up to 14 joints");
	arm_joint_state_s state{};
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, false, false));

	for (int joint_count : {4, 7, 14}) {
		state.joint_count = joint_count;
		state.arm_velocity_valid = false;
		EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, true, false));
		EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, false, true));
		EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, true, true));
		state.arm_velocity_valid = true;
		EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, false, true));
		EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, true, true));
	}

	state.joint_count = 3;
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, true, false));
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, false, true));
	state.joint_count = 15;
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, true, false));
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, false, true));

	state.joint_count = 7;
	state.arm_position[0] = NAN;
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, true, false));
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, false, true));
	state.arm_position[0] = 0.f;
	state.arm_velocity[0] = NAN;
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, true, false));
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, false, true));
	state.arm_velocity[0] = 0.f;
	state.arm_velocity_valid = false;
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, true, false));
	EXPECT_FALSE(AmPoseControl::armJointStateValuesValid(state, false, true));
	state.arm_velocity_valid = true;
	state.arm_position[6] = NAN;
	state.arm_velocity[6] = INFINITY;
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, true, false));
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, false, true));
	EXPECT_TRUE(AmPoseControl::armJointStateValuesValid(state, true, true));
}

TEST(AmPoseControlTest, ObservationContractRequiresAnUnambiguousDimensionAndFeaturePair)
{
	using Adapter = AmPosePolicyAdapter;
	EXPECT_EQ(Adapter::observationContractFor(130, Adapter::ArmObservationNone),
		  Adapter::ObservationContract::NoArm);
	EXPECT_EQ(Adapter::observationContractFor(134, Adapter::ArmObservationPosition),
		  Adapter::ObservationContract::Position);
	EXPECT_EQ(Adapter::observationContractFor(134, Adapter::ArmObservationVelocity),
		  Adapter::ObservationContract::Velocity);
	EXPECT_EQ(Adapter::observationContractFor(138, Adapter::ArmObservationPosition | Adapter::ArmObservationVelocity),
		  Adapter::ObservationContract::PositionVelocity);
	EXPECT_EQ(Adapter::observationContractFor(130, Adapter::ArmObservationPosition),
		  Adapter::ObservationContract::Unsupported);
	EXPECT_EQ(Adapter::observationContractFor(134, Adapter::ArmObservationNone),
		  Adapter::ObservationContract::Unsupported);
	EXPECT_EQ(Adapter::observationContractFor(138, Adapter::ArmObservationVelocity),
		  Adapter::ObservationContract::Unsupported);
}

TEST(AmPoseControlTest, BlendStartsFromLastAppliedActionWithoutZeroFrame)
{
	const float start[4] {0.3f, 0.4f, 0.5f, 0.6f};
	const float requested[4] {0.8f, -1.f, 2.f, 0.2f};
	float blended[4] {};

	AmPoseControl::blendPolicyActions(start, requested, 0.f, blended);

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(blended[i], start[i]);
	}

	AmPoseControl::blendPolicyActions(start, requested, 1.f, blended);
	EXPECT_FLOAT_EQ(blended[0], 0.8f);
	EXPECT_FLOAT_EQ(blended[1], 0.f);
	EXPECT_FLOAT_EQ(blended[2], 1.f);
	EXPECT_FLOAT_EQ(blended[3], 0.2f);
}

TEST(AmPoseControlTest, AllocatorHandoverCopiesAllFourMotorsAtomically)
{
	actuator_motors_s motors{};
	motors.control[0] = 0.2f;
	motors.control[1] = 0.3f;
	motors.control[2] = 0.4f;
	motors.control[3] = 1.2f;
	float action[4] {-1.f, -1.f, -1.f, -1.f};

	EXPECT_TRUE(AmPoseControl::extractFiniteMotorAction(motors, action));
	EXPECT_FLOAT_EQ(action[0], 0.2f);
	EXPECT_FLOAT_EQ(action[1], 0.3f);
	EXPECT_FLOAT_EQ(action[2], 0.4f);
	EXPECT_FLOAT_EQ(action[3], 1.f);

	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, 1_s);
	AmPoseControl::PoseObservationState state{};

	for (int motor = 0; motor < 4; ++motor) {
		state.last_action[motor] = action[motor];
	}

	AmPosePolicyAdapter::Observation observation{};
	AmPoseControl::fillPolicyInput(observation, trajectory, state, 1_s,
				      AmPosePolicyAdapter::ObservationContract::Position);

	for (int motor = 0; motor < 4; ++motor) {
		EXPECT_FLOAT_EQ(observation[130 + motor], action[motor]);
	}

	motors.control[2] = NAN;
	EXPECT_FALSE(AmPoseControl::extractFiniteMotorAction(motors, action));
	EXPECT_FLOAT_EQ(action[0], 0.2f);
	EXPECT_FLOAT_EQ(action[1], 0.3f);
	EXPECT_FLOAT_EQ(action[2], 0.4f);
	EXPECT_FLOAT_EQ(action[3], 1.f);
}

TEST(AmPoseControlTest, AllocatorHandoverExpiresAfterFiftyMilliseconds)
{
	EXPECT_FALSE(AmPoseControl::handoverActionFreshAt(0, 10_ms));
	EXPECT_TRUE(AmPoseControl::handoverActionFreshAt(1_s, 1050_ms));
	EXPECT_FALSE(AmPoseControl::handoverActionFreshAt(1_s, 1050_ms + 1));
	EXPECT_FALSE(AmPoseControl::handoverActionFreshAt(1_s, 999_ms));
}

TEST(AmPoseControlTest, GroundEntryRemainsReadyAfterTakeoffWithoutPidHandover)
{
	EXPECT_TRUE(AmPoseControl::handoverActionReady(false, false, false));
	EXPECT_TRUE(AmPoseControl::handoverActionReady(true, true, false));
	EXPECT_TRUE(AmPoseControl::handoverActionReady(true, false, true));
	EXPECT_FALSE(AmPoseControl::handoverActionReady(true, false, false));
}

TEST(AmPoseControlTest, MotorSetpointClampsAndRecordsAppliedAction)
{
	const AmPosePolicyAdapter::Action requested{-0.5f, 0.5f, 1.5f, 0.25f};
	AmPosePolicyAdapter::Action executed{};
	actuator_motors_s motors{};

	AmPoseControl::fillMotorSetpointFromAction(motors, executed, requested, 2000, 1000);

	EXPECT_FLOAT_EQ(motors.control[0], 0.f);
	EXPECT_FLOAT_EQ(motors.control[1], 0.5f);
	EXPECT_FLOAT_EQ(motors.control[2], 1.f);
	EXPECT_FLOAT_EQ(motors.control[3], 0.25f);
	EXPECT_TRUE(isnan(motors.control[4]));

	for (int i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(executed[i], motors.control[i]);
	}
}

TEST(AmPoseControlTest, TakeoffReleaseLatchDoesNotRegateUntilModeExit)
{
	bool released = AmPoseControl::nextTakeoffOutputReleased(
				takeoff_status_s::TAKEOFF_STATE_RAMPUP, true, true, false);
	EXPECT_TRUE(released);
	EXPECT_FALSE(AmPoseControl::takeoffOutputGateActive(
			     takeoff_status_s::TAKEOFF_STATE_READY_FOR_TAKEOFF, released));

	released = AmPoseControl::nextTakeoffOutputReleased(
			   takeoff_status_s::TAKEOFF_STATE_READY_FOR_TAKEOFF, true, false, released);
	EXPECT_FALSE(released);
}

TEST(AmPoseControlTest, SetpointFreshnessUsesHeldAndLossWindows)
{
	trajectory_setpoint_s setpoint = emptyTrajectorySetpoint();
	setpoint.timestamp = 1_s;

	EXPECT_TRUE(AmPoseControl::trajectorySetpointFreshAt(setpoint, 1200_ms, 500_ms));
	EXPECT_FALSE(AmPoseControl::trajectorySetpointFreshAt(setpoint, 1600_ms, 500_ms));
	EXPECT_FALSE(AmPoseControl::trajectorySetpointLostAt(setpoint, 1600_ms, 1_s));
	EXPECT_TRUE(AmPoseControl::trajectorySetpointLostAt(setpoint, 2100_ms, 1_s));
}
