/****************************************************************************
 *
 * AM Pose direct actuator controller.
 *
 ****************************************************************************/
#pragma once

#include <lib/sticks/Sticks.hpp>
#include <mathlib/math/Functions.hpp>
#include <mathlib/mathlib.h>
#include <matrix/matrix/math.hpp>
#include <perf/perf_counter.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/SubscriptionInterval.hpp>
#include <uORB/topics/actuator_motors.h>
#include <uORB/topics/am_pose_control_status.h>
#include <uORB/topics/am_pose_policy_observation.h>
#include <uORB/topics/am_pose_trajectory.h>
#include <uORB/topics/arm_joint_state.h>
#include <uORB/topics/offboard_control_mode.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_constraints.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vehicle_thrust_setpoint.h>

#include <Takeoff.hpp>

#include "am_pose_policy_adapter.hpp"
#include "am_pose_trajectory.hpp"

using namespace time_literals;

class AmPoseControl : public ModuleBase<AmPoseControl>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	static constexpr hrt_abstime PolicyIntervalUs{10'000};
	static constexpr hrt_abstime WatchdogIntervalUs{100'000};

	static bool policyStepDue(hrt_abstime sample_timestamp, hrt_abstime &next_deadline)
	{
		if (sample_timestamp == 0) {
			return false;
		}

		if (next_deadline == 0) {
			next_deadline = sample_timestamp + PolicyIntervalUs;
			return true;
		}

		if (sample_timestamp < next_deadline) {
			return false;
		}

		const hrt_abstime elapsed = sample_timestamp - next_deadline;
		next_deadline += (elapsed / PolicyIntervalUs + 1) * PolicyIntervalUs;
		return true;
	}

	AmPoseControl();
	~AmPoseControl() override;

	static int task_spawn(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);
	int print_status() override;

	bool init();
	struct PolicyObservationTiming {
		hrt_abstime observation_build_timestamp{0};
		hrt_abstime vehicle_local_position_timestamp{0};
		hrt_abstime vehicle_local_position_timestamp_sample{0};
		hrt_abstime vehicle_attitude_timestamp{0};
		hrt_abstime vehicle_attitude_timestamp_sample{0};
		hrt_abstime vehicle_angular_velocity_timestamp{0};
		hrt_abstime vehicle_angular_velocity_timestamp_sample{0};
		hrt_abstime arm_joint_state_timestamp{0};
		hrt_abstime arm_joint_state_timestamp_sample{0};
		uint32_t arm_joint_state_sequence{0};
		uint32_t arm_joint_state_position_wrap_mask{0};
		hrt_abstime trajectory_setpoint_timestamp{0};
		hrt_abstime am_pose_trajectory_timestamp{0};
		hrt_abstime offboard_control_mode_timestamp{0};
		hrt_abstime policy_inference_start_timestamp{0};
		hrt_abstime policy_inference_finish_timestamp{0};
		uint32_t policy_sequence{0};
		uint8_t takeoff_state{takeoff_status_s::TAKEOFF_STATE_UNINITIALIZED};
	};

	struct PoseObservationState {
		matrix::Vector3f position_w{};
		matrix::Quatf attitude_w{};
		matrix::Vector3f linear_velocity_b{};
		matrix::Vector3f angular_velocity_b{};
		float arm_position[4] {};
		float arm_velocity[4] {};
		float last_action[4] {};
	};

	static void fillPolicyInput(AmPosePolicyAdapter::Observation &observation,
				    const AmPoseTrajectory &trajectory, const PoseObservationState &state,
				    hrt_abstime now, AmPosePolicyAdapter::ObservationContract contract)
	{
		static_assert(AmPoseTrajectory::kPreviewCount * 12 + 3 + 3
			      == AmPosePolicyAdapter::BaseObservationDim,
			      "AM Pose common observation layout mismatch");
		static_assert(AmPoseTrajectory::kPreviewCount * 12 + 3 + 3 + 4
			      == AmPosePolicyAdapter::Pose130ObservationDim,
			      "130D AM Pose observation layout mismatch");
		static_assert(AmPoseTrajectory::kPreviewCount * 12 + 3 + 3 + 4 + 4
			      == AmPosePolicyAdapter::Pose134ObservationDim,
			      "134D AM Pose observation layout mismatch");
		static_assert(AmPoseTrajectory::kPreviewCount * 12 + 3 + 3 + 4 + 4 + 4
			      == AmPosePolicyAdapter::Pose138ObservationDim,
			      "138D AM Pose observation layout mismatch");

		for (float &value : observation) {
			value = 0.f;
		}

		const matrix::Dcmf world_to_body(state.attitude_w.inversed());
		int index = 0;

		for (int preview = 0; preview < AmPoseTrajectory::kPreviewCount; ++preview) {
			const AmPoseTrajectory::Sample target = trajectory.sample(now, AmPoseTrajectory::previewOffset(preview));
			const matrix::Vector3f position_error_b = world_to_body * (target.position - state.position_w);
			const matrix::Dcmf rotation_error_b = world_to_body * target.attitude;

			for (int axis = 0; axis < 3; ++axis) {
				observation[index++] = position_error_b(axis);
			}

			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					observation[index++] = rotation_error_b(row, column);
				}
			}
		}

		for (int axis = 0; axis < 3; ++axis) {
			observation[index++] = state.linear_velocity_b(axis);
		}

		for (int axis = 0; axis < 3; ++axis) {
			observation[index++] = state.angular_velocity_b(axis);
		}

		if (AmPosePolicyAdapter::usesArmPosition(contract)) {
			for (float value : state.arm_position) {
				observation[index++] = value;
			}
		}

		if (AmPosePolicyAdapter::usesArmVelocity(contract)) {
			for (float value : state.arm_velocity) {
				observation[index++] = value;
			}
		}

		for (float value : state.last_action) {
			observation[index++] = value;
		}
	}

	static void fillPolicyObservation(am_pose_policy_observation_s &policy_observation,
					  const AmPosePolicyAdapter::Observation &observation,
					  const AmPosePolicyAdapter::Action &action,
					  const actuator_motors_s &actuator_motors, uint32_t degraded_flags,
					  const PolicyObservationTiming &timing,
					  AmPosePolicyAdapter::ObservationContract contract)
	{
		policy_observation = {};
		policy_observation.timestamp = actuator_motors.timestamp;
		policy_observation.timestamp_sample = timing.vehicle_angular_velocity_timestamp_sample;
		policy_observation.observation_build_timestamp = timing.observation_build_timestamp;
		policy_observation.vehicle_local_position_timestamp = timing.vehicle_local_position_timestamp;
		policy_observation.vehicle_local_position_timestamp_sample = timing.vehicle_local_position_timestamp_sample;
		policy_observation.vehicle_attitude_timestamp = timing.vehicle_attitude_timestamp;
		policy_observation.vehicle_attitude_timestamp_sample = timing.vehicle_attitude_timestamp_sample;
		policy_observation.vehicle_angular_velocity_timestamp = timing.vehicle_angular_velocity_timestamp;
		policy_observation.vehicle_angular_velocity_timestamp_sample = timing.vehicle_angular_velocity_timestamp_sample;
		policy_observation.arm_joint_state_timestamp = timing.arm_joint_state_timestamp;
		policy_observation.arm_joint_state_timestamp_sample = timing.arm_joint_state_timestamp_sample;
		policy_observation.arm_joint_state_sequence = timing.arm_joint_state_sequence;
		policy_observation.arm_joint_state_position_wrap_mask = timing.arm_joint_state_position_wrap_mask;
		policy_observation.trajectory_setpoint_timestamp = timing.trajectory_setpoint_timestamp;
		policy_observation.am_pose_trajectory_timestamp = timing.am_pose_trajectory_timestamp;
		policy_observation.offboard_control_mode_timestamp = timing.offboard_control_mode_timestamp;
		policy_observation.policy_inference_start_timestamp = timing.policy_inference_start_timestamp;
		policy_observation.policy_inference_finish_timestamp = timing.policy_inference_finish_timestamp;
		policy_observation.policy_sequence = timing.policy_sequence;
		policy_observation.takeoff_state = timing.takeoff_state;
		policy_observation.degraded_flags = degraded_flags;
		policy_observation.observation_dim = AmPosePolicyAdapter::observationDimFor(contract);
		policy_observation.arm_observation_features = AmPosePolicyAdapter::armObservationFeaturesFor(contract);

		for (int i = 0; i < AmPosePolicyAdapter::observationDimFor(contract); ++i) {
			policy_observation.observation[i] = observation[i];
		}

		for (int i = 0; i < kActionDim; ++i) {
			policy_observation.raw_action[i] = action[i];
			policy_observation.mapped_action[i] = actuator_motors.control[i];
		}
	}

	static bool extractFiniteMotorAction(const actuator_motors_s &actuator_motors, float action[4])
	{
		float candidate[4] {};

		for (int motor = 0; motor < kActionDim; ++motor) {
			if (!PX4_ISFINITE(actuator_motors.control[motor])) {
				return false;
			}

			candidate[motor] = clampNormalizedMotorControl(actuator_motors.control[motor]);
		}

		for (int motor = 0; motor < kActionDim; ++motor) {
			action[motor] = candidate[motor];
		}

		return true;
	}

	static bool handoverActionFreshAt(hrt_abstime timestamp, hrt_abstime now)
	{
		return timestamp != 0 && now >= timestamp && now <= timestamp + kHandoverActionTimeout;
	}

	static bool handoverActionReady(bool handover_required, bool entry_accepted, bool fresh_action)
	{
		return !handover_required || entry_accepted || fresh_action;
	}

	// Condition arm joint positions for the policy observation. This is not
	// mathematical normalization: joints 0-3 stay in radians and are wrapped
	// to [-pi, pi]. Non-finite inputs pass through so armStateValid() can reject them.
	static float prepareArmJointPositionForPolicy(float joint_position, uint32_t &wrap_mask, int joint_index)
	{
		if (!PX4_ISFINITE(joint_position)) {
			return joint_position;
		}

		if (joint_position > M_PI_F || joint_position < -M_PI_F) {
			wrap_mask |= 1u << joint_index;
			joint_position = fmodf(joint_position, 2.f * M_PI_F);

			if (joint_position > M_PI_F) {
				joint_position -= 2.f * M_PI_F;

			} else if (joint_position < -M_PI_F) {
				joint_position += 2.f * M_PI_F;
			}
		}

		return joint_position;
	}

	static bool armJointStateValuesValid(const arm_joint_state_s &arm_joint_state, bool require_arm_position,
					    bool require_arm_velocity)
	{
		if (!require_arm_position && !require_arm_velocity) {
			return true;
		}

		if ((arm_joint_state.joint_count < kPolicyArmJointDim)
		    || (arm_joint_state.joint_count > arm_joint_state_s::MAX_JOINTS)) {
			return false;
		}

		if (require_arm_velocity && !arm_joint_state.arm_velocity_valid) {
			return false;
		}

		for (int i = 0; i < kPolicyArmJointDim; ++i) {
			if (require_arm_position && !PX4_ISFINITE(arm_joint_state.arm_position[i])) {
				return false;
			}

			if (require_arm_velocity && !PX4_ISFINITE(arm_joint_state.arm_velocity[i])) {
				return false;
			}
		}

		return true;
	}

	static bool vehicleStateValidStrict(const vehicle_local_position_s &position, bool attitude_valid,
					    bool angular_velocity_valid, hrt_abstime now)
	{
		if ((position.timestamp == 0) || (now > position.timestamp + kStateTimeout)) {
			return false;
		}

		const bool xy_valid = position.xy_valid && PX4_ISFINITE(position.x) && PX4_ISFINITE(position.y);
		const bool z_valid = position.z_valid && PX4_ISFINITE(position.z);
		const bool v_xy_valid = position.v_xy_valid && PX4_ISFINITE(position.vx) && PX4_ISFINITE(position.vy);
		const bool v_z_valid = position.v_z_valid && PX4_ISFINITE(position.vz);
		const bool heading_valid = PX4_ISFINITE(position.heading);

		return xy_valid && z_valid && v_xy_valid && v_z_valid && heading_valid && attitude_valid && angular_velocity_valid;
	}

	static bool offboardControlModeSupported(const offboard_control_mode_s &offboard_control_mode)
	{
		const bool unsupported = offboard_control_mode.attitude || offboard_control_mode.body_rate
					 || offboard_control_mode.thrust_and_torque
					 || offboard_control_mode.direct_actuator;
		return (offboard_control_mode.position || offboard_control_mode.velocity) && !unsupported;
	}

	static bool offboardPositionModeSelected(const offboard_control_mode_s &offboard_control_mode)
	{
		return offboard_control_mode.position;
	}

	static bool offboardVelocityModeSelected(const offboard_control_mode_s &offboard_control_mode)
	{
		return !offboard_control_mode.position && offboard_control_mode.velocity;
	}

	static bool trajectorySetpointHasPosition(const trajectory_setpoint_s &setpoint)
	{
		return PX4_ISFINITE(setpoint.position[0]) || PX4_ISFINITE(setpoint.position[1])
		       || PX4_ISFINITE(setpoint.position[2]);
	}

	static bool trajectorySetpointHasLinearInput(const trajectory_setpoint_s &setpoint)
	{
		return PX4_ISFINITE(setpoint.position[0]) || PX4_ISFINITE(setpoint.position[1])
		       || PX4_ISFINITE(setpoint.position[2]) || PX4_ISFINITE(setpoint.velocity[0])
		       || PX4_ISFINITE(setpoint.velocity[1]) || PX4_ISFINITE(setpoint.velocity[2]);
	}

	static bool trajectorySetpointHasCommandInput(const trajectory_setpoint_s &setpoint)
	{
		return trajectorySetpointHasLinearInput(setpoint)
		       || PX4_ISFINITE(setpoint.acceleration[0]) || PX4_ISFINITE(setpoint.acceleration[1])
		       || PX4_ISFINITE(setpoint.acceleration[2]) || PX4_ISFINITE(setpoint.yaw)
		       || PX4_ISFINITE(setpoint.yawspeed);
	}

	static bool offboardAmPoseExternalSetpointUsable(const offboard_control_mode_s &offboard_control_mode,
			const trajectory_setpoint_s &setpoint, bool offboard_control_mode_valid, bool trajectory_setpoint_fresh)
	{
		if (!offboard_control_mode_valid || !trajectory_setpoint_fresh) {
			return false;
		}

		if (!offboardControlModeSupported(offboard_control_mode)) {
			return false;
		}

		return offboardVelocityModeSelected(offboard_control_mode)
		       || (offboardPositionModeSelected(offboard_control_mode) && trajectorySetpointHasPosition(setpoint));
	}

	static bool offboardAmPoseInputRejected(const offboard_control_mode_s &offboard_control_mode,
					    const trajectory_setpoint_s &setpoint, bool offboard_control_mode_fresh, bool trajectory_setpoint_fresh)
	{
		return offboard_control_mode_fresh && trajectory_setpoint_fresh
		       && trajectorySetpointHasCommandInput(setpoint) && !offboardControlModeSupported(offboard_control_mode);
	}

	static void fillThrustSetpointFromMotors(vehicle_thrust_setpoint_s &thrust_setpoint, hrt_abstime now,
			hrt_abstime timestamp_sample, const actuator_motors_s &actuator_motors)
	{
		float thrust_sum = 0.0f;
		int thrust_count = 0;

		for (int i = 0; i < kActionDim; ++i) {
			if (PX4_ISFINITE(actuator_motors.control[i])) {
				thrust_sum += actuator_motors.control[i];
				++thrust_count;
			}
		}

		thrust_setpoint = {};
		thrust_setpoint.timestamp = now;
		thrust_setpoint.timestamp_sample = timestamp_sample;
		thrust_setpoint.xyz[0] = 0.0f;
		thrust_setpoint.xyz[1] = 0.0f;
		thrust_setpoint.xyz[2] = thrust_count > 0 ? -(thrust_sum / static_cast<float>(thrust_count)) : 0.0f;
	}

	static void fillIdleMotorSetpoint(actuator_motors_s &actuator_motors, hrt_abstime now, hrt_abstime timestamp_sample)
	{
		actuator_motors = {};
		actuator_motors.timestamp = now;
		actuator_motors.timestamp_sample = timestamp_sample;

		for (int i = 0; i < kActionDim; ++i) {
			actuator_motors.control[i] = 0.0f;
		}

		for (int i = kActionDim; i < kMotorControlDim; ++i) {
			actuator_motors.control[i] = NAN;
		}

		actuator_motors.reversible_flags = 0;
	}

	static float clampNormalizedMotorControl(float action)
	{
		return math::constrain(action, 0.0f, 1.0f);
	}

	static matrix::Vector3f constrainVelocityCommand(const matrix::Vector3f &velocity, float maximum_velocity)
	{
		if (!velocity.isAllFinite()) {
			return {};
		}

		matrix::Vector3f constrained = velocity;
		const float speed = constrained.norm();
		const float speed_limit = math::max(maximum_velocity, 0.f);

		if (speed > speed_limit && speed > FLT_EPSILON) {
			constrained *= speed_limit / speed;
		}

		return constrained;
	}

	static void blendPolicyActions(const float start[4], const float requested[4], float alpha, float blended[4])
	{
		const float bounded_alpha = math::constrain(alpha, 0.f, 1.f);

		for (int motor = 0; motor < 4; ++motor) {
			blended[motor] = start[motor]
					 + bounded_alpha * (clampNormalizedMotorControl(requested[motor]) - start[motor]);
		}
	}

	template<typename ExecutedAction, typename RequestedAction>
	static void fillMotorSetpointFromAction(actuator_motors_s &actuator_motors,
						ExecutedAction &executed_action,
						const RequestedAction &action, hrt_abstime now,
						hrt_abstime timestamp_sample)
	{
		actuator_motors = {};
		actuator_motors.timestamp = now;
		actuator_motors.timestamp_sample = timestamp_sample;

		for (int i = 0; i < kActionDim; ++i) {
			const float clamped_action = clampNormalizedMotorControl(action[i]);
			actuator_motors.control[i] = clamped_action;
			executed_action[i] = clamped_action;
		}

		for (int i = kActionDim; i < kMotorControlDim; ++i) {
			actuator_motors.control[i] = NAN;
		}

		actuator_motors.reversible_flags = 0;
	}

	static bool offboardSetpointWantsTakeoff(const offboard_control_mode_s &offboard_control_mode,
			const trajectory_setpoint_s &setpoint,
			const vehicle_local_position_s &position, hrt_abstime timestamp_sample)
	{
		const bool setpoint_recent = timestamp_sample < setpoint.timestamp + 1_s;

		if (setpoint_recent && offboardControlModeSupported(offboard_control_mode)) {
			if (offboardPositionModeSelected(offboard_control_mode)) {
				return PX4_ISFINITE(setpoint.position[2]) && PX4_ISFINITE(position.z)
				       && setpoint.position[2] < position.z;
			}

			if (offboardVelocityModeSelected(offboard_control_mode)) {
				return PX4_ISFINITE(setpoint.velocity[2]) && setpoint.velocity[2] < -kCmdZeroEps;
			}
		}

		return false;
	}

	static bool trajectorySetpointFreshAt(const trajectory_setpoint_s &setpoint, hrt_abstime now, hrt_abstime timeout)
	{
		return (setpoint.timestamp != 0) && (now <= setpoint.timestamp + timeout);
	}

	static bool trajectorySetpointLostAt(const trajectory_setpoint_s &setpoint, hrt_abstime now, hrt_abstime loss_timeout)
	{
		return (setpoint.timestamp == 0) || (now > setpoint.timestamp + loss_timeout);
	}

	static bool shouldSkipTakeoffGateOnAmModeEntry(bool using_am_mode, bool was_using_am_mode, bool landed)
	{
		return using_am_mode && !was_using_am_mode && !landed;
	}

	static bool takeoffStateRequiresOutputGate(uint8_t takeoff_state)
	{
		return takeoff_state < takeoff_status_s::TAKEOFF_STATE_RAMPUP;
	}

	static bool nextTakeoffOutputReleased(uint8_t takeoff_state, bool armed, bool using_am_mode, bool released)
	{
		if (!armed || !using_am_mode) {
			return false;
		}

		return released || takeoff_state >= takeoff_status_s::TAKEOFF_STATE_RAMPUP;
	}

	static bool takeoffOutputGateActive(uint8_t takeoff_state, bool released)
	{
		return !released && takeoffStateRequiresOutputGate(takeoff_state);
	}

	static bool takeoffStateAllowsPolicyStateCommit(uint8_t takeoff_state)
	{
		return takeoff_state >= takeoff_status_s::TAKEOFF_STATE_RAMPUP;
	}

	static bool commandActive(float command)
	{
		return fabsf(command) > kCmdZeroEps;
	}

	enum class ActiveMode : uint8_t {
		None = 0,
		Manual,
		Offboard
	};

	enum class ReferenceSource : uint8_t {
		Hold = am_pose_control_status_s::REFERENCE_HOLD,
		Point = am_pose_control_status_s::REFERENCE_POINT,
		Trajectory = am_pose_control_status_s::REFERENCE_TRAJECTORY,
		Velocity = am_pose_control_status_s::REFERENCE_VELOCITY,
	};

private:
	static constexpr int kActionDim = 4;
	static constexpr int kPolicyArmJointDim = 4;
	static constexpr int kMotorControlDim = 12;
	static constexpr uint32_t kDegradedSetpointHeld = 1u << 3;
	static constexpr float kCmdZeroEps = 1e-6f;
	static constexpr hrt_abstime kStateTimeout = 1_s;
	static constexpr hrt_abstime kArmStateTimeout = 200_ms;
	static constexpr hrt_abstime kHandoverActionTimeout = 50_ms;
	static constexpr hrt_abstime kPoseReferenceUpdateInterval = 10_ms;
	static constexpr float kPoseTargetPositionEpsilon = 1e-3f;
	static constexpr float kPoseTargetYawEpsilon = math::radians(0.1f);

	template<typename Action>
	static bool actionFinite(const Action &action)
	{
		for (int motor = 0; motor < kActionDim; ++motor) {
			if (!PX4_ISFINITE(action[motor])) {
				return false;
			}
		}

		return true;
	}

	void Run() override;
	void buildObservation(AmPosePolicyAdapter::Observation &observation, hrt_abstime now);
	void updatePoseReference(ActiveMode mode, hrt_abstime now);
	void processTrajectoryCommand(hrt_abstime now, float maximum_velocity, float maximum_yaw_rate);
	void resetPoseReference(hrt_abstime now);
	void beginPolicyControl(ActiveMode mode, hrt_abstime now);
	void blendPolicyAction(const float requested[kActionDim], float blended[kActionDim], hrt_abstime now) const;
	void applyAction(const AmPosePolicyAdapter::Observation &observation,
			 const AmPosePolicyAdapter::Action &action, AmPosePolicyAdapter::Action &executed_action,
			 uint32_t degraded_flags, const PolicyObservationTiming &timing);
	void publishPolicyObservation(const AmPosePolicyAdapter::Observation &observation,
				      const AmPosePolicyAdapter::Action &action,
				      const actuator_motors_s &actuator_motors, uint32_t degraded_flags,
				      const PolicyObservationTiming &timing);
	void publishStopSetpoint();
	void publishIdleSetpoint();
	void publishTakeoffStatus();
	bool updateTakeoffGate(ActiveMode mode, bool was_using_am_mode, float dt_s);
	void updateActionHistory(const AmPosePolicyAdapter::Action &action);
	void resetActionHistory();
	bool handoverActionFresh(hrt_abstime now) const;
	void publishStatus();
	void resetState();
	bool updateVehicleState(float &dt_s);
	void updateConvertedState();
	void updatePolicyArmJointState();
	bool armStateValid() const;
	bool vehicleStateValid() const;
	bool attitudeValid() const;
	bool angularVelocityValid() const;
	bool trajectorySetpointFresh() const;
	bool trajectorySetpointLost() const;
	bool trajectorySetpointHasLinearInput() const;
	bool offboardTrajectorySetpointValid() const;
	bool trajectorySetpointValid(ActiveMode mode) const;
	bool offboardControlModeFresh() const;
	bool offboardControlModeSupported() const;
	bool offboardControlModeValid() const;
	void updateOffboardControlMode();
	bool manualControlAvailable();
	ActiveMode activeMode();

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _offboard_control_mode_sub{ORB_ID(offboard_control_mode)};
	uORB::Subscription _actuator_motors_sub{ORB_ID(actuator_motors)};
	uORB::Subscription _vehicle_constraints_sub{ORB_ID(vehicle_constraints)};
	uORB::Subscription _vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
	uORB::Subscription _vehicle_land_detected_sub{ORB_ID(vehicle_land_detected)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _am_pose_trajectory_sub{ORB_ID(am_pose_trajectory)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _arm_joint_state_sub{ORB_ID(arm_joint_state)};
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};

	uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
	uORB::Publication<am_pose_control_status_s> _status_pub{ORB_ID(am_pose_control_status)};
	uORB::Publication<am_pose_policy_observation_s> _policy_observation_pub{ORB_ID(am_pose_policy_observation)};
	uORB::PublicationData<takeoff_status_s> _takeoff_status_pub{ORB_ID(takeoff_status)};
	uORB::Publication<vehicle_thrust_setpoint_s> _vehicle_thrust_setpoint_pub{ORB_ID(vehicle_thrust_setpoint)};

	perf_counter_t _loop_perf;
	perf_counter_t _trajectory_reference_perf;
	perf_counter_t _trajectory_command_perf;
	hrt_abstime _next_policy_deadline{0};
	hrt_abstime _last_offboard_diag{0};
	hrt_abstime _last_setpoint_diag{0};
	bool _use_am_mode{false};
	bool _policy_loaded{false};

	offboard_control_mode_s _offboard_control_mode{};
	uint8_t _received_offboard_controller_type{offboard_control_mode_s::CONTROLLER_TYPE_NATIVE};
	hrt_abstime _received_offboard_control_mode_timestamp{0};
	vehicle_constraints_s _vehicle_constraints{0, NAN, NAN, false, {}};
	vehicle_control_mode_s _vehicle_control_mode{};
	vehicle_land_detected_s _vehicle_land_detected {
		.timestamp = 0,
		.freefall = false,
		.ground_contact = true,
		.maybe_landed = true,
		.landed = true,
	};
	trajectory_setpoint_s _trajectory_setpoint{};
	am_pose_trajectory_s _am_pose_trajectory{};
	vehicle_angular_velocity_s _angular_velocity{};
	vehicle_local_position_s _position{};
	vehicle_attitude_s _attitude{};
	arm_joint_state_s _arm_joint_state{};
	matrix::Vector3f _root_pos_w{};
	matrix::Quaternionf _root_quat_w{};
	matrix::Vector3f _root_lin_vel_w{};
	matrix::Vector3f _root_lin_vel_b{};
	matrix::Vector3f _root_ang_vel_b{};
	float _heading_w{0.0f};
	float _prev_action[kActionDim] {0.f, 0.f, 0.f, 0.f};
	float _blend_start_action[kActionDim] {0.f, 0.f, 0.f, 0.f};
	float _handover_action[kActionDim] {0.f, 0.f, 0.f, 0.f};
	hrt_abstime _handover_action_timestamp{0};
	bool _handover_action_valid{false};
	bool _handover_action_consumed{false};
	bool _handover_entry_accepted{false};
	float _policy_arm_position[kPolicyArmJointDim] {};
	float _policy_arm_velocity[kPolicyArmJointDim] {};
	uint32_t _policy_sequence{0};
	uint32_t _arm_joint_state_position_wrap_mask{0};
	bool _takeoff_output_released{false};
	ActiveMode _policy_mode{ActiveMode::None};
	hrt_abstime _blend_start{0};
	hrt_abstime _last_pose_reference_update{0};
	matrix::Vector3f _last_pose_target_position{};
	float _last_pose_target_yaw{0.f};
	bool _last_pose_target_valid{false};
	bool _manual_hold_anchor_active{false};
	bool _trajectory_message_valid{false};
	ReferenceSource _active_reference_source{ReferenceSource::Hold};
	uint32_t _active_trajectory_id{0};
	uint32_t _last_accepted_trajectory_id{0};
	uint8_t _active_trajectory_point_count{0};
	hrt_abstime _last_trajectory_message_timestamp{0};
	hrt_abstime _last_point_setpoint_timestamp{0};
	hrt_abstime _last_velocity_setpoint_timestamp{0};
	bool _has_accepted_trajectory_id{false};
	bool _offboard_velocity_hold_active{false};
	bool _offboard_velocity_setpoint_lost{false};
	float _active_trajectory_min_z_ned{NAN};
	bool _estimator_reset_counters_initialized{false};
	uint8_t _xy_reset_counter{0};
	uint8_t _z_reset_counter{0};
	uint8_t _heading_reset_counter{0};
	uint8_t _attitude_reset_counter{0};

	AmPosePolicyAdapter _adapter{};
	AmPoseTrajectory _pose_trajectory{};
	TakeoffHandling _takeoff{};
	Sticks _sticks{this};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::AMPC_VEL_MANUAL>) _param_ampc_vel_manual,
		(ParamFloat<px4::params::AMPC_Z_VEL_UP>) _param_ampc_z_vel_up,
		(ParamFloat<px4::params::COM_OF_LOSS_T>) _param_com_of_loss_t,
		(ParamFloat<px4::params::COM_SPOOLUP_TIME>) _param_com_spoolup_time,
		(ParamFloat<px4::params::AMPC_TKO_RAMP_T>) _param_ampc_tko_ramp_t,
		(ParamFloat<px4::params::AMPC_TRJ_TIMEOUT>) _param_ampc_trj_timeout,
		(ParamFloat<px4::params::AMPC_TRJ_LOSS_T>) _param_ampc_trj_loss_t,
		(ParamFloat<px4::params::AMPC_POSE_VMAX>) _param_ampc_pose_vmax,
		(ParamFloat<px4::params::AMPC_POSE_YMAX>) _param_ampc_pose_ymax,
		(ParamFloat<px4::params::AMPC_SW_BLEND_T>) _param_ampc_sw_blend_t,
		(ParamFloat<px4::params::AMPC_MAN_Y_MAX>) _param_ampc_man_y_max,
		(ParamFloat<px4::params::AMPC_MAN_ACC_HOR>) _param_ampc_man_acc_hor,
		(ParamFloat<px4::params::AMPC_MAN_JRK_HOR>) _param_ampc_man_jerk_hor,
		(ParamFloat<px4::params::AMPC_MAN_ACC_UP>) _param_ampc_man_acc_up,
		(ParamFloat<px4::params::AMPC_MAN_ACC_DN>) _param_ampc_man_acc_dn,
		(ParamFloat<px4::params::AMPC_MAN_JERK_Z>) _param_ampc_man_jerk_z,
		(ParamFloat<px4::params::AMPC_MAN_Y_ACC>) _param_ampc_man_y_acc,
		(ParamFloat<px4::params::AMPC_MAN_Y_JERK>) _param_ampc_man_y_jerk
	)
};
