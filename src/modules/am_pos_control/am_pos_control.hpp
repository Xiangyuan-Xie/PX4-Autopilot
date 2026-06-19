/****************************************************************************
 *
 * AM Position direct actuator controller.
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
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/topics/actuator_motors.h>
#include <uORB/topics/am_pos_control_status.h>
#include <uORB/topics/am_policy_observation.h>
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

#include "am_policy_adapter.hpp"

using namespace time_literals;

class AmPosControl : public ModuleBase<AmPosControl>, public ModuleParams, public px4::WorkItem
{
public:
	AmPosControl();
	~AmPosControl() override;

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
		hrt_abstime offboard_control_mode_timestamp{0};
		hrt_abstime policy_inference_start_timestamp{0};
		hrt_abstime policy_inference_finish_timestamp{0};
		uint32_t policy_sequence{0};
		uint8_t takeoff_state{takeoff_status_s::TAKEOFF_STATE_UNINITIALIZED};
		float takeoff_ramp_scale{1.0f};
		float takeoff_ramped_speed_up{0.0f};
	};

	static void fillPolicyObservation(am_policy_observation_s &policy_observation,
					  const AmPolicyAdapter::Observation &observation,
					  const AmPolicyAdapter::Action &action,
					  const actuator_motors_s &actuator_motors, uint32_t degraded_flags,
					  const PolicyObservationTiming &timing)
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
		policy_observation.offboard_control_mode_timestamp = timing.offboard_control_mode_timestamp;
		policy_observation.policy_inference_start_timestamp = timing.policy_inference_start_timestamp;
		policy_observation.policy_inference_finish_timestamp = timing.policy_inference_finish_timestamp;
		policy_observation.policy_sequence = timing.policy_sequence;
		policy_observation.takeoff_state = timing.takeoff_state;
		policy_observation.takeoff_ramp_scale = timing.takeoff_ramp_scale;
		policy_observation.takeoff_ramped_speed_up = timing.takeoff_ramped_speed_up;
		policy_observation.degraded_flags = degraded_flags;

		for (int i = 0; i < AmPolicyAdapter::ObservationDim; ++i) {
			policy_observation.observation[i] = observation[i];
		}

		for (int i = 0; i < kActionDim; ++i) {
			policy_observation.raw_action[i] = action[i];
			policy_observation.mapped_action[i] = actuator_motors.control[i];
		}
	}

	static float wrapArmJointPositionForPolicy(float joint_position, uint32_t &wrap_mask, int joint_index)
	{
		if (!PX4_ISFINITE(joint_position)) {
			return joint_position;
		}

		if (joint_index == 4) {
			constexpr float kJoint5OpenAngleRad = 1.723f;
			return math::constrain(-joint_position / kJoint5OpenAngleRad, 0.0f, 1.0f);
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

	static void fillAmOffboardHoldSetpoint(trajectory_setpoint_s &setpoint, hrt_abstime now,
					       uint32_t &degraded_flags)
	{
		setpoint = {};
		setpoint.timestamp = now;

		for (int i = 0; i < 3; ++i) {
			setpoint.position[i] = NAN;
			setpoint.velocity[i] = 0.0f;
			setpoint.acceleration[i] = NAN;
			setpoint.jerk[i] = NAN;
		}

		setpoint.yaw = NAN;
		setpoint.yawspeed = 0.0f;
		degraded_flags |= am_policy_observation_s::DEGRADED_SETPOINT_DEFAULTED;
	}

	static bool amOffboardExternalSetpointUsable(const offboard_control_mode_s &offboard_control_mode,
			const trajectory_setpoint_s &setpoint, bool offboard_control_mode_valid, bool trajectory_setpoint_fresh)
	{
		if (!offboard_control_mode_valid || !trajectory_setpoint_fresh) {
			return false;
		}

		if (offboard_control_mode.velocity) {
			return PX4_ISFINITE(setpoint.velocity[0]) || PX4_ISFINITE(setpoint.velocity[1])
			       || PX4_ISFINITE(setpoint.velocity[2]);
		}

		return false;
	}

	static bool offboardControlModeSupported(const offboard_control_mode_s &offboard_control_mode)
	{
		const bool unsupported = offboard_control_mode.acceleration || offboard_control_mode.attitude
					 || offboard_control_mode.body_rate || offboard_control_mode.thrust_and_torque
					 || offboard_control_mode.direct_actuator;
		return offboard_control_mode.velocity && !unsupported;
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

	static float takeoffRampOutputScale(uint8_t takeoff_state, float ramped_speed_up, float target_speed_up)
	{
		if (takeoff_state >= takeoff_status_s::TAKEOFF_STATE_FLIGHT) {
			return 1.0f;
		}

		if (takeoff_state != takeoff_status_s::TAKEOFF_STATE_RAMPUP) {
			return 0.0f;
		}

		if (!PX4_ISFINITE(ramped_speed_up) || !PX4_ISFINITE(target_speed_up) || target_speed_up <= FLT_EPSILON) {
			return 0.0f;
		}

		return math::constrain(ramped_speed_up / target_speed_up, 0.0f, 1.0f);
	}

	static void fillMotorSetpointFromAction(actuator_motors_s &actuator_motors,
						AmPolicyAdapter::Action &executed_action,
						const AmPolicyAdapter::Action &action, hrt_abstime now,
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

	static float constrainUpwardVelocityNed(float velocity_z_ned, float ramped_speed_up)
	{
		if (!PX4_ISFINITE(velocity_z_ned)) {
			return 0.0f;
		}

		if (!PX4_ISFINITE(ramped_speed_up) || ramped_speed_up <= 0.0f) {
			return math::max(velocity_z_ned, 0.0f);
		}

		return math::max(velocity_z_ned, -ramped_speed_up);
	}

	static bool offboardSetpointWantsTakeoff(const trajectory_setpoint_s &setpoint,
			const vehicle_local_position_s &position, hrt_abstime timestamp_sample)
	{
		const bool setpoint_recent = timestamp_sample < setpoint.timestamp + 1_s;

		if (setpoint_recent && PX4_ISFINITE(setpoint.position[2]) && PX4_ISFINITE(position.z)
		    && (setpoint.position[2] < position.z)) {
			return true;
		}

		if (setpoint_recent && PX4_ISFINITE(setpoint.velocity[2]) && (setpoint.velocity[2] < 0.f)) {
			return true;
		}

		if (setpoint_recent && PX4_ISFINITE(setpoint.acceleration[2]) && (setpoint.acceleration[2] < 0.f)) {
			return true;
		}

		return false;
	}

	static bool shouldSkipTakeoffRampOnAmModeEntry(bool using_am_mode, bool was_using_am_mode, bool landed)
	{
		return using_am_mode && !was_using_am_mode && !landed;
	}

	static bool takeoffStateRequiresOutputGate(uint8_t takeoff_state, bool ground_contact)
	{
		const bool not_taken_off = takeoff_state < takeoff_status_s::TAKEOFF_STATE_RAMPUP;
		const bool flying_but_ground_contact = (takeoff_state >= takeoff_status_s::TAKEOFF_STATE_FLIGHT) && ground_contact;
		return not_taken_off || flying_but_ground_contact;
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

	static bool manualYawRateActive(float yawspeed, bool was_yaw_active, hrt_abstime &release_start,
					hrt_abstime now)
	{
		const float abs_yawspeed = fabsf(yawspeed);

		if (abs_yawspeed >= kManualYawActivateThreshold) {
			release_start = 0;
			return true;
		}

		if (!was_yaw_active) {
			release_start = 0;
			return false;
		}

		if (abs_yawspeed > kManualYawReleaseThreshold) {
			release_start = 0;
			return true;
		}

		if (release_start == 0) {
			release_start = now;
			return true;
		}

		if (now < release_start + kManualYawReleaseDelay) {
			return true;
		}

		release_start = 0;
		return false;
	}

	static bool yawRateActiveForMode(float yawspeed, bool was_yaw_active, ActiveMode mode,
					 hrt_abstime &manual_yaw_release_start, hrt_abstime now)
	{
		if (!PX4_ISFINITE(yawspeed)) {
			manual_yaw_release_start = 0;
			return false;
		}

		if (mode == ActiveMode::Manual) {
			return manualYawRateActive(yawspeed, was_yaw_active, manual_yaw_release_start, now);
		}

		manual_yaw_release_start = 0;
		return commandActive(yawspeed);
	}

	static void activeAxesFromCommand(const matrix::Vector3f &command, bool active[3])
	{
		for (int i = 0; i < 3; ++i) {
			active[i] = commandActive(command(i));
		}
	}

	static matrix::Vector3f desiredYawRateBodyFromNed(const matrix::Quatf &root_quat_w, float yawspeed_ned)
	{
		return root_quat_w.inversed().rotateVector(matrix::Vector3f{0.0f, 0.0f, -yawspeed_ned});
	}

	static float desiredYawForCommandReference(float previous_desired_yaw_w, float current_heading_w,
			float trajectory_yaw_w, bool yaw_active, bool was_yaw_active, bool respect_trajectory_yaw)
	{
		if (yaw_active || was_yaw_active) {
			return current_heading_w;
		}

		if (respect_trajectory_yaw && PX4_ISFINITE(trajectory_yaw_w)) {
			return trajectory_yaw_w;
		}

		return previous_desired_yaw_w;
	}

	static matrix::Vector3f gatePositionErrorForPolicy(const matrix::Vector3f &pos_error_w,
			const matrix::Quatf &root_quat_w, const matrix::Quatf &heading_quat_w, const bool active_lin_h[3])
	{
		matrix::Vector3f gated_pos_error_h = heading_quat_w.inversed().rotateVector(pos_error_w);

		for (int i = 0; i < 3; ++i) {
			if (active_lin_h[i]) {
				gated_pos_error_h(i) = 0.0f;
			}
		}

		const matrix::Vector3f gated_pos_error_w = heading_quat_w.rotateVector(gated_pos_error_h);
		return root_quat_w.inversed().rotateVector(gated_pos_error_w);
	}

	static matrix::Vector3f linearVelocityErrorForPolicy(const matrix::Vector3f &desired_vel_w,
			const matrix::Vector3f &actual_vel_w, const matrix::Quatf &root_quat_w,
			const matrix::Quatf &heading_quat_w, const bool active_lin_h[3])
	{
		const matrix::Vector3f desired_vel_h = heading_quat_w.inversed().rotateVector(desired_vel_w);
		matrix::Vector3f gated_desired_vel_h{};
		gated_desired_vel_h.zero();

		for (int i = 0; i < 3; ++i) {
			if (active_lin_h[i]) {
				gated_desired_vel_h(i) = desired_vel_h(i);
			}
		}

		const matrix::Vector3f gated_desired_vel_w = heading_quat_w.rotateVector(gated_desired_vel_h);
		return root_quat_w.inversed().rotateVector(gated_desired_vel_w - actual_vel_w);
	}

	static matrix::Vector3f angularVelocityErrorForPolicy(const matrix::Vector3f &desired_ang_vel_w,
			const matrix::Vector3f &actual_ang_vel_w, const matrix::Quatf &root_quat_w, bool yaw_active)
	{
		matrix::Vector3f desired_ang_vel_for_error_w{};
		desired_ang_vel_for_error_w.zero();

		if (yaw_active) {
			desired_ang_vel_for_error_w(2) = desired_ang_vel_w(2);
		}

		return root_quat_w.inversed().rotateVector(desired_ang_vel_for_error_w - actual_ang_vel_w);
	}

	static matrix::Quatf shortestArcQuat(matrix::Vector3f from_vec, matrix::Vector3f to_vec)
	{
		from_vec = from_vec.unit_or_zero();
		to_vec = to_vec.unit_or_zero();
		const float dot = math::constrain(from_vec.dot(to_vec), -1.0f, 1.0f);

		if (dot < -1.0f + 1e-6f) {
			return matrix::Quatf(0.0f, 1.0f, 0.0f, 0.0f);
		}

		const matrix::Vector3f cross = from_vec.cross(to_vec);
		matrix::Quatf quat(1.0f + dot, cross(0), cross(1), cross(2));
		quat.normalize();
		return quat;
	}

	static matrix::Quatf attitudeErrorQuatForPolicy(const matrix::Quatf &root_quat_w,
			const matrix::Quatf &desired_quat_w, bool yaw_active)
	{
		if (!yaw_active) {
			matrix::Quatf att_error_quat = root_quat_w.inversed() * desired_quat_w;
			att_error_quat.normalize();
			return att_error_quat;
		}

		const matrix::Vector3f projected_gravity_b = root_quat_w.inversed().rotateVector(matrix::Vector3f{0.0f, 0.0f, -1.0f});
		return shortestArcQuat(matrix::Vector3f{0.0f, 0.0f, -1.0f}, projected_gravity_b);
	}

private:
	static constexpr int kActionDim = 4;
	static constexpr int kArmJointDim = 5;
	static constexpr int kMotorControlDim = 12;
	static constexpr float kCmdZeroEps = 1e-6f;
	static constexpr float kManualYawActivateThreshold = 0.05f;
	static constexpr float kManualYawReleaseThreshold = 0.025f;
	static constexpr hrt_abstime kManualYawReleaseDelay = 150_ms;
	static constexpr int kStartupDiagSamples = 8;
	static constexpr float kDiagLowMeanCommand = 0.75f;
	static constexpr float kDiagLowMinCommand = 0.45f;
	static constexpr float kDiagWideSpread = 0.35f;
	static constexpr hrt_abstime kStateTimeout = 1_s;
	static constexpr hrt_abstime kArmStateTimeout = 200_ms;
	static constexpr hrt_abstime kTrajectorySetpointTimeout = 200_ms;

	struct CommandReference {
		matrix::Vector3f desired_lin_vel_w{};
		matrix::Vector3f desired_ang_vel_w{};
		matrix::Vector3f desired_pos_w{};
		matrix::Quatf desired_quat_w{};
		bool lin_cmd_active_h[3] {false, false, false};
		bool yaw_cmd_active{false};
		bool has_lin_vel_cmd{false};
		bool has_ang_vel_cmd{false};
		bool initialized{false};
	};

	void Run() override;
	void updateTargets();
	void updateTargets(bool respect_trajectory_yaw);
	void buildObservation(AmPolicyAdapter::Observation &observation);
	void applyAction(const AmPolicyAdapter::Observation &observation, const AmPolicyAdapter::Action &action,
			 AmPolicyAdapter::Action &executed_action, ActiveMode mode, bool publish_outputs,
			 uint32_t degraded_flags, const PolicyObservationTiming &timing);
	void publishPolicyObservation(const AmPolicyAdapter::Observation &observation, const AmPolicyAdapter::Action &action,
				      const actuator_motors_s &actuator_motors, uint32_t degraded_flags,
				      const PolicyObservationTiming &timing);
	void publishStopSetpoint();
	void publishIdleSetpoint();
	void publishTakeoffStatus();
	bool updateTakeoffGate(ActiveMode mode, bool was_using_am_mode, float dt_s);
	void updateActionHistory(const AmPolicyAdapter::Action &action);
	void resetActionHistory();
	void maybeLogPolicyDiagnostics(const AmPolicyAdapter::Observation &observation, const AmPolicyAdapter::Action &action,
				       const AmPolicyAdapter::Action &executed_action);
	void publishStatus();
	void resetCommandReference();
	void resetState();
	bool updateVehicleState(float &dt_s);
	void updateConvertedState();
	void updateNormalizedArmJointState();
	bool anyAxisActive(const bool axes[3]) const;
	bool armStateValid() const;
	bool vehicleStateValid() const;
	bool attitudeValid() const;
	bool angularVelocityValid() const;
	bool trajectorySetpointFresh() const;
	bool trajectorySetpointHasLinearInput() const;
	bool offboardTrajectorySetpointValid() const;
	bool trajectorySetpointValid() const;
	bool offboardControlModeFresh() const;
	bool offboardControlModeSupported() const;
	bool offboardControlModeValid() const;
	bool manualControlAvailable();
	ActiveMode activeMode();

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _offboard_control_mode_sub{ORB_ID(offboard_control_mode)};
	uORB::Subscription _vehicle_constraints_sub{ORB_ID(vehicle_constraints)};
	uORB::Subscription _vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
	uORB::Subscription _vehicle_land_detected_sub{ORB_ID(vehicle_land_detected)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _arm_joint_state_sub{ORB_ID(arm_joint_state)};
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};

	uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
	uORB::Publication<am_pos_control_status_s> _status_pub{ORB_ID(am_pos_control_status)};
	uORB::Publication<am_policy_observation_s> _policy_observation_pub{ORB_ID(am_policy_observation)};
	uORB::PublicationData<takeoff_status_s> _takeoff_status_pub{ORB_ID(takeoff_status)};
	uORB::Publication<vehicle_thrust_setpoint_s> _vehicle_thrust_setpoint_pub{ORB_ID(vehicle_thrust_setpoint)};

	perf_counter_t _loop_perf;
	hrt_abstime _last_run{0};
	hrt_abstime _last_offboard_diag{0};
	hrt_abstime _last_setpoint_diag{0};
	bool _use_am_mode{false};
	bool _am_offboard_using_external_setpoint{false};
	CommandReference _current_cmd_ref{};

	offboard_control_mode_s _offboard_control_mode{};
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
	float _normalized_arm_position[kArmJointDim] {};
	float _takeoff_ramped_speed_up{0.0f};
	float _takeoff_target_speed_up{0.0f};
	hrt_abstime _manual_yaw_release_start{0};
	int _startup_diag_samples_remaining{0};
	uint32_t _policy_sequence{0};
	uint32_t _arm_joint_state_position_wrap_mask{0};

	AmPolicyAdapter _adapter{};
	TakeoffHandling _takeoff{};
	Sticks _sticks{this};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::AMPC_Z_VEL_UP>) _param_ampc_z_vel_up,
		(ParamFloat<px4::params::COM_OF_LOSS_T>) _param_com_of_loss_t,
		(ParamFloat<px4::params::COM_SPOOLUP_TIME>) _param_com_spoolup_time,
		(ParamFloat<px4::params::MPC_TKO_RAMP_T>) _param_mpc_tko_ramp_t
	)
};
